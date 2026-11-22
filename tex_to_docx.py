"""Word version of the manuscript:  manuscript/jfm_draft.tex  ->  manuscript/jfm_draft.docx

The LaTeX source stays the master copy; this script only derives a Word file from it, for reading, commenting and
circulation.  It needs pandoc (3.x) and python-docx, and the files that a LaTeX run leaves in manuscript/
(jfm_draft.aux for the numbers of sections, equations, figures and tables; figures/*.png for the figures).

What it does
  1. takes the numbers of all labels from jfm_draft.aux and writes them into the text in place of \\ref and \\eqref;
     numbers the sections and the appendices in their headings as the JFM class does (2.1., Appendix A.);
  2. separates every displayed equation from the surrounding text and gives it its number (the numbers are computed
     here and checked against the .aux for every labelled equation);
  3. converts with pandoc: equations become editable Word equations, citations and the reference list are formatted by
     citeproc from jfm_refs.bib (author-year), figures (PNG) and tables keep their captions and are numbered by Word;
  4. puts each displayed equation in a borderless three-column table with its number at the right margin, and sets
     the fonts and the page (A4, Times New Roman).

Intermediate files go to tmp/docx_build/.
usage:  python tex_to_docx.py [--tex ../manuscript/jfm_draft.tex] [--out ../manuscript/jfm_draft.docx]
"""
import argparse
import copy
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
BRACED = r"((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)"

LUA = r"""
-- red text of the LaTeX source (the \todo placeholders) -> character style "Todo" of the reference document
function Span(el)
  local st = el.attributes["style"]
  if st and st:match("color") then
    el.attributes["style"] = nil
    el.attributes["custom-style"] = "Todo"
    return el
  end
end
"""


def read_aux(path):
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{" + BRACED + r"\}", open(path).read()):
        labels[m.group(1)] = m.group(2).replace("\\,", "\u00a0").strip()
    return labels


def preprocess(src, labels):
    """LaTeX source of the manuscript -> plain article that pandoc reads; returns it with the list of equation numbers."""
    title = re.search(r"\\title\{" + BRACED + r"\}", src).group(1)
    abstract = src[src.index("\\begin{abstract}") + len("\\begin{abstract}"):src.index("\\end{abstract}")].strip()
    # JFM keywords are chosen in the submission system and are not entered in the manuscript; kept only if present
    keywords = src[src.index("\\begin{keywords}") + len("\\begin{keywords}"):src.index("\\end{keywords}")].strip() if "\\begin{keywords}" in src else ""
    body = src[src.index("\\section{Introduction}"):src.index("\\IfFileExists{jfm.bst}")]

    state = dict(sec=0, sub=0, eq=0, app=False, letter="")
    numbers, checked = [], []

    def eqnum(content):
        state["eq"] += 1
        num = (f"{state['letter']}\u00a0{state['eq']}" if state["app"] else f"{state['sec']}.{state['eq']}")
        for lab in re.findall(r"\\label\{([^}]*)\}", content):
            if labels.get(lab) != num:
                raise SystemExit(f"equation {lab}: numbered {num} here, {labels.get(lab)} in the .aux file")
            checked.append(lab)
        numbers.append(num)
        clean = re.sub(r"\\label\{[^}]*\}", "", content).strip()
        return "\n\n\\begin{equation}\n" + clean + "\n\\end{equation}\n\n"

    def sub(m):
        t = m.group(0)
        if t == "\\begin{appen}":
            state.update(app=True, sec=0)
            return ""
        if t == "\\end{appen}":
            return ""
        if t.startswith("\\section"):
            state["sec"] += 1; state["sub"] = 0; state["eq"] = 0
            if state["app"]:
                state["letter"] = chr(ord("A") + state["sec"] - 1)
                return f"\\section{{Appendix {state['letter']}. {m.group('title')}}}"
            return f"\\section{{{state['sec']}. {m.group('title')}}}"
        if t.startswith("\\subsection"):
            state["sub"] += 1
            return f"\\subsection{{{state['sec']}.{state['sub']}. {m.group('title')}}}"
        if t.startswith("\\begin{equation}"):
            return eqnum(m.group("eq"))
        if t.startswith("\\begin{align}"):
            return "".join(eqnum(row.replace("&", "")) for row in re.split(r"\\\\", m.group("al")))
        raise ValueError(t[:40])

    pat = re.compile(r"\\begin\{appen\}|\\end\{appen\}|\\(?:sub)?section\{(?P<title>" + BRACED[1:-1] + r")\}(?:\\label\{[^}]*\})?"
                     r"|\\begin\{equation\}(?P<eq>.*?)\\end\{equation\}|\\begin\{align\}(?P<al>.*?)\\end\{align\}", re.S)
    body = pat.sub(sub, body)
    body = re.sub(r"\\begin\{bmhead\}\[(.*?)\]\s*(.*?)\\end\{bmhead\}", lambda m: f"\n\n\\textbf{{{m.group(1)}}} {m.group(2).strip()}\n\n", body, flags=re.S)

    def refs(text):
        text = re.sub(r"\\eqref\{([^}]*)\}", lambda m: "(" + labels[m.group(1)] + ")", text)
        text = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels[m.group(1)], text)
        return re.sub(r"\\label\{[^}]*\}", "", text)

    body, abstract = refs(body), refs(abstract)
    body = re.sub(r"\\includegraphics\[width=\\textwidth\]\{figures/(\w+)\}", r"\\includegraphics[width=1.0\\textwidth]{figures/\1.png}", body)
    authors = re.search(r"\\author\{(.*)\}\n", src).group(1)
    authors = re.sub(r"\\aff\{(\w+)\}", r"\\textsuperscript{\1}", authors).replace("\\and", "and")
    affs = re.search(r"\\affiliation\{(.*?)\}\n\\corresau", src, flags=re.S).group(1)
    affs = [m.strip() for m in re.split(r"\\aff\{(\w+)\}", affs)]               # ['', '1', 'text', '2', 'text', ...]
    aff = " \\\\ ".join(f"\\textsuperscript{{{n}}}{t}" for n, t in zip(affs[1::2], affs[2::2]))
    corr = re.search(r"\\corresau\{(.*)\}\n", src).group(1)
    names = re.sub(r"\\textsuperscript\{\w+\}", "", authors).replace(" and ", "; ").strip()
    out = "\n".join([
        "\\documentclass{article}", "\\usepackage{amsmath,amssymb,graphicx,booktabs,color,natbib}",
        "\\newcommand{\\todo}[1]{\\textcolor{red}{[#1]}}", "\\newcommand{\\dd}{\\mathrm{d}}", "\\newcommand{\\Hmin}{H_{\\min}}", "\\newcommand{\\Rey}{\\mathit{Re}}",
        "\\newcommand{\\Ar}{\\mathcal{A}}", f"\\title{{{title}}}", f"\\author{{{authors}}}",
        f"\\date{{{aff} \\\\ Corresponding author: {corr}}}", "\\begin{document}", "\\maketitle",
        "\\begin{abstract}", abstract, "\\end{abstract}", "", (f"\\textbf{{Key words:}} {keywords}" if keywords else ""), "", body, "\\end{document}", ""])
    print(f"sections and appendices numbered; {len(numbers)} displayed equations, {len(checked)} of them checked against the .aux file")
    return out, numbers, names


def citation_style(path):
    """pandoc's default author-year style (Chicago), adjusted towards the journal: '&', 'et al.' from three authors,
    no initials added to tell authors apart, no dash in place of a repeated author in the list."""
    csl = subprocess.run(["pandoc", "--print-default-data-file", "default.csl"], check=True, capture_output=True, text=True).stdout
    for a, b in (('et-al-min="4" et-al-use-first="1"', 'et-al-min="3" et-al-use-first="1"'),
                 ('disambiguate-add-names="true"', 'disambiguate-add-names="false"'),
                 ('disambiguate-add-givenname="true"', 'disambiguate-add-givenname="false"'),
                 (' subsequent-author-substitute="&#8212;&#8212;&#8212;"', ""),
                 ('demote-non-dropping-particle="display-and-sort"', 'demote-non-dropping-particle="sort-only"'),
                 ('page-range-format="chicago"', 'page-range-format="expanded"')):
        if a not in csl:
            raise SystemExit("the default citation style of pandoc has changed: " + a)
        csl = csl.replace(a, b)
    csl = csl.replace(' text-case="title"', "").replace(' quotes="true"', "")          # titles as they are in the .bib file
    csl = csl.replace('and="text"', 'and="symbol"').replace('<name form="short" and="symbol" delimiter=", "', '<name form="short" and="symbol" delimiter=", " delimiter-precedes-last="never"')
    open(path, "w").write(csl)


def reference_doc(path):
    """pandoc's default reference document with the fonts, the page and the few styles used here."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor
    from docx.enum.style import WD_STYLE_TYPE
    with open(path, "wb") as f:
        f.write(subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"], check=True, capture_output=True).stdout)
    d = Document(path)
    for s in d.sections:
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.left_margin = s.right_margin = s.top_margin = s.bottom_margin = Cm(2.5)
        s.header_distance = s.footer_distance = Cm(1.25); s.gutter = Cm(0)
    byname = {st.element.find(W + "name").get(W + "val"): st for st in d.styles}

    def style(name, size=None, bold=None, italic=None, align=None, before=None, after=None, keep=None, line=None):
        st = byname[name]
        st.font.name = "Times New Roman"
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(W + "rFonts")
        if rf is not None:
            for a in ("asciiTheme", "hAnsiTheme", "cstheme", "eastAsiaTheme"):
                rf.attrib.pop(W + a, None)
            rf.set(W + "cs", "Times New Roman")
        st.font.color.rgb = RGBColor(0, 0, 0)
        for a in ("themeColor", "themeTint", "themeShade"):
            rpr.find(W + "color").attrib.pop(W + a, None)
        if size: st.font.size = Pt(size)
        if bold is not None: st.font.bold = bold
        if italic is not None: st.font.italic = italic
        if st.type == WD_STYLE_TYPE.PARAGRAPH:
            pf = st.paragraph_format
            if align is not None: pf.alignment = align
            if before is not None: pf.space_before = Pt(before)
            if after is not None: pf.space_after = Pt(after)
            if keep is not None: pf.keep_with_next = keep
            if line is not None: pf.line_spacing = line

    J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
    style("Normal", 11)
    style("Body Text", 11, align=J, before=0, after=6, line=1.15)
    style("First Paragraph", 11, align=J, before=0, after=6, line=1.15)
    style("Compact", 9.5, before=0, after=0)
    style("Title", 17, bold=True, align=L, before=0, after=10)
    style("Author", 12, bold=True, align=L, after=4)
    style("Date", 10, align=L, after=10)
    style("Abstract", 10.5, align=J, before=0, after=8, line=1.15)
    style("Abstract Title", 10.5, bold=True, align=L)
    style("Heading 1", 12, bold=True, align=L, before=14, after=6, keep=True)
    style("Heading 2", 11, bold=False, italic=True, align=C, before=10, after=5, keep=True)
    style("Image Caption", 9.5, italic=False, align=J, before=4, after=12)
    style("Table Caption", 9.5, italic=False, align=J, before=10, after=4, keep=True)
    style("Caption", 9.5, italic=False)
    style("Captioned Figure", align=C, before=8, after=0, keep=True)
    style("Figure", align=C)
    style("Bibliography", 10, align=L, before=0, after=3)
    todo = d.styles.add_style("Todo", WD_STYLE_TYPE.CHARACTER)
    todo.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
    d.save(path)


def number_equations(path, numbers):
    """Each displayed equation goes into a borderless table: blank | equation | number at the right margin."""
    from docx import Document
    from docx.oxml import OxmlElement
    d = Document(path)
    body = d.element.body
    def el(tag, **attrs):
        e = OxmlElement("w:" + tag)
        for k, v in attrs.items():
            e.set(W + k, str(v))
        return e

    for tbl in body.iterchildren(W + "tbl"):                # the tables of the manuscript, before those of the equations
        pr = tbl.find(W + "tblPr")
        for old in pr.findall(W + "tblW") + pr.findall(W + "tblLayout") + pr.findall(W + "tblBorders"):
            pr.remove(old)
        pr.append(el("tblW", w=5000, type="pct"))
        bd = el("tblBorders")
        for side_ in ("top", "bottom"):
            bd.append(el(side_, val="single", sz=8, space=0, color="000000"))
        pr.append(bd); pr.append(el("tblLayout", type="autofit"))
        for tcw in list(tbl.iter(W + "tcW")):
            tcw.getparent().remove(tcw)
        for tr in tbl.iter(W + "tr"):
            trp = tr.find(W + "trPr")
            if trp is None:
                trp = el("trPr"); tr.insert(0, trp)
            if trp.find(W + "cantSplit") is None:
                trp.insert(0, el("cantSplit"))
    paras = [p for p in body.iterchildren(W + "p") if p.find(".//" + M + "oMathPara") is not None]
    if len(paras) != len(numbers):
        raise SystemExit(f"{len(paras)} displayed equations in the Word file, {len(numbers)} in the source")
    sect = body.find(W + "sectPr")
    width = int(sect.find(W + "pgSz").get(W + "w")) - int(sect.find(W + "pgMar").get(W + "left")) - int(sect.find(W + "pgMar").get(W + "right"))
    side = 880
    cols = (side, width - 2 * side, side)

    for p, num in zip(paras, numbers):
        if "".join(t.text or "" for t in p.iter(W + "t")).strip():
            raise SystemExit(f"equation ({num}) shares its paragraph with text")
        tbl = el("tbl")
        pr = el("tblPr"); pr.append(el("tblW", w=width, type="dxa")); pr.append(el("jc", val="center")); pr.append(el("tblLayout", type="fixed"))
        mar = el("tblCellMar")
        for side_ in ("left", "right"):
            mar.append(el(side_, w=0, type="dxa"))
        pr.append(mar); tbl.append(pr)
        grid = el("tblGrid")
        for c in cols:
            grid.append(el("gridCol", w=c))
        tbl.append(grid)
        tr = el("tr"); trp = el("trPr"); trp.append(el("cantSplit")); tr.append(trp)
        p.addprevious(tbl)
        for i, c in enumerate(cols):
            tc = el("tc"); tcp = el("tcPr"); tcp.append(el("tcW", w=c, type="dxa")); tcp.append(el("vAlign", val="center")); tc.append(tcp)
            if i == 1:
                tc.append(p)
            else:
                q = el("p"); qp = el("pPr"); sp = el("spacing", before=0, after=0); qp.append(sp); qp.append(el("jc", val="right")); q.append(qp)
                if i == 2:
                    r = el("r"); t = el("t"); t.text = f"({num})"; r.append(t); q.append(r)
                tc.append(q)
            tr.append(tc)
        tbl.append(tr)
        ppr = p.find(W + "pPr")
        if ppr is None:
            ppr = el("pPr"); p.insert(0, ppr)
        for old in ppr.findall(W + "spacing"):
            ppr.remove(old)
        ppr.append(el("spacing", before=60, after=60))
    d.save(path)
    return len(paras)


ORDER = {
    W + "style": "name aliases basedOn next link autoRedefine hidden uiPriority semiHidden unhideWhenUsed qFormat locked personal personalCompose personalReply rsid pPr rPr tblPr trPr tcPr tblStylePr",
    W + "pPr": "pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange",
    W + "tcPr": "cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark",
    W + "tblPr": "tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription",
    W + "trPr": "cnfStyle divId gridBefore gridAfter wBefore wAfter cantSplit trHeight tblHeader tblCellSpacing jc hidden",
    W + "settings": "writeProtection view zoom removePersonalInformation removeDateAndTime doNotDisplayPageBoundaries displayBackgroundShape printPostScriptOverText printFractionalCharacterWidth printFormsData embedTrueTypeFonts embedSystemFonts saveSubsetFonts saveFormsData mirrorMargins alignBordersAndEdges bordersDoNotSurroundHeader bordersDoNotSurroundFooter gutterAtTop hideSpellingErrors hideGrammaticalErrors activeWritingStyle proofState formsDesign attachedTemplate linkStyles stylePaneFormatFilter stylePaneSortMethod documentType mailMerge revisionView trackRevisions doNotTrackMoves doNotTrackFormatting documentProtection autoFormatOverride styleLockTheme styleLockQFSet defaultTabStop autoHyphenation consecutiveHyphenLimit hyphenationZone doNotHyphenateCaps showEnvelope summaryLength clickAndTypeStyle defaultTableStyle evenAndOddHeaders bookFoldRevPrinting bookFoldPrinting bookFoldPrintingSheets drawingGridHorizontalSpacing drawingGridVerticalSpacing displayHorizontalDrawingGridEvery displayVerticalDrawingGridEvery doNotUseMarginsForDrawingGridOrigin drawingGridHorizontalOrigin drawingGridVerticalOrigin doNotShadeFormData noPunctuationKerning characterSpacingControl printTwoOnOne strictFirstAndLastChars noLineBreaksAfter noLineBreaksBefore savePreviewPicture doNotValidateAgainstSchema saveInvalidXml ignoreMixedContent alwaysShowPlaceholderText doNotDemarcateInvalidXml saveXmlDataOnly useXSLTWhenSaving saveThroughXslt showXMLTags alwaysMergeEmptyNamespace updateFields hdrShapeDefaults footnotePr endnotePr compat docVars rsids mathPr attachedSchema themeFontLang clrSchemeMapping doNotIncludeSubdocsInStats doNotAutoCompressPictures forceUpgrade captions readModeInkLockDown smartTagType schemaLibrary shapeDefaults doNotEmbedSmartTags decimalSymbol listSeparator",
    M + "rPr": "lit nor scr sty brk aln",
    M + "dPr": "begChr sepChr endChr grow shp ctrlPr",
}


def tidy(path):
    """Put the children of a few elements in the order of the schema (pandoc does not always), and fix two details."""
    from docx import Document
    d = Document(path)
    roots = [d.element, d.styles.element, d.settings.element]
    try:
        num = d.part.numbering_part.element
        roots.append(num)
        for e in num.iter(W + "nsid"):
            e.set(W + "val", (e.get(W + "val") or "").rjust(8, "0").upper())
    except Exception:
        pass
    for rpr in d.element.iter(M + "rPr"):          # normal text takes no script or style
        if rpr.find(M + "nor") is not None:
            for k in rpr.findall(M + "scr") + rpr.findall(M + "sty"):
                rpr.remove(k)
    n = 0
    for root in roots:
        for tag, order in ORDER.items():
            ns = tag[:tag.index("}") + 1]
            rank = {ns + t: i for i, t in enumerate(order.split())}
            if tag == W + "settings":
                rank[M + "mathPr"] = rank.pop(W + "mathPr")
            for el in [root] if root.tag == tag else root.iter(tag):
                kids = list(el)
                if any(k.tag not in rank for k in kids):
                    known = [k for k in kids if k.tag in rank]
                else:
                    known = kids
                want = sorted(known, key=lambda k: rank[k.tag])
                if want != known:
                    slots = [i for i, k in enumerate(kids) if k.tag in rank]
                    for k in known:
                        el.remove(k)
                    for i, k in zip(slots, want):
                        el.insert(i, k)
                    n += 1
    d.save(path)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=os.path.join(HERE, "..", "manuscript", "jfm_draft.tex"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "manuscript", "jfm_draft.docx"))
    a = ap.parse_args()
    man = os.path.dirname(os.path.abspath(a.tex))
    build = os.path.join(HERE, "..", "tmp", "docx_build")
    os.makedirs(build, exist_ok=True)
    aux = os.path.splitext(a.tex)[0] + ".aux"
    if not os.path.exists(aux):
        sys.exit("run LaTeX on the manuscript first: the numbers are read from " + aux)
    pre, numbers, names = preprocess(open(a.tex).read(), read_aux(aux))
    tex = os.path.join(build, "jfm_draft_word.tex"); open(tex, "w").write(pre)
    lua = os.path.join(build, "todo.lua"); open(lua, "w").write(LUA)
    ref = os.path.join(build, "reference.docx"); reference_doc(ref)
    csl = os.path.join(build, "author_year.csl"); citation_style(csl)
    # a copy of the references in which name particles are kept in citations ("van Beek et al."), as in the journal
    bib = os.path.join(build, "refs_word.bib")
    open(bib, "w").write(re.sub(r"(@\w+\{[^,\s]+,)", r"\1\n  options = {useprefix=true},", open(os.path.join(man, "jfm_refs.bib")).read()))
    cmd = ["pandoc", tex, "-f", "latex", "-t", "docx+native_numbering", "--citeproc", "--bibliography", bib,
           "--csl", csl, "-M", "reference-section-title=References", "--reference-doc", ref, "--lua-filter", lua,
           "--resource-path", man, "-o", os.path.abspath(a.out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stderr.strip():
        print("pandoc:", r.stderr.strip())
    if r.returncode:
        sys.exit(r.returncode)
    n = number_equations(os.path.abspath(a.out), numbers)
    from docx import Document
    d = Document(os.path.abspath(a.out))            # document properties: the authors' names, without affiliation marks
    d.core_properties.author = names; d.core_properties.last_modified_by = ""; d.core_properties.comments = ""
    d.save(os.path.abspath(a.out))
    tidy(os.path.abspath(a.out))
    print(f"written {os.path.abspath(a.out)} ({n} numbered equations)")


if __name__ == "__main__":
    main()
