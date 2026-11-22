"""Build the Computers and Geotechnics (Elsevier, elsarticle) copy of the manuscript in submit_CG/.
The JFM master copy manuscript/jfm_draft.tex is not modified."""
import re, shutil, pathlib
root = pathlib.Path(__file__).resolve().parents[1]
src = (root/'manuscript/jfm_draft.tex').read_text()
out = root/'submit_CG'
(out/'figures').mkdir(parents=True, exist_ok=True)

# body: from the Introduction to the end of the conclusions; appendices separately
body = src[src.index(r'\section{Introduction}'):src.index(r'\begin{bmhead}[Funding.]')]
app = src[src.index(r'\begin{appen}')+len(r'\begin{appen}'):src.index(r'\end{appen}')]
abstract = src[src.index(r'\begin{abstract}')+len(r'\begin{abstract}'):src.index(r'\end{abstract}')].strip()

def elsevier_refs(t):
    t = t.replace(r'\S\S\,\ref', r'Sections~\ref')
    t = re.sub(r'\\S\\,\\ref', r'Section~\\ref', t)
    t = t.replace(r'appendix~\ref', r'Appendix~\ref')
    t = t.replace(r'table~\ref', r'Table~\ref')
    t = t.replace(r'figure~\ref', r'Fig.~\ref')
    return t
body, app = elsevier_refs(body), elsevier_refs(app)
# two tables are wider than the elsarticle text block with author-year citations
for old, new in [(r"\centering\small" "\n" r"\begin{tabular}{@{}llllp{0.29\textwidth}@{}}",
                  r"\centering\footnotesize\setlength{\tabcolsep}{4pt}" "\n" r"\begin{tabular}{@{}llllp{0.27\textwidth}@{}}"),
                 (r"\centering\small" "\n" r"\begin{tabular}{@{}lp{0.50\textwidth}l@{}}",
                  r"\centering\footnotesize" "\n" r"\begin{tabular}{@{}lp{0.47\textwidth}l@{}}")]:
    assert body.count(old) == 1, old
    body = body.replace(old, new)
assert r'\S' not in body+app, 'section sign left'

head = r'''%% Backward erosion pipes at the threshold of motion -- manuscript for Computers and Geotechnics.
%% Elsevier article class (elsarticle), author-year references (elsarticle-harv), as required by the journal.
%% elsarticle.cls and elsarticle-harv.bst are part of TeX Live, MiKTeX and Overleaf.
\documentclass[preprint,12pt,authoryear]{elsarticle}
\usepackage{amsmath,amssymb,graphicx,booktabs}
\usepackage{lineno}
\newcommand{\dd}{\mathrm{d}}
\newcommand{\Hmin}{H_{\min}}
\newcommand{\Ar}{\mathcal{A}}
\newcommand{\Rey}{\mathit{Re}}

\journal{Computers and Geotechnics}

\begin{document}
\linenumbers

\begin{frontmatter}

\title{Backward erosion pipes at the threshold of motion}

\author[tud]{Bo-Rong Peng\corref{cor1}}
\ead{B.PENG-1@tudelft.nl}
\author[csu]{Ya-Ge Wu}
\cortext[cor1]{Corresponding author.}

\affiliation[tud]{organization={Faculty of Civil Engineering and Geosciences, Delft University of Technology},
            city={Delft},
            postcode={2628CN},
            country={The Netherlands}}
\affiliation[csu]{organization={School of Civil Engineering, Central South University},
            city={Changsha},
            postcode={410075},
            country={China}}

\begin{abstract}
''' + abstract + r'''
\end{abstract}

\begin{keyword}
Backward erosion piping \sep internal erosion \sep threshold of motion \sep seepage \sep free-boundary problem
\sep bifurcation
\end{keyword}

\end{frontmatter}

'''
endmatter = r'''
\section*{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have
appeared to influence the work reported in this paper.

\section*{Funding}
This work was supported by the research programme Digital Dikes: a multiscale framework for risk assessment of backward
erosion piping, funded through the Perspectief programme of the Dutch Research Council (NWO).

\section*{Data availability}
The code that produces every figure and number in this paper is available from the corresponding author on reasonable
request. The experimental data are those of \citet{pol2022}, \citet{pol2022data}, \citet{vanbeek2015} and
\citet{pol2020data}.

\section*{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process}
During the preparation of this work, the authors used Claude (Anthropic; models Claude Fable 5.2 and Claude Opus 5.5,
accessed through claude.ai in November 2026) in order to develop the derivations of Sections~\ref{sec:section}--\ref{sec:inertia}
and of the appendices, to write the computer code for the numerical solutions, for the processing of the published
experimental records and for the figures, to search for and check the references, and to draft and edit the text.
After using this tool, the authors reviewed and edited the content as needed, checked the derivations, the code and
the references, and take full responsibility for the content of the published article.

\appendix
'''
tail = r'''
\bibliographystyle{elsarticle-harv}
\bibliography{refs}

\end{document}
'''
tex = head + body + endmatter + app.strip() + '\n' + tail
# figures are read from figures/, as in the master copy
(out/'manuscript_CG.tex').write_text(tex)
shutil.copy(root/'manuscript/jfm_refs.bib', out/'refs.bib')
for f in sorted((root/'manuscript/figures').glob('fig0*.pdf')):
    shutil.copy(f, out/'figures'/f.name)
print('written', out/'manuscript_CG.tex', len(tex.split('\n')), 'lines;',
      'figures:', [f.name for f in sorted((out/'figures').glob('*.pdf'))])
x = re.sub(r'\$[^$]*\$', 'X', abstract); x = re.sub(r'\\[A-Za-z]+', '', x)
print('abstract words:', len(x.split()))
