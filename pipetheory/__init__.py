"""pipetheory: threshold theory of backward-erosion pipes (NumPy/SciPy, no torch needed).

section.py   cross-section of a laminar erosion pipe under a rigid roof whose bed grains are
             everywhere at the threshold of motion (lubrication formulae + full Stokes solution)
aquifer.py   3-D Darcy flow in a rectangular sand box with a pipe on its roof (response matrix)
coupled.py   pipe law + aquifer response: head needed for a pipe of given length to transport
             sediment (fold of the coupled problem), profiles along the pipe
"""
RHO_W = 1000.0
G = 9.81
NU = 1.0e-6
