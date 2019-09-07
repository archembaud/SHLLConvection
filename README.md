# SHLLConvection

3D and 2D Convection simulation with compressible Euler equations.

## Breakdown

There is a two-dimensional proof of concept, and a GPU (CUDA) powered 3D solver, which is still in progress.

### 2D Demo

The code is set up to simulate a square box with a heated wall. All conditions are normalized. Gravity is hard coded into the Y direction in this code and arbitrarily normalized and set; you can experiment with the amount of gravity by changing the variable G.

The results expected after 10k steps are shown in the pictures included with the source code.

### Render

This folder contains the python script which generates the input file for the 3D GPU solver. This file reads a json file containing a description of the geometry being simulated, and some constants.

### Solver

The CUDA 3D solver code. Work in progress; for the hackathon.
