# 17. References

Every model of the simulator comes from a source that can be read and checked.  The key
in brackets is how the other documents and the code docstrings cite it; the last column
says where it is used.

## Heat transfer, materials and correlations

| Key | Reference | Used for (module) |
|---|---|---|
| [Incropera] | F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine, *Fundamentals of Heat and Mass Transfer*, 7th ed., Wiley, 2011. Tables A.3 (solids), A.4 (air), A.6 (water vapour); §3.3 (radial conduction); Table 4.1 (conduction shape factors); §8.5 (Dittus-Boelter), §8.4 (laminar Nu = 3.66) | gas properties `src/solver/fluid.py::property_shape`; quartz grain of silica sand `src/core/materials.py`; the shape factor of `tests/test_well_model.py`; `pipe_h` |
| [VDI-D6.3] | E. Tsotsas, "D6.3 Thermal conductivity of packed beds", in *VDI Heat Atlas*, 2nd ed., Springer, 2010 | packed-bed conductivity `src/core/materials.py::PackedBed` |
| [ZS70] | P. Zehner, E. U. Schlünder, "Wärmeleitfähigkeit von Schüttungen bei mäßigen Temperaturen", *Chem. Ing. Tech.* 42 (1970) 933-941 | the unit-cell model behind [VDI-D6.3] |
| [BB80] | G. Breitbach, H. Barthels, "The radiant heat transfer in the high temperature reactor core after failure of the afterheat removal systems", *Nucl. Technol.* 49 (1980) 392-399 | the radiative term `k_rad = 4 sigma T^3 d / (2/eps - 1)` of the bed |
| [VDI4640] | VDI 4640 Part 1, *Thermal use of the underground*, 2010, Table 1 | soil under the foundation (moist sand, 1.5 W/(m K)) `src/core/materials.py` |
| [ISO6946] | ISO 6946:2017, *Building components - Thermal resistance and thermal transmittance*, Annex C (external surface coefficient 4 + 4 v) | wind part of the outer film `src/core/environment.py` |
| [CC75] | S. W. Churchill, H. H. S. Chu, "Correlating equations for laminar and turbulent free convection from a vertical plate", *Int. J. Heat Mass Transfer* 18 (1975) 1323-1329 | natural convection on the vessel `src/core/environment.py` |
| [Haaland] | S. E. Haaland, "Simple and explicit formulas for the friction factor in turbulent pipe flow", *J. Fluids Eng.* 105 (1983) 89-90 | `src/solver/fluid.py::friction_factor` |
| [Idelchik] | I. E. Idelchik, *Handbook of Hydraulic Resistance*, 4th ed., Begell House, 2007 | pipe roughness of drawn and commercial steel `src/core/pipe_network.py::PIPE_MATERIALS` |

## The tube in the bed

| Key | Reference | Used for (module) |
|---|---|---|
| [Pea78] | D. W. Peaceman, "Interpretation of well-block pressures in numerical reservoir simulation", *SPE J.* 18 (1978) 183-194 | the equivalent radius `r_eq = 0.198 h` of a line source in a square cell `src/solver/fluid.py` (well model) |
| [Pea83] | D. W. Peaceman, "Interpretation of well-block pressures in numerical reservoir simulation with nonsquare grid blocks and anisotropic permeability", *SPE J.* 23 (1983) 531-543 | `r_eq = 0.14 sqrt(a^2 + b^2)` for an `a x b` cell (the flat and tall leaves of the box tree) |
| [AS79] | K. Aziz, A. Settari, *Petroleum Reservoir Simulation*, Applied Science, 1979, ch. 4 and 8 | layered (2.5-D) grids and the well index as the standard way to put a line source in a finite-volume grid |
| [Kays] | W. M. Kays, A. L. London, *Compact Heat Exchangers*, 3rd ed., McGraw-Hill, 1984 | the effectiveness-NTU relation of a pipe at a wall temperature `src/solver/fluid.py` |

## Numerics

| Key | Reference | Used for (module) |
|---|---|---|
| [Patankar] | S. V. Patankar, *Numerical Heat Transfer and Fluid Flow*, Hemisphere, 1980, ch. 4 | cell-centred finite volume, the harmonic mean at interfaces, the half-cell film `src/solver/matrix.py`, `src/core/physics.py` |
| [BWG11] | C. Burstedde, L. C. Wilcox, O. Ghattas, "p4est: Scalable algorithms for parallel adaptive mesh refinement on forests of octrees", *SIAM J. Sci. Comput.* 33 (2011) 1103-1133 | the linear octree (integer corners, per-level codes), the 2:1 balance `src/core/octree.py`, `src/core/box_tree.py` |
| [RS87] | J. W. Ruge, K. Stüben, "Algebraic multigrid", in *Multigrid Methods*, SIAM Frontiers in Applied Mathematics 3, 1987, 73-130 | the AMG preconditioner (through PyAMG) `src/solver/linear.py` |
| [PyAMG] | N. Bell, L. N. Olson, J. Schroder, B. Southworth, "PyAMG: Algebraic multigrid solvers in Python", *J. Open Source Softw.* 8 (2023) 5495 | the implementation `pyamg.ruge_stuben_solver` |
| [Saad] | Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003, §6.7 (CG), §9 (preconditioning) | CG on the symmetrised operator; a preconditioner only has to be close to the operator (the AMG reuse) |
| [Roache] | P. J. Roache, *Verification and Validation in Computational Science and Engineering*, Hermosa, 1998 | the Grid Convergence Index `src/analysis/convergence.py` |
| [Celik08] | I. B. Celik et al., "Procedure for estimation and reporting of uncertainty due to discretization in CFD applications", *J. Fluids Eng.* 130 (2008) 078001 | the three-grid procedure of the automatic mesh |

## The plant

| Key | Reference | Used for |
|---|---|---|
| [PNE] | Polar Night Energy, published descriptions of the Kankaanpää pilot (2022: ~4 m wide, 7 m tall, ~100 t of sand, 8 MWh, 200 kW charge, 100 kW discharge) and of the Pornainen unit (2025: 15 m x 13 m, ~2000 t of crushed soapstone, 1 MW / 100 MWh) | the default vessel and plant (`docs/06`), the closed air loop through buried pipes |

Where a number of this simulator is *not* from a reference - the 10 K hysteresis of the
gas properties, the 2 % step of the bed conductivity, the 20 % reuse window of the AMG
hierarchy - it is a numerical choice, measured and stated in
[12_METHODS.md](12_METHODS.md) and [18_SOLVER.md](18_SOLVER.md).
