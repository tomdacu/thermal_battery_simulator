# Documentation index

| # | Document | Read it for |
|---|----------|-------------|
| 01 | [Theory](01_THEORY.md) | heat equation, porous media, dimensionless numbers, energy/exergy definitions |
| 02 | [FDM discretization](02_FDM_DISCRETIZATION.md) | the exact discrete operators, boundary treatments, matrix structure, validation |
| 03 | [Geometry model](03_GEOMETRY.md) | zones, paint order, heater/tube patterns, validation rules |
| 04 | [GUI design](04_GUI_DESIGN.md) | panels, run state machine, threading, results |
| 05 | [Architecture](05_ARCHITECTURE.md) | layers, dependency rules, unit contract, extension points |
| 06 | [GUI configuration](06_GUI_CONFIGURATION.md) | every control with its default, range and unit |
| 07 | [Code structure](07_CODE_STRUCTURE.md) | module map and public API |
| 08 | [Analysis workflows](08_ANALYSIS_WORKFLOWS.md) | steady / losses / transient runs step by step, reported quantities |
| 09 | [Testing](09_TESTING.md) | what is verified, by which test, and how to run it |
| 10 | [Graded mesh and realistic heaters](10_MESH_AND_HEATERS.md) | design and migration plan for the next phase |
| 11 | [Handoff](11_HANDOFF.md) | state of the work: read this first after a context reset |

## How these documents are kept honest

The documents describe the code that is in the repository, and every statement
that can be checked mechanically is checked:

* the defaults in [06](06_GUI_CONFIGURATION.md) are the live widget values
  (`python -c "from gui.views... import ..."`), not a copy that can drift;
* every module and symbol in [07](07_CODE_STRUCTURE.md) exists - the public
  names are the ones re-exported by the package `__init__` files;
* the discrete operators in [02](02_FDM_DISCRETIZATION.md) are the ones the
  solver assembles, and `tests/test_solver.py` pins them against an
  independently written reference;
* the formulas in [01](01_THEORY.md) and [08](08_ANALYSIS_WORKFLOWS.md) are
  implemented in `src/analysis/fluxes.py` and `src/analysis/balance.py`.

When you change a behaviour, change the document that describes it in the same
commit.  If a document and the code disagree, the code wins: fix the document.
