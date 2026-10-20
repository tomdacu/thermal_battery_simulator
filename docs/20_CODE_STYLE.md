# 20. Code style: module descriptions and comments

The source follows the house style of the author's Master thesis packages
(`Master_thesis/packages`: `pvweather`, `urban3ddata`, `snowpvsim`,
`urbanpvfactors`, `pvinterop`).  The style is about *descriptions*, not about
code layout: a reader must be able to open any module and learn what it is
for, how it works, and what it does not cover, without reading the body first.

Only docstrings and comments follow this document.  The executable code is
unchanged by a description pass: the AST without docstrings is identical
before and after.

## 20.1 Module docstring

Every Python module opens with a docstring:

```
"""
src.core.mesh
=============
Structured Cartesian 3D mesh for the FDM heat solver.

WHAT IT DOES
------------
One or two paragraphs: the responsibility of the module and the objects it
owns.

HOW IT WORKS
------------
-> INDEXING <-
   The rule, the formula, the invariant.  A topic is a noun phrase in capitals
   between ``->`` and ``<-``, and its body is indented three spaces.

-> BOUNDARY CONDITIONS <-
   ...

LIMITS
------
What the module deliberately does not model, and the assumption a caller
must not break.

SOURCES
-------
- The citation already used by the module, with the key of
  [17_REFERENCES.md](17_REFERENCES.md).
"""
```

Rules:

* The first line is the dotted module path (`src.solver.fluid`, `gui.safe`,
  `tests.test_octree`), underlined with `=` of exactly the same length.  A
  seven-character title takes eight `=`, because a line of seven is a merge
  marker to Git and patch tools.
* The next paragraph is one sentence: the responsibility of the module.
* Section titles are ALL CAPS and underlined with `-` of exactly the same
  length.  Use only the sections that carry information; an empty section is
  noise.
* Section vocabulary: `WHAT IT DOES`, `WHY`, `HOW IT WORKS`, `INPUTS`,
  `OUTPUTS`, `OPTIONS`, `LIMITS`, `SOURCES`.  `WHY` holds the decision and
  the rejected alternative; `LIMITS` holds what is not modelled.
* Inside `HOW IT WORKS`, an item is `-> LABEL <-` at column 0, with the body
  indented three spaces.  A nested list may stay a Markdown list.
* Prose is wrapped at 88 columns (never above 100).  Formulas, units and
  cross-references are kept exactly as they were: the pass reorganises
  existing technical content, it does not invent new claims.
* `SOURCES` lists only references the module already cites.  The keys are
  the ones in [17_REFERENCES.md](17_REFERENCES.md); a module with no
  citation has no `SOURCES` section.

## 20.2 Section banners

Long modules divide their top level with banners of exactly 80 columns:

```
# ==============================================================================
# BOUNDARY CONDITIONS
# ==============================================================================
```

The line is `# ` followed by 78 `=`.  A banner names a group of definitions,
not a step of an algorithm; algorithms stay inside the function that runs
them.  Sections *inside* a class or a function keep the shorter
`# ---- title ----` form, as in the thesis packages: the banner marks the
file's structure, the dash line marks a local grouping.

## 20.3 Constants, fields and comments

* A module constant or a dataclass field whose meaning is not obvious from
  its name carries a `#:` comment directly above it.  The comment states the
  unit and the contract, not the type that the annotation already gives.
* Inline comments explain *why*, not *what*.  A comment that restates the
  next line is removed; a comment that names a physical assumption, a sign
  convention or a numerical choice stays.
* Every temperature is kelvin, every other quantity SI, unless the name or
  the comment says otherwise (the package contract is in
  [src/constants.py](../src/constants.py) and [05_ARCHITECTURE.md](05_ARCHITECTURE.md)).

## 20.4 Function docstrings

A function docstring opens with one line in the imperative: what the
function returns or does.  Public functions whose arguments are not obvious
from their names carry `INPUTS` and `RETURNS` sections, in the same
`-> name <-` form:

```
"""Return the face conductance of a pair of cells.

INPUTS
------
-> mesh <-
   Mesh whose face table is read.
-> face <-
   Face record ``(cell_i, cell_j, axis, area, d_centers)``.

RETURNS
-------
Conductance in W/K, positive from ``cell_i`` to ``cell_j``.
"""
```

Private helpers need the one-line summary only.  A function whose contract
is already stated by the module docstring does not repeat it.

## 20.5 Tests

A test module opens with the dotted path, a one-line claim and a `WHAT IT
DOES` section: what is checked, and against what (analytic answer, structured
twin, reference implementation, measured range).  A test function name says
the behaviour under test; its docstring, when present, says the expected
result and the reason it matters.  A test that only restates the
implementation is not documentation and is not kept.

## 20.6 What a description pass may not change

* No executable statement, signature, default, import or type annotation.
* No module, class, function or test name.
* No test assertion.
* No new reference, number or physical claim.

The check is mechanical: parse every file before and after with
`ast.parse`, drop the docstrings, and compare `ast.dump(..., include_attributes=False)`.
Comments never enter the AST, so an identical dump proves that only
descriptions moved.
