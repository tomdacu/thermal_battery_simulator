"""
src.constants
=============
Physical constants and unit conventions for the whole package.

WHAT IT DOES
------------
Holds the physical constants used inside ``src/`` and fixes the unit contract
every module is read against.

HOW IT WORKS
------------
-> TEMPERATURE <-
   Every temperature handled inside ``src/`` (mesh fields, boundary conditions,
   material limits, solver output) is an ABSOLUTE temperature in KELVIN [K].
   Conversions live in :mod:`src.units` and are applied only at the UI/report
   boundary (``gui/units.py``) or when reading user input.

-> OTHER QUANTITIES <-
   All other quantities are SI: m, s, W, J, kg, W/(m*K), W/(m^3).

LIMITS
------
``T_MIN_VALID`` is a plausibility guard, not a physical limit: a field above
it may still be wrong, but a field below it is in degC for certain.
"""

# ==============================================================================
# TEMPERATURE
# ==============================================================================

#: Zero degrees Celsius expressed on the Kelvin scale [K].
T0 = 273.15
#: Default ambient temperature [K]: 20 degC.
T_AMBIENT_DEFAULT = 293.15
#: Default ground temperature [K]: 10 degC.
T_GROUND_DEFAULT = 283.15
#: Default initial field of a mesh [K]: 20 degC.
T_INITIAL_DEFAULT = 293.15
#: Lowest value a plausible absolute temperature can take [K]: below it a
#: field is certainly in degC.
T_MIN_VALID = 100.0

# ==============================================================================
# RADIATION
# ==============================================================================

#: Stefan-Boltzmann constant [W/(m^2*K^4)].
SIGMA = 5.670374419e-8

# ==============================================================================
# MEDIA
# ==============================================================================

#: Density of air [kg/m^3].
RHO_AIR = 1.2
#: Specific heat of air at constant pressure [J/(kg*K)].
CP_AIR = 1005.0
#: Thermal conductivity of air [W/(m*K)].
K_AIR = 0.026
#: Packing fraction of the granular bed [-], dimensionless: used when no value
#: is given.
PACKING_FRACTION_DEFAULT = 0.63

# ==============================================================================
# ENVIRONMENT
# ==============================================================================

#: Gravitational acceleration [m/s^2], used by the natural-convection correlations.
GRAVITY = 9.81

# ==============================================================================
# THE GAS CIRCUIT
# ==============================================================================

#: Rating window of the *buried pipe* surface [W/cm^2]: the power the electric
#: resistors put into the gas leaves through the walls of the network, and a wall
#: that carries more than this is a wall nobody builds.  The window is the one
#: immersion heaters are rated by - the resistors sit in the circuit, but the
#: surface that actually delivers the power to the bed is the tubes'.
#:
#: Lower end of the rating window [W/cm^2].
PIPE_SURFACE_POWER_MIN_W_CM2 = 3.0
#: Upper end of the rating window [W/cm^2].
PIPE_SURFACE_POWER_LIMIT_W_CM2 = 8.0

# ==============================================================================
# GEOMETRY AND NUMERICS
# ==============================================================================

#: Minimum number of cells per axis of a structured mesh [-], dimensionless.
MIN_CELLS_PER_AXIS = 3
#: Default target cell size of a uniform mesh [m].
DEFAULT_SPACING = 0.2
#: Guard added to a denominator to keep a harmonic mean finite [-], dimensionless.
EPS = 1e-20
