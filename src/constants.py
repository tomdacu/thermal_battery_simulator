"""Physical constants and unit conventions for the whole package.

CONTRACT
--------
Every temperature handled inside ``src/`` (mesh fields, boundary conditions,
material limits, solver output) is an ABSOLUTE temperature in KELVIN [K].
Conversions live in :mod:`src.units` and are applied only at the UI/report
boundary (``gui/units.py``) or when reading user input.
All other quantities are SI: m, s, W, J, kg, W/(m*K), W/(m^3).
"""

# --- temperature ----------------------------------------------------------
T0 = 273.15                    # 0 degC expressed in K
T_AMBIENT_DEFAULT = 293.15     # 20 degC
T_GROUND_DEFAULT = 283.15      # 10 degC
T_INITIAL_DEFAULT = 293.15     # mesh initial field, 20 degC
T_MIN_VALID = 100.0            # [K] below this a field is certainly in degC

# --- radiation ------------------------------------------------------------
SIGMA = 5.670374419e-8         # Stefan-Boltzmann constant [W/(m^2*K^4)]

# --- media ----------------------------------------------------------------
RHO_AIR = 1.2                  # [kg/m^3]
CP_AIR = 1005.0                # [J/(kg*K)]
K_AIR = 0.026                  # [W/(m*K)]
PACKING_FRACTION_DEFAULT = 0.63

# --- environment ----------------------------------------------------------
GRAVITY = 9.81                 # [m/s^2] used by the natural-convection correlations

# --- the gas circuit ------------------------------------------------------
#: rating window of the *buried pipe* surface [W/cm^2]: the power the electric
#: resistors put into the gas leaves through the walls of the network, and a wall that
#: carries more than this is a wall nobody builds.  The window is the one immersion
#: heaters are rated by - the resistors sit in the circuit, but the surface that
#: actually delivers the power to the bed is the tubes'.
PIPE_SURFACE_POWER_MIN_W_CM2 = 3.0
PIPE_SURFACE_POWER_LIMIT_W_CM2 = 8.0

# --- geometry / numerics --------------------------------------------------
MIN_CELLS_PER_AXIS = 3
DEFAULT_SPACING = 0.2          # [m] target cell size
EPS = 1e-20                    # division guard for harmonic means
