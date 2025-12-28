# Thermal Battery Geometry

## 1. Overview

The battery geometry is defined in the `src/core/geometry.py` module. The system uses a simplified approach with **4 concentric zones**:
- A central **STORAGE zone** containing the thermal storage material with embedded tubes and heaters
- **Discrete elements** for heaters and heat exchanger tubes positioned INSIDE the storage

---

## 2. 4-Zone Concentric Structure

The battery cylinder is divided into **4 simplified zones** (from center outward):

| Zone | Description | Default Material |
|------|-------------|------------------|
| **STORAGE** | Thermal storage material with embedded tubes and heaters | Sand / Steatite |
| **INSULATION** | Thermal insulation layer | Rock wool |
| **STEEL** | External structural shell | Carbon steel |
| **AIR** | External air (outside the shell) | Air |

### Main Geometric Parameters:
- `r_storage`: Radius of the storage zone [m]
- `insulation_thickness`: Insulation layer thickness [m]
- `shell_thickness`: Steel shell thickness [m]
- `phase_offset_deg`: Angular offset between tubes and heaters (to avoid overlaps) [degrees]

---

## 3. Vertical Insulation Slabs

In addition to the radial zones, the geometry includes **horizontal insulation slabs** for vertical thermal protection:

### 3.1 Bottom Slab
- `insulation_slab_bottom`: Thickness of the insulation layer below the storage [m]
- Separates the storage from the ground/foundation

### 3.2 Top Slab
- `insulation_slab_top`: Thickness of the insulation layer above the storage [m]
- `steel_slab_top`: Optional steel plate thickness on top [m]
- Protects against heat losses to the atmosphere

---

## 4. Conical Roof (Optional)

For more realistic modeling, an optional **conical roof** can be enabled:

- `enable_cone_roof`: Boolean flag to enable/disable the cone
- `roof_angle_deg`: Angle of the cone from horizontal [degrees]

When enabled, the top of the battery is capped with a conical shape rather than a flat top, which better represents real industrial designs.

---

## 5. Heating Elements (Heaters)

Electric heaters are **embedded in the STORAGE zone**. They can be modeled in two ways:
1.  **Uniform Zone**: Total power is distributed uniformly throughout the storage volume.
2.  **Discrete Elements**: Vertical cylindrical heaters positioned according to a specific pattern.

### Available Patterns:
- `UNIFORM_ZONE`: Continuous volumetric distribution.
- `GRID_VERTICAL`: Rectangular grid of heaters.
- `RADIAL_ARRAY`: Heaters arranged in concentric rings.
- `SPIRAL`: Spiral arrangement from center.
- `CUSTOM`: Manually defined (x, y) positions.

### Angular Offset:
When both tubes and heaters use radial patterns, an **angular offset** (phase_offset_deg) can be set to avoid overlaps. The offset is applied to heaters, while tubes maintain the reference position.

---

## 6. Heat Exchangers (Tubes)

Heat extraction tubes are **embedded in the STORAGE zone** and positioned according to geometric patterns:
- `CENTRAL_CLUSTER`: Group of tubes at the center.
- `RADIAL_ARRAY`: Tubes arranged in concentric rings.
- `HEXAGONAL`: Maximum density pattern (hexagonal).
- `SINGLE_CENTRAL`: A single large central tube.

Each tube is characterized by:
- **Radius** [m]
- **Convective coefficient ($h$)** of the internal fluid [W/(m²·K)]
- **Fluid temperature ($T_{fluid}$)** [°C]

---

## 7. Integration with the Mesh

The `BatteryGeometry` class is responsible for "mapping" these geometric entities onto the 3D mesh via the `apply_to_mesh()` method.

### Mapping Process:
1.  For each mesh cell, coordinates $(x, y, z)$ are calculated.
2.  The radius $r = \sqrt{(x-x_c)^2 + (y-y_c)^2}$ is computed.
3.  The main zone (STORAGE, INSULATION, STEEL, AIR) is determined based on radial position.
4.  For cells in the STORAGE zone:
    - Check if the point belongs to a discrete element (tube or heater).
    - If it belongs to a tube: assign tube properties and BC.
    - If it belongs to a heater: assign properties with heat source Q.
    - Otherwise: assign storage material properties (sand/steatite).
5.  For cells above/below the cylinder: insulation slab properties are assigned.
6.  Thermophysical properties ($k, \rho, c_p$) and heat source ($Q$) are assigned accordingly.
7.  Boundary conditions (BC) are set on external faces and tube interfaces.

---

## 8. Example Configuration (YAML)

```yaml
geometry:
  cylinder:
    r_storage: 3.5              # Storage zone radius [m]
    insulation_thickness: 0.3   # Insulation thickness [m]
    shell_thickness: 0.01       # Steel shell thickness [m]
    height: 7.0                 # Total height [m]
    center_x: 5.0               # X center in domain [m]
    center_y: 5.0               # Y center in domain [m]
    base_z: 0.5                 # Z coordinate of base [m]
    phase_offset_deg: 15.0      # Angular offset tubes-heaters [degrees]
    
  slabs:
    insulation_slab_bottom: 0.2 # Bottom insulation thickness [m]
    insulation_slab_top: 0.15   # Top insulation thickness [m]
    steel_slab_top: 0.005       # Top steel plate thickness [m]
    
  roof:
    enable_cone_roof: true      # Enable conical roof
    roof_angle_deg: 30.0        # Cone angle from horizontal [degrees]
```

---

## 9. Zone Determination Algorithm

The zone assignment follows this priority order:

```python
def get_zone(x, y, z, geometry):
    r = distance_from_axis(x, y, geometry.center_x, geometry.center_y)
    
    # Check if outside cylinder height
    if z < geometry.base_z or z > geometry.top_z:
        return check_slab_zones(z, r, geometry)
    
    # Check radial zones (from outer to inner)
    if r > geometry.r_total:
        return Zone.AIR
    if r > geometry.r_storage + geometry.insulation_thickness:
        return Zone.STEEL
    if r > geometry.r_storage:
        return Zone.INSULATION
    
    # Inside storage - check for discrete elements
    if is_inside_tube(x, y, geometry.tubes):
        return Zone.TUBE
    if is_inside_heater(x, y, geometry.heaters):
        return Zone.HEATER
    
    return Zone.STORAGE
```

---

## 10. Material Property Mapping

| Zone | k [W/(m·K)] | ρ [kg/m³] | cp [J/(kg·K)] | Q [W/m³] |
|------|-------------|-----------|---------------|----------|
| STORAGE | 0.5-3.5 | 1500-2800 | 800-1000 | 0 |
| INSULATION | 0.03-0.05 | 100-150 | 800-1000 | 0 |
| STEEL | 45-50 | 7800-8000 | 450-500 | 0 |
| AIR | 0.026 | 1.2 | 1005 | 0 |
| HEATER | varies | varies | varies | P_total/V_heaters |
| TUBE | internal BC | - | - | - |
