"""
geometry.py - Definizione della geometria della Sand Battery

Gestisce:
- Geometria cilindrica con zone radiali
- Posizionamento resistenze e tubi
- Assegnazione materiali alla mesh
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from .mesh import Mesh3D, MaterialID, NodeProperties, BoundaryType
from .materials import MaterialManager, ThermalProperties


@dataclass
class CylinderGeometry:
    """
    Geometria del cilindro della batteria.
    
    La batteria è modellata come un cilindro con zone radiali concentriche:
    1. Zona tubi centrali (scambiatori)
    2. Sabbia interna
    3. Zona resistenze (riscaldatori)
    4. Sabbia esterna
    5. Isolamento
    6. Guscio in acciaio
    """
    
    # Centro del cilindro
    center_x: float = 5.0
    center_y: float = 5.0
    
    # Dimensioni verticali
    base_z: float = 0.5      # Quota base (sopra fondazione)
    height: float = 7.0      # Altezza sabbia
    
    # Raggi delle zone (dall'interno verso l'esterno)
    r_tubes: float = 0.5         # Raggio zona tubi centrali
    r_sand_inner: float = 2.0    # Raggio sabbia interna
    r_heaters: float = 2.3       # Raggio zona resistenze
    r_sand_outer: float = 3.5    # Raggio sabbia esterna
    r_insulation: float = 4.0    # Raggio isolamento
    r_shell: float = 4.01        # Raggio guscio esterno
    
    @property
    def top_z(self) -> float:
        """Quota superiore della sabbia"""
        return self.base_z + self.height
    
    def get_zone_at_radius(self, r: float) -> str:
        """Restituisce il nome della zona per un dato raggio"""
        if r < self.r_tubes:
            return "tubes"
        elif r < self.r_sand_inner:
            return "sand_inner"
        elif r < self.r_heaters:
            return "heaters"
        elif r < self.r_sand_outer:
            return "sand_outer"
        elif r < self.r_insulation:
            return "insulation"
        elif r < self.r_shell:
            return "shell"
        else:
            return "exterior"


class HeaterPattern:
    """Pattern di distribuzione delle resistenze"""
    UNIFORM_ZONE = "uniform_zone"           # Zona anulare uniforme (attuale)
    GRID_VERTICAL = "grid_vertical"         # Griglia regolare verticale
    CHESS_PATTERN = "chess_pattern"         # Pattern a scacchiera
    RADIAL_ARRAY = "radial_array"           # Array radiale (come in foto)
    SPIRAL = "spiral"                       # Pattern a spirale
    CONCENTRIC_RINGS = "concentric_rings"   # Anelli concentrici
    CUSTOM = "custom"                       # Posizioni personalizzate


class TubePattern:
    """Pattern di distribuzione dei tubi"""
    CENTRAL_CLUSTER = "central_cluster"     # Cluster centrale
    RADIAL_ARRAY = "radial_array"           # Array radiale
    GRID = "grid"                           # Griglia
    HEXAGONAL = "hexagonal"                 # Pattern esagonale
    SINGLE_CENTRAL = "single_central"       # Singolo tubo centrale
    CUSTOM = "custom"                       # Posizioni personalizzate


@dataclass
class HeaterElement:
    """Singolo elemento riscaldante"""
    x: float                    # Posizione X del centro [m]
    y: float                    # Posizione Y del centro [m]
    z_bottom: float             # Quota inferiore [m]
    z_top: float                # Quota superiore [m]
    radius: float = 0.02       # Raggio resistenza [m]
    power: float = 0.0          # Potenza [kW]


@dataclass
class TubeElement:
    """Singolo tubo scambiatore"""
    x: float                    # Posizione X del centro [m]
    y: float                    # Posizione Y del centro [m]
    z_bottom: float             # Quota inferiore [m]
    z_top: float                # Quota superiore [m]
    radius: float = 0.025       # Raggio tubo [m]
    h_fluid: float = 500.0      # Coefficiente convettivo [W/(m²·K)]
    T_fluid: float = 60.0       # Temperatura fluido [°C]


@dataclass
class HeaterConfig:
    """
    Configurazione delle resistenze elettriche.
    
    Supporta diversi pattern di distribuzione:
    - uniform_zone: Zona anulare con sorgente uniforme
    - grid_vertical: Resistenze verticali in griglia regolare
    - chess_pattern: Pattern a scacchiera
    - radial_array: Array radiale (come elementi tubolari in foto)
    - spiral: Pattern a spirale
    - concentric_rings: Anelli concentrici
    - custom: Posizioni definite manualmente
    """
    power_total: float = 100.0              # kW - Potenza totale
    n_heaters: int = 12                     # Numero di resistenze
    pattern: str = HeaterPattern.UNIFORM_ZONE  # Pattern di distribuzione
    
    # Parametri geometrici
    heater_radius: float = 0.02             # Raggio singola resistenza [m]
    heater_length: float = None             # Lunghezza (None = altezza batteria)
    
    # Per pattern radiale/grid
    n_rings: int = 2                        # Numero di anelli (per radial)
    n_per_ring: List[int] = None            # Resistenze per anello [ring1, ring2, ...]
    ring_radii: List[float] = None          # Raggi degli anelli [m]
    
    # Per grid pattern
    grid_rows: int = 4                      # Righe della griglia
    grid_cols: int = 4                      # Colonne della griglia
    grid_spacing: float = 0.3               # Spaziatura griglia [m]
    
    # Per custom pattern
    custom_positions: List[Tuple[float, float]] = None  # Lista (x, y)
    
    # Elementi calcolati
    elements: List[HeaterElement] = field(default_factory=list)
    
    def __post_init__(self):
        if self.n_per_ring is None:
            self.n_per_ring = []
        if self.ring_radii is None:
            self.ring_radii = []
        if self.custom_positions is None:
            self.custom_positions = []
    
    @property
    def power_per_heater(self) -> float:
        """Potenza per resistenza [kW]"""
        return self.power_total / max(self.n_heaters, 1)
    
    def generate_positions(self, center_x: float, center_y: float,
                           r_inner: float, r_outer: float,
                           z_bottom: float, z_top: float) -> List[HeaterElement]:
        """
        Genera le posizioni delle resistenze secondo il pattern selezionato.
        
        Args:
            center_x, center_y: Centro della batteria
            r_inner, r_outer: Raggi interno ed esterno della zona resistenze
            z_bottom, z_top: Altezza della zona resistenze
            
        Returns:
            Lista di HeaterElement con posizioni calcolate
        """
        self.elements = []
        power_each = self.power_per_heater
        length = self.heater_length if self.heater_length else (z_top - z_bottom)
        
        if self.pattern == HeaterPattern.UNIFORM_ZONE:
            # Nessun elemento discreto, zona uniforme
            return []
        
        elif self.pattern == HeaterPattern.GRID_VERTICAL:
            # Griglia regolare centrata
            positions = self._generate_grid_positions(
                center_x, center_y, r_inner, r_outer
            )
            
        elif self.pattern == HeaterPattern.CHESS_PATTERN:
            # Pattern a scacchiera
            positions = self._generate_chess_positions(
                center_x, center_y, r_inner, r_outer
            )
            
        elif self.pattern == HeaterPattern.RADIAL_ARRAY:
            # Array radiale come in foto
            positions = self._generate_radial_positions(
                center_x, center_y, r_inner, r_outer
            )
            
        elif self.pattern == HeaterPattern.SPIRAL:
            # Spirale
            positions = self._generate_spiral_positions(
                center_x, center_y, r_inner, r_outer
            )
            
        elif self.pattern == HeaterPattern.CONCENTRIC_RINGS:
            # Anelli concentrici
            positions = self._generate_ring_positions(
                center_x, center_y, r_inner, r_outer
            )
            
        elif self.pattern == HeaterPattern.CUSTOM:
            # Posizioni personalizzate
            positions = [(x, y) for x, y in self.custom_positions]
            
        else:
            positions = []
        
        # Crea elementi
        for x, y in positions:
            self.elements.append(HeaterElement(
                x=x, y=y,
                z_bottom=z_bottom, z_top=z_bottom + length,
                radius=self.heater_radius,
                power=power_each
            ))
        
        # Aggiorna il numero effettivo di resistenze
        self.n_heaters = len(self.elements) if self.elements else self.n_heaters
        
        return self.elements
    
    def _generate_grid_positions(self, cx: float, cy: float, 
                                  r_in: float, r_out: float) -> List[Tuple[float, float]]:
        """Genera posizioni a griglia regolare"""
        positions = []
        half_width = (self.grid_cols - 1) * self.grid_spacing / 2
        half_height = (self.grid_rows - 1) * self.grid_spacing / 2
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x = cx - half_width + col * self.grid_spacing
                y = cy - half_height + row * self.grid_spacing
                
                # Verifica che sia nella zona resistenze
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                if r_in <= r <= r_out:
                    positions.append((x, y))
        
        return positions
    
    def _generate_chess_positions(self, cx: float, cy: float,
                                   r_in: float, r_out: float) -> List[Tuple[float, float]]:
        """Genera posizioni a scacchiera"""
        positions = []
        half_width = (self.grid_cols - 1) * self.grid_spacing / 2
        half_height = (self.grid_rows - 1) * self.grid_spacing / 2
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                # Solo celle "bianche" della scacchiera
                if (row + col) % 2 == 0:
                    x = cx - half_width + col * self.grid_spacing
                    y = cy - half_height + row * self.grid_spacing
                    
                    r = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if r_in <= r <= r_out:
                        positions.append((x, y))
        
        return positions
    
    def _generate_radial_positions(self, cx: float, cy: float,
                                    r_in: float, r_out: float) -> List[Tuple[float, float]]:
        """
        Genera posizioni in array radiale (come elementi tubolari in foto).
        Le resistenze sono disposte su anelli concentrici.
        """
        positions = []
        
        if self.ring_radii and self.n_per_ring:
            # Usa configurazione esplicita
            radii = self.ring_radii
            counts = self.n_per_ring
        else:
            # Calcola automaticamente
            n_rings = max(1, self.n_rings)
            radii = np.linspace(r_in + 0.1, r_out - 0.1, n_rings).tolist()
            
            # Distribuisci le resistenze proporzionalmente al raggio
            total_circumference = sum(2 * np.pi * r for r in radii)
            counts = []
            remaining = self.n_heaters
            for i, r in enumerate(radii):
                if i == len(radii) - 1:
                    counts.append(remaining)
                else:
                    n = int(self.n_heaters * (2 * np.pi * r) / total_circumference)
                    counts.append(max(1, n))
                    remaining -= counts[-1]
        
        # Genera posizioni per ogni anello
        for radius, n_elements in zip(radii, counts):
            if n_elements > 0:
                for i in range(n_elements):
                    angle = 2 * np.pi * i / n_elements
                    x = cx + radius * np.cos(angle)
                    y = cy + radius * np.sin(angle)
                    positions.append((x, y))
        
        return positions
    
    def _generate_spiral_positions(self, cx: float, cy: float,
                                    r_in: float, r_out: float) -> List[Tuple[float, float]]:
        """Genera posizioni a spirale"""
        positions = []
        n = self.n_heaters
        
        for i in range(n):
            t = i / max(n - 1, 1)  # 0 to 1
            r = r_in + t * (r_out - r_in)
            angle = t * 4 * np.pi  # 2 giri completi
            
            x = cx + r * np.cos(angle)
            y = cy + r * np.sin(angle)
            positions.append((x, y))
        
        return positions
    
    def _generate_ring_positions(self, cx: float, cy: float,
                                  r_in: float, r_out: float) -> List[Tuple[float, float]]:
        """Genera posizioni su anelli concentrici uniformi"""
        positions = []
        n_rings = max(1, self.n_rings)
        radii = np.linspace(r_in + 0.05, r_out - 0.05, n_rings)
        
        heaters_per_ring = max(1, self.n_heaters // n_rings)
        
        for ring_idx, r in enumerate(radii):
            n_on_ring = heaters_per_ring
            # Offset angolare alternato per anelli
            offset = (np.pi / n_on_ring) if ring_idx % 2 == 1 else 0
            
            for i in range(n_on_ring):
                angle = offset + 2 * np.pi * i / n_on_ring
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                positions.append((x, y))
        
        return positions


@dataclass
class TubeConfig:
    """
    Configurazione dei tubi scambiatori.
    
    Supporta diversi pattern di distribuzione:
    - central_cluster: Cluster di tubi al centro
    - radial_array: Array radiale
    - grid: Griglia regolare
    - hexagonal: Pattern esagonale (alta densità)
    - single_central: Singolo tubo centrale
    - custom: Posizioni definite manualmente
    """
    n_tubes: int = 8                        # Numero di tubi
    diameter: float = 0.05                  # Diametro tubo [m]
    h_fluid: float = 500.0                  # Coefficiente convettivo [W/(m²·K)]
    T_fluid: float = 60.0                   # Temperatura fluido [°C]
    active: bool = False                    # True durante scarica
    pattern: str = TubePattern.RADIAL_ARRAY # Pattern di distribuzione
    
    # Parametri geometrici
    tube_length: float = None               # Lunghezza (None = altezza batteria)
    
    # Per pattern radiale
    n_rings: int = 2                        # Numero di anelli
    n_per_ring: List[int] = None            # Tubi per anello
    ring_radii: List[float] = None          # Raggi degli anelli [m]
    
    # Per grid pattern
    grid_rows: int = 3
    grid_cols: int = 3
    grid_spacing: float = 0.2               # Spaziatura griglia [m]
    
    # Per custom pattern
    custom_positions: List[Tuple[float, float]] = None
    
    # Elementi calcolati
    elements: List[TubeElement] = field(default_factory=list)
    
    def __post_init__(self):
        if self.n_per_ring is None:
            self.n_per_ring = []
        if self.ring_radii is None:
            self.ring_radii = []
        if self.custom_positions is None:
            self.custom_positions = []
    
    @property
    def radius(self) -> float:
        """Raggio del tubo [m]"""
        return self.diameter / 2
    
    def generate_positions(self, center_x: float, center_y: float,
                           r_max: float,
                           z_bottom: float, z_top: float) -> List[TubeElement]:
        """
        Genera le posizioni dei tubi secondo il pattern selezionato.
        
        Args:
            center_x, center_y: Centro della batteria
            r_max: Raggio massimo della zona tubi
            z_bottom, z_top: Altezza della zona tubi
            
        Returns:
            Lista di TubeElement con posizioni calcolate
        """
        self.elements = []
        length = self.tube_length if self.tube_length else (z_top - z_bottom)
        
        if self.pattern == TubePattern.SINGLE_CENTRAL:
            positions = [(center_x, center_y)]
            
        elif self.pattern == TubePattern.CENTRAL_CLUSTER:
            positions = self._generate_cluster_positions(center_x, center_y, r_max)
            
        elif self.pattern == TubePattern.RADIAL_ARRAY:
            positions = self._generate_radial_positions(center_x, center_y, r_max)
            
        elif self.pattern == TubePattern.GRID:
            positions = self._generate_grid_positions(center_x, center_y, r_max)
            
        elif self.pattern == TubePattern.HEXAGONAL:
            positions = self._generate_hexagonal_positions(center_x, center_y, r_max)
            
        elif self.pattern == TubePattern.CUSTOM:
            positions = [(x, y) for x, y in self.custom_positions]
            
        else:
            positions = [(center_x, center_y)]
        
        # Crea elementi
        for x, y in positions:
            self.elements.append(TubeElement(
                x=x, y=y,
                z_bottom=z_bottom, z_top=z_bottom + length,
                radius=self.radius,
                h_fluid=self.h_fluid,
                T_fluid=self.T_fluid
            ))
        
        self.n_tubes = len(self.elements)
        return self.elements
    
    def _generate_cluster_positions(self, cx: float, cy: float,
                                     r_max: float) -> List[Tuple[float, float]]:
        """Genera cluster di tubi al centro"""
        positions = [(cx, cy)]  # Tubo centrale
        
        if self.n_tubes > 1:
            # Aggiungi tubi attorno al centro
            n_around = min(6, self.n_tubes - 1)
            r_ring = min(r_max * 0.6, self.grid_spacing * 2)
            
            for i in range(n_around):
                angle = 2 * np.pi * i / n_around
                x = cx + r_ring * np.cos(angle)
                y = cy + r_ring * np.sin(angle)
                positions.append((x, y))
        
        return positions[:self.n_tubes]
    
    def _generate_radial_positions(self, cx: float, cy: float,
                                    r_max: float) -> List[Tuple[float, float]]:
        """Genera tubi in array radiale"""
        positions = []
        
        if self.n_tubes == 1:
            return [(cx, cy)]
        
        if self.ring_radii and self.n_per_ring:
            radii = self.ring_radii
            counts = self.n_per_ring
        else:
            # Calcola automaticamente
            n_rings = max(1, self.n_rings)
            radii = np.linspace(r_max * 0.3, r_max * 0.8, n_rings).tolist()
            
            # Distribuisci uniformemente
            base_per_ring = self.n_tubes // n_rings
            counts = [base_per_ring] * n_rings
            # Aggiungi i rimanenti al primo anello
            for i in range(self.n_tubes % n_rings):
                counts[i] += 1
        
        for radius, n_elements in zip(radii, counts):
            for i in range(n_elements):
                angle = 2 * np.pi * i / n_elements
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                positions.append((x, y))
        
        return positions
    
    def _generate_grid_positions(self, cx: float, cy: float,
                                  r_max: float) -> List[Tuple[float, float]]:
        """Genera tubi in griglia"""
        positions = []
        half_width = (self.grid_cols - 1) * self.grid_spacing / 2
        half_height = (self.grid_rows - 1) * self.grid_spacing / 2
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x = cx - half_width + col * self.grid_spacing
                y = cy - half_height + row * self.grid_spacing
                
                # Verifica che sia dentro il raggio massimo
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                if r <= r_max:
                    positions.append((x, y))
        
        return positions[:self.n_tubes]
    
    def _generate_hexagonal_positions(self, cx: float, cy: float,
                                       r_max: float) -> List[Tuple[float, float]]:
        """Genera tubi in pattern esagonale (massima densità)"""
        positions = [(cx, cy)]  # Centro
        
        spacing = self.grid_spacing
        rings = 1
        
        while len(positions) < self.n_tubes:
            # Aggiungi anello esagonale
            r = rings * spacing
            if r > r_max:
                break
                
            n_on_ring = 6 * rings
            for i in range(n_on_ring):
                angle = 2 * np.pi * i / n_on_ring + np.pi / 6
                x = cx + r * np.cos(angle)
                y = cy + r * np.sin(angle)
                
                if np.sqrt((x - cx)**2 + (y - cy)**2) <= r_max:
                    positions.append((x, y))
                    
            rings += 1
        
        return positions[:self.n_tubes]


@dataclass
class BatteryGeometry:
    """
    Geometria completa della Sand Battery.
    
    Combina cilindro, resistenze e tubi in una configurazione completa.
    """
    
    # Geometria cilindro
    cylinder: CylinderGeometry = field(default_factory=CylinderGeometry)
    
    # Configurazioni
    heaters: HeaterConfig = field(default_factory=HeaterConfig)
    tubes: TubeConfig = field(default_factory=TubeConfig)
    
    # Materiali (nomi)
    storage_material: str = "steatite"
    insulation_material: str = "rock_wool"
    shell_material: str = "carbon_steel"
    
    # Packing
    packing_fraction: float = 0.63
    
    def apply_to_mesh(self, mesh: Mesh3D, mat_manager: MaterialManager):
        """
        Applica la geometria alla mesh, assegnando materiali e proprietà.
        
        Supporta sia zone uniformi che elementi discreti (resistenze e tubi).
        
        Args:
            mesh: Mesh3D da configurare
            mat_manager: MaterialManager per le proprietà
        """
        cyl = self.cylinder
        
        # Ottieni proprietà materiali
        sand_props = mat_manager.compute_packed_bed_properties(
            self.storage_material, self.packing_fraction
        )
        insul_props = mat_manager.get(self.insulation_material)
        steel_props = mat_manager.get(self.shell_material)
        air_props = mat_manager.get("air")
        
        # Genera posizioni elementi se pattern discreto
        heater_elements = []
        tube_elements = []
        
        if self.heaters.pattern != HeaterPattern.UNIFORM_ZONE:
            heater_elements = self.heaters.generate_positions(
                cyl.center_x, cyl.center_y,
                cyl.r_sand_inner, cyl.r_heaters,
                cyl.base_z, cyl.top_z
            )
        
        tube_elements = self.tubes.generate_positions(
            cyl.center_x, cyl.center_y,
            cyl.r_tubes,
            cyl.base_z, cyl.top_z
        )
        
        # Calcola sorgente di calore per le resistenze
        if self.heaters.pattern == HeaterPattern.UNIFORM_ZONE:
            # Zona anulare uniforme
            V_heaters = (np.pi * (cyl.r_heaters**2 - cyl.r_sand_inner**2) * cyl.height)
            Q_heaters = self.heaters.power_total * 1000 / V_heaters  # W/m³
        else:
            # Elementi discreti - calcola Q per ogni elemento
            if heater_elements:
                heater_length = self.heaters.heater_length if self.heaters.heater_length else cyl.height
                V_single = np.pi * self.heaters.heater_radius**2 * heater_length
                Q_heaters = (self.heaters.power_per_heater * 1000) / V_single
            else:
                Q_heaters = 0.0
        
        heater_node_props = NodeProperties(
            k=sand_props.k,
            rho=sand_props.rho,
            cp=sand_props.cp,
            Q=Q_heaters
        )
        
        # Itera su tutte le celle della mesh
        for i in range(mesh.Nx):
            for j in range(mesh.Ny):
                x, y = mesh.x[i], mesh.y[j]
                
                # Calcola distanza radiale dal centro
                r = np.sqrt((x - cyl.center_x)**2 + (y - cyl.center_y)**2)
                
                for k in range(mesh.Nz):
                    z = mesh.z[k]
                    
                    # Verifica se siamo nell'altezza della batteria
                    if cyl.base_z <= z <= cyl.top_z:
                        
                        # Prima controlla se siamo in un elemento discreto
                        is_in_heater = False
                        is_in_tube = False
                        
                        # Controlla resistenze discrete
                        if heater_elements:
                            for heater in heater_elements:
                                if heater.z_bottom <= z <= heater.z_top:
                                    dist = np.sqrt((x - heater.x)**2 + (y - heater.y)**2)
                                    if dist <= heater.radius:
                                        is_in_heater = True
                                        break
                        
                        # Controlla tubi discreti
                        if tube_elements:
                            for tube in tube_elements:
                                if tube.z_bottom <= z <= tube.z_top:
                                    dist = np.sqrt((x - tube.x)**2 + (y - tube.y)**2)
                                    if dist <= tube.radius:
                                        is_in_tube = True
                                        # Applica lo scambio solo se i tubi sono attivi (scarica).
                                        # Se non attivi, i tubi restano come regione materiale senza sink convettivo.
                                        if self.tubes.active:
                                            mesh.bc_h[i, j, k] = tube.h_fluid
                                            mesh.bc_T_inf[i, j, k] = tube.T_fluid
                                            mesh.boundary_type[i, j, k] = BoundaryType.CONVECTION
                                        else:
                                            mesh.bc_h[i, j, k] = 0.0
                                            mesh.boundary_type[i, j, k] = BoundaryType.INTERNAL
                                        break
                        
                        if is_in_heater:
                            # Nodo nella resistenza
                            mesh.material_id[i, j, k] = MaterialID.HEATERS
                            mesh.k[i, j, k] = heater_node_props.k
                            mesh.rho[i, j, k] = heater_node_props.rho
                            mesh.cp[i, j, k] = heater_node_props.cp
                            mesh.Q[i, j, k] = heater_node_props.Q
                            
                        elif is_in_tube:
                            # Nodo nel tubo
                            self._set_node_properties(mesh, i, j, k,
                                                     MaterialID.TUBES, sand_props)
                        
                        # Altrimenti usa le zone radiali
                        elif r < cyl.r_tubes:
                            # Zona tubi centrale (se non elementi discreti)
                            self._set_node_properties(mesh, i, j, k, 
                                                     MaterialID.TUBES, sand_props)
                            if self.tubes.active and not tube_elements:
                                mesh.bc_h[i, j, k] = self.tubes.h_fluid
                                mesh.bc_T_inf[i, j, k] = self.tubes.T_fluid
                                mesh.boundary_type[i, j, k] = BoundaryType.CONVECTION
                                
                        elif r < cyl.r_sand_inner:
                            # Sabbia interna
                            self._set_node_properties(mesh, i, j, k,
                                                     MaterialID.SAND, sand_props)
                            
                        elif r < cyl.r_heaters:
                            # Zona resistenze (uniforme o con elementi discreti)
                            if self.heaters.pattern == HeaterPattern.UNIFORM_ZONE:
                                mesh.material_id[i, j, k] = MaterialID.HEATERS
                                mesh.k[i, j, k] = heater_node_props.k
                                mesh.rho[i, j, k] = heater_node_props.rho
                                mesh.cp[i, j, k] = heater_node_props.cp
                                mesh.Q[i, j, k] = heater_node_props.Q
                            else:
                                # Con elementi discreti, la zona è sabbia
                                self._set_node_properties(mesh, i, j, k,
                                                         MaterialID.SAND, sand_props)
                            
                        elif r < cyl.r_sand_outer:
                            # Sabbia esterna
                            self._set_node_properties(mesh, i, j, k,
                                                     MaterialID.SAND, sand_props)
                            
                        elif r < cyl.r_insulation:
                            # Isolamento
                            self._set_node_properties(mesh, i, j, k,
                                                     MaterialID.INSULATION, insul_props)
                            
                        elif r < cyl.r_shell:
                            # Guscio in acciaio
                            self._set_node_properties(mesh, i, j, k,
                                                     MaterialID.STEEL, steel_props)
                        else:
                            # Aria esterna
                            self._set_node_properties(mesh, i, j, k,
                                                     MaterialID.AIR, air_props)
                    else:
                        # Sopra o sotto la batteria
                        if z < cyl.base_z:
                            # Fondazione (calcestruzzo)
                            concrete_props = mat_manager.get("concrete")
                            self._set_node_properties(mesh, i, j, k,
                                                     MaterialID.CONCRETE, concrete_props)
                        else:
                            # Aria sopra
                            self._set_node_properties(mesh, i, j, k,
                                                     MaterialID.AIR, air_props)
        
        # Imposta condizioni al contorno
        self._apply_boundary_conditions(mesh)
    
    def _set_node_properties(self, mesh: Mesh3D, i: int, j: int, k: int,
                             material_id: MaterialID, props: ThermalProperties):
        """Helper per impostare le proprietà di un nodo"""
        mesh.material_id[i, j, k] = material_id
        mesh.k[i, j, k] = props.k
        mesh.rho[i, j, k] = props.rho
        mesh.cp[i, j, k] = props.cp
    
    def _apply_boundary_conditions(self, mesh: Mesh3D, 
                                   h_top: float = 10.0,
                                   h_lateral: float = 5.0,
                                   T_ambient: float = 20.0,
                                   T_ground: float = 10.0):
        """Applica condizioni al contorno al dominio"""
        
        # Faccia superiore - convezione con aria
        mesh.set_convection_bc('z_max', h_top, T_ambient)
        
        # Facce laterali - convezione
        mesh.set_convection_bc('x_min', h_lateral, T_ambient)
        mesh.set_convection_bc('x_max', h_lateral, T_ambient)
        mesh.set_convection_bc('y_min', h_lateral, T_ambient)
        mesh.set_convection_bc('y_max', h_lateral, T_ambient)
        
        # Faccia inferiore - temperatura fissa (terreno)
        mesh.set_fixed_temperature_bc('z_min', T_ground)
    
    def get_zone_volumes(self) -> dict:
        """Calcola i volumi di ogni zona [m³]"""
        cyl = self.cylinder
        h = cyl.height
        
        volumes = {
            'tubes': np.pi * cyl.r_tubes**2 * h,
            'sand_inner': np.pi * (cyl.r_sand_inner**2 - cyl.r_tubes**2) * h,
            'heaters': np.pi * (cyl.r_heaters**2 - cyl.r_sand_inner**2) * h,
            'sand_outer': np.pi * (cyl.r_sand_outer**2 - cyl.r_heaters**2) * h,
            'insulation': np.pi * (cyl.r_insulation**2 - cyl.r_sand_outer**2) * h,
            'shell': np.pi * (cyl.r_shell**2 - cyl.r_insulation**2) * h,
        }
        
        volumes['sand_total'] = volumes['sand_inner'] + volumes['sand_outer'] + volumes['heaters']
        volumes['total'] = np.pi * cyl.r_shell**2 * h
        
        return volumes
    
    def get_zone_masses(self, mat_manager: MaterialManager) -> dict:
        """Calcola le masse di ogni zona [kg]"""
        volumes = self.get_zone_volumes()
        
        sand_props = mat_manager.compute_packed_bed_properties(
            self.storage_material, self.packing_fraction
        )
        insul_props = mat_manager.get(self.insulation_material)
        steel_props = mat_manager.get(self.shell_material)
        
        masses = {
            'sand_total': volumes['sand_total'] * sand_props.rho,
            'insulation': volumes['insulation'] * insul_props.rho,
            'shell': volumes['shell'] * steel_props.rho,
        }
        
        masses['total'] = sum(masses.values())
        
        return masses
    
    def estimate_energy_capacity(self, mat_manager: MaterialManager,
                                  T_high: float = 380.0,
                                  T_low: float = 90.0,
                                  efficiency: float = 0.87) -> dict:
        """
        Stima la capacità energetica della batteria.
        
        Returns:
            dict con energia termica e utilizzabile [kWh e MWh]
        """
        masses = self.get_zone_masses(mat_manager)
        sand_props = mat_manager.compute_packed_bed_properties(
            self.storage_material, self.packing_fraction
        )
        
        delta_T = T_high - T_low
        
        # Energia termica totale
        E_thermal_J = masses['sand_total'] * sand_props.cp * delta_T
        E_thermal_kWh = E_thermal_J / 3.6e6
        E_thermal_MWh = E_thermal_kWh / 1000
        
        # Energia utilizzabile (con efficienza)
        E_usable_kWh = E_thermal_kWh * efficiency
        E_usable_MWh = E_usable_kWh / 1000
        
        return {
            'E_thermal_kWh': E_thermal_kWh,
            'E_thermal_MWh': E_thermal_MWh,
            'E_usable_kWh': E_usable_kWh,
            'E_usable_MWh': E_usable_MWh,
            'mass_sand_kg': masses['sand_total'],
            'mass_sand_tonnes': masses['sand_total'] / 1000,
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_small_test_geometry() -> BatteryGeometry:
    """Crea geometria piccola per test (8 MWh circa)"""
    
    cylinder = CylinderGeometry(
        center_x=3.0,
        center_y=3.0,
        base_z=0.3,
        height=4.0,
        r_tubes=0.3,
        r_sand_inner=1.0,
        r_heaters=1.2,
        r_sand_outer=2.0,
        r_insulation=2.3,
        r_shell=2.32,
    )
    
    return BatteryGeometry(
        cylinder=cylinder,
        heaters=HeaterConfig(power_total=50),
        tubes=TubeConfig(n_tubes=6),
        storage_material="steatite",
        packing_fraction=0.63,
    )
