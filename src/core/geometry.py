"""
geometry.py - Definizione della geometria della Thermal Battery

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
    Geometria del cilindro della batteria termica.
    
    STRUTTURA COMPLETA CON ISOLAMENTO E TETTO:
    
    Dal basso verso l'alto:
    - FONDAZIONE (calcestruzzo, sotto base_z)
    - SLAB ISOLANTE INFERIORE (sotto lo storage)
    - STORAGE (materiale di accumulo + tubi + resistenze)
    - SLAB ISOLANTE SUPERIORE (sopra lo storage)
    - SLAB ACCIAIO (opzionale, sotto il tetto)
    - RIEMPIMENTO SABBIA CONO (opzionale)
    - TETTO CONICO (acciaio, inclinato)
    
    Radialmente (dal centro verso l'esterno):
    1. STORAGE + SLAB ISOLANTI (r < r_storage)
    2. INSULATION radiale (r_storage < r < r_insulation)
    3. STEEL shell (r_insulation < r < r_shell)
    4. AIR (r > r_shell)
    
    L'acciaio laterale (shell) copre tutta l'altezza:
    da slab_inf a tetto (inclusi gli slab di isolamento).
    """
    
    # Centro del cilindro
    center_x: float = 5.0
    center_y: float = 5.0
    
    # Dimensioni verticali dello STORAGE
    base_z: float = 0.5      # Quota base della struttura (sopra fondazione)
    height: float = 7.0      # Altezza della zona STORAGE (sabbia)
    
    # Raggi delle zone (dall'interno verso l'esterno)
    r_storage: float = 2.0        # Raggio zona storage
    insulation_thickness: float = 0.3   # Spessore isolamento radiale [m]
    shell_thickness: float = 0.02       # Spessore guscio acciaio [m]
    
    # === NUOVI PARAMETRI: SLAB ISOLANTI ===
    insulation_slab_bottom: float = 0.2   # Altezza slab isolante inferiore [m]
    insulation_slab_top: float = 0.2      # Altezza slab isolante superiore [m]
    
    # === NUOVI PARAMETRI: TETTO ===
    roof_angle_deg: float = 15.0          # Inclinazione tetto [gradi] (0 = piatto)
    steel_slab_top: float = 0.0           # Spessore slab acciaio sotto tetto [m] (0 = disabilitato)
    fill_cone_with_sand: bool = False     # Riempie il volume sotto il cono con sabbia
    enable_cone_roof: bool = True         # Abilita tetto conico (False = tetto piatto con steel_slab)
    
    # Sfasamento angolare tra tubi e resistenze [gradi]
    phase_offset_deg: float = 15.0
    
    # === PROPRIETÀ CALCOLATE ===
    
    @property
    def r_insulation(self) -> float:
        """Raggio esterno dell'isolamento"""
        return self.r_storage + self.insulation_thickness
    
    @property
    def r_shell(self) -> float:
        """Raggio esterno del guscio (raggio totale batteria)"""
        return self.r_insulation + self.shell_thickness
    
    @property
    def roof_angle_rad(self) -> float:
        """Inclinazione tetto in radianti"""
        return np.radians(self.roof_angle_deg)
    
    @property
    def roof_height(self) -> float:
        """Altezza del cono del tetto al centro"""
        if not self.enable_cone_roof or self.roof_angle_deg <= 0:
            return 0.0
        return self.r_shell * np.tan(self.roof_angle_rad)
    
    @property
    def z_tubes_top(self) -> float:
        """Quota superiore dei tubi (sopra steel_slab o slab_top)"""
        return self.z_steel_slab_end
    
    # === QUOTE Z (dal basso verso l'alto) ===
    
    @property
    def z_slab_bottom_start(self) -> float:
        """Inizio slab isolante inferiore"""
        return self.base_z
    
    @property
    def z_slab_bottom_end(self) -> float:
        """Fine slab isolante inferiore = inizio storage"""
        return self.base_z + self.insulation_slab_bottom
    
    @property
    def z_storage_start(self) -> float:
        """Inizio zona storage"""
        return self.z_slab_bottom_end
    
    @property
    def z_storage_end(self) -> float:
        """Fine zona storage"""
        return self.z_storage_start + self.height
    
    @property
    def z_slab_top_start(self) -> float:
        """Inizio slab isolante superiore"""
        return self.z_storage_end
    
    @property
    def z_slab_top_end(self) -> float:
        """Fine slab isolante superiore"""
        return self.z_slab_top_start + self.insulation_slab_top
    
    @property
    def z_steel_slab_end(self) -> float:
        """Fine slab acciaio (se presente)"""
        return self.z_slab_top_end + self.steel_slab_top
    
    @property
    def z_cone_base(self) -> float:
        """Base del cono (tetto)"""
        return self.z_steel_slab_end
    
    @property
    def z_cone_apex(self) -> float:
        """Apice del cono (punto più alto)"""
        return self.z_cone_base + self.roof_height
    
    @property
    def z_shell_top(self) -> float:
        """Quota superiore del guscio laterale (base del cono)"""
        return self.z_cone_base
    
    @property
    def total_height(self) -> float:
        """Altezza totale della struttura (escluso terreno)"""
        return self.z_cone_apex - self.base_z
    
    # Proprietà legacy per compatibilità
    @property
    def top_z(self) -> float:
        """Quota superiore della zona storage (compatibilità)"""
        return self.z_storage_end
    
    @property
    def phase_offset_rad(self) -> float:
        """Sfasamento angolare in radianti"""
        return np.radians(self.phase_offset_deg)
    
    def get_zone_at_radius(self, r: float) -> str:
        """Restituisce il nome della zona per un dato raggio"""
        if r < self.r_storage:
            return "storage"
        elif r < self.r_insulation:
            return "insulation"
        elif r < self.r_shell:
            return "shell"
        else:
            return "exterior"
    
    def is_inside_cone(self, r: float, z: float) -> bool:
        """Verifica se un punto (r, z) è dentro il cono del tetto"""
        if z < self.z_cone_base or z > self.z_cone_apex:
            return False
        if self.roof_height <= 0:
            return False
        # Raggio del cono a quota z
        z_rel = z - self.z_cone_base
        r_cone_at_z = self.r_shell * (1 - z_rel / self.roof_height)
        return r < r_cone_at_z


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
    
    Le resistenze stanno SOLO nella zona STORAGE.
    Gli offset permettono di farle partire/finire prima dei bordi dello storage.
    """
    power_total: float = 100.0              # kW - Potenza totale
    n_heaters: int = 12                     # Numero di resistenze
    pattern: str = HeaterPattern.UNIFORM_ZONE  # Pattern di distribuzione
    
    # Parametri geometrici
    heater_radius: float = 0.02             # Raggio singola resistenza [m]
    heater_length: float = None             # Lunghezza (None = altezza batteria)
    
    # === OFFSET VERTICALI (rispetto ai bordi dello storage) ===
    offset_bottom: float = 0.0              # Distanza dalla fine dello slab inferiore [m]
    offset_top: float = 0.0                 # Distanza prima dell'inizio dello slab superiore [m]
    
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
                           z_bottom: float, z_top: float,
                           phase_offset: float = 0.0) -> List[HeaterElement]:
        """
        Genera le posizioni delle resistenze secondo il pattern selezionato.
        
        Args:
            center_x, center_y: Centro della batteria
            r_inner, r_outer: Raggi interno ed esterno della zona dove posizionare
            z_bottom, z_top: Altezza della zona resistenze
            phase_offset: Sfasamento angolare in radianti (per evitare sovrapposizioni con tubi)
            
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
                center_x, center_y, r_inner, r_outer, phase_offset
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
                                    r_in: float, r_out: float,
                                    phase_offset: float = 0.0) -> List[Tuple[float, float]]:
        """
        Genera posizioni in array radiale (come elementi tubolari in foto).
        Le resistenze sono disposte su anelli concentrici.
        
        Args:
            cx, cy: Centro
            r_in, r_out: Raggi interno ed esterno
            phase_offset: Sfasamento angolare in radianti
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
        
        # Genera posizioni per ogni anello con sfasamento
        for radius, n_elements in zip(radii, counts):
            if n_elements > 0:
                for i in range(n_elements):
                    angle = 2 * np.pi * i / n_elements + phase_offset
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
    Geometria completa della Thermal Battery.
    
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
        
        VERSIONE VETTORIZZATA - 10-100x più veloce del loop Python.
        
        STRUTTURA COMPLETA:
        - FONDAZIONE (sotto base_z)
        - SLAB ISOLANTE INFERIORE
        - STORAGE (sabbia + tubi + resistenze)
        - SLAB ISOLANTE SUPERIORE
        - SLAB ACCIAIO (opzionale)
        - RIEMPIMENTO SABBIA SOTTO CONO (opzionale)
        - TETTO CONICO (acciaio) - opzionale, controllato da enable_cone_roof
        - ACCIAIO LATERALE (shell) che copre tutta l'altezza
        - ISOLAMENTO RADIALE attorno a storage + slab
        
        TUBI: vanno da z_slab_bottom_start a z_steel_slab_end (attraversano tutto)
        RESISTENZE: vanno da z_storage_start + offset_bottom a z_storage_end - offset_top
        
        Args:
            mesh: Mesh3D da configurare
            mat_manager: MaterialManager per le proprietà
        """
        cyl = self.cylinder
        
        # Ottieni proprietà materiali
        storage_props = mat_manager.compute_packed_bed_properties(
            self.storage_material, self.packing_fraction
        )
        insul_props = mat_manager.get(self.insulation_material)
        steel_props = mat_manager.get(self.shell_material)
        air_props = mat_manager.get("air")
        concrete_props = mat_manager.get("concrete")
        
        # =================================================================
        # POSIZIONI ELEMENTI DISCRETI
        # =================================================================
        heater_elements = []
        tube_elements = []
        
        # RESISTENZE: nella zona storage con offset
        # Partono da: z_storage_start + offset_bottom
        # Arrivano a: z_storage_end - offset_top
        heater_z_start = cyl.z_storage_start + self.heaters.offset_bottom
        heater_z_end = cyl.z_storage_end - self.heaters.offset_top
        
        if self.heaters.pattern != HeaterPattern.UNIFORM_ZONE:
            heater_elements = self.heaters.generate_positions(
                cyl.center_x, cyl.center_y,
                0, cyl.r_storage * 0.9,
                heater_z_start, heater_z_end,
                phase_offset=cyl.phase_offset_rad
            )
        
        # TUBI: attraversano tutto, da slab bottom start a steel slab end
        # Vanno: slab_bottom -> storage -> slab_top -> steel_slab
        tube_z_start = cyl.z_slab_bottom_start
        tube_z_end = cyl.z_steel_slab_end  # = z_tubes_top
        
        tube_elements = self.tubes.generate_positions(
            cyl.center_x, cyl.center_y,
            cyl.r_storage * 0.9,
            tube_z_start, tube_z_end
        )
        
        # Calcola sorgente di calore per le resistenze
        if self.heaters.pattern == HeaterPattern.UNIFORM_ZONE:
            V_storage = np.pi * cyl.r_storage**2 * cyl.height
            Q_heaters = self.heaters.power_total * 1000 / V_storage
        else:
            if heater_elements:
                heater_length = heater_z_end - heater_z_start
                V_single = np.pi * self.heaters.heater_radius**2 * heater_length
                Q_heaters = (self.heaters.power_per_heater * 1000) / V_single
            else:
                Q_heaters = 0.0
        
        # =================================================================
        # VERSIONE VETTORIZZATA - usa NumPy broadcasting
        # =================================================================
        
        # Crea griglie 3D di coordinate
        X, Y, Z = np.meshgrid(mesh.x, mesh.y, mesh.z, indexing='ij')
        
        # Distanza radiale dal centro del cilindro
        R = np.sqrt((X - cyl.center_x)**2 + (Y - cyl.center_y)**2)
        
        # =====================================================================
        # 1. Inizializza tutto come AIR (default)
        # =====================================================================
        mesh.material_id[:] = MaterialID.AIR
        mesh.k[:] = air_props.k
        mesh.rho[:] = air_props.rho
        mesh.cp[:] = air_props.cp
        mesh.Q[:] = 0.0
        
        # =====================================================================
        # 2. Fondazione (sotto la struttura)
        # =====================================================================
        below_structure = Z < cyl.base_z
        mesh.material_id[below_structure] = MaterialID.CONCRETE
        mesh.k[below_structure] = concrete_props.k
        mesh.rho[below_structure] = concrete_props.rho
        mesh.cp[below_structure] = concrete_props.cp
        
        # =====================================================================
        # 3. ACCIAIO LATERALE (shell) - copre tutta l'altezza della struttura
        # =====================================================================
        in_shell_height = (Z >= cyl.base_z) & (Z <= cyl.z_shell_top)
        mask_shell = in_shell_height & (R >= cyl.r_insulation) & (R < cyl.r_shell)
        mesh.material_id[mask_shell] = MaterialID.STEEL
        mesh.k[mask_shell] = steel_props.k
        mesh.rho[mask_shell] = steel_props.rho
        mesh.cp[mask_shell] = steel_props.cp
        
        # =====================================================================
        # 4. ISOLAMENTO RADIALE - attorno a storage e slab isolanti
        # =====================================================================
        in_insul_height = (Z >= cyl.base_z) & (Z <= cyl.z_slab_top_end)
        mask_insul_radial = in_insul_height & (R >= cyl.r_storage) & (R < cyl.r_insulation)
        mesh.material_id[mask_insul_radial] = MaterialID.INSULATION
        mesh.k[mask_insul_radial] = insul_props.k
        mesh.rho[mask_insul_radial] = insul_props.rho
        mesh.cp[mask_insul_radial] = insul_props.cp
        
        # =====================================================================
        # 5. SLAB ISOLANTE INFERIORE
        # =====================================================================
        in_slab_bottom = (Z >= cyl.z_slab_bottom_start) & (Z < cyl.z_slab_bottom_end)
        mask_slab_bottom = in_slab_bottom & (R < cyl.r_storage)
        mesh.material_id[mask_slab_bottom] = MaterialID.INSULATION
        mesh.k[mask_slab_bottom] = insul_props.k
        mesh.rho[mask_slab_bottom] = insul_props.rho
        mesh.cp[mask_slab_bottom] = insul_props.cp
        
        # =====================================================================
        # 6. STORAGE (zona principale con sabbia)
        # =====================================================================
        in_storage = (Z >= cyl.z_storage_start) & (Z < cyl.z_storage_end)
        mask_storage = in_storage & (R < cyl.r_storage)
        mesh.material_id[mask_storage] = MaterialID.SAND
        mesh.k[mask_storage] = storage_props.k
        mesh.rho[mask_storage] = storage_props.rho
        mesh.cp[mask_storage] = storage_props.cp
        
        # Se pattern uniforme, applica Q a tutto lo storage
        if self.heaters.pattern == HeaterPattern.UNIFORM_ZONE:
            mesh.Q[mask_storage] = Q_heaters
        
        # =====================================================================
        # 7. SLAB ISOLANTE SUPERIORE
        # =====================================================================
        in_slab_top = (Z >= cyl.z_slab_top_start) & (Z < cyl.z_slab_top_end)
        mask_slab_top = in_slab_top & (R < cyl.r_storage)
        mesh.material_id[mask_slab_top] = MaterialID.INSULATION
        mesh.k[mask_slab_top] = insul_props.k
        mesh.rho[mask_slab_top] = insul_props.rho
        mesh.cp[mask_slab_top] = insul_props.cp
        
        # =====================================================================
        # 8. SLAB ACCIAIO (opzionale, sotto il tetto)
        # =====================================================================
        if cyl.steel_slab_top > 0:
            in_steel_slab = (Z >= cyl.z_slab_top_end) & (Z < cyl.z_steel_slab_end)
            mask_steel_slab = in_steel_slab & (R < cyl.r_insulation)
            mesh.material_id[mask_steel_slab] = MaterialID.STEEL
            mesh.k[mask_steel_slab] = steel_props.k
            mesh.rho[mask_steel_slab] = steel_props.rho
            mesh.cp[mask_steel_slab] = steel_props.cp
        
        # =====================================================================
        # 9. TETTO CONICO (acciaio) e riempimento sabbia opzionale
        # Solo se enable_cone_roof è True e roof_height > 0
        # =====================================================================
        if cyl.enable_cone_roof and cyl.roof_height > 0:
            # Calcola raggio del cono per ogni punto Z
            in_cone_region = (Z >= cyl.z_cone_base) & (Z <= cyl.z_cone_apex)
            
            # Raggio del cono a ogni quota: r_cone(z) = r_shell * (1 - (z - z_base) / h_cone)
            Z_rel = np.clip(Z - cyl.z_cone_base, 0, cyl.roof_height)
            R_cone = cyl.r_shell * (1 - Z_rel / cyl.roof_height)
            
            # Interno del cono
            inside_cone = in_cone_region & (R < R_cone)
            
            # Riempimento sabbia sotto il cono (opzionale)
            if cyl.fill_cone_with_sand:
                # La sabbia riempie tutto l'interno del cono
                mesh.material_id[inside_cone] = MaterialID.SAND
                mesh.k[inside_cone] = storage_props.k
                mesh.rho[inside_cone] = storage_props.rho
                mesh.cp[inside_cone] = storage_props.cp
            
            # Tetto conico in acciaio (guscio sottile sulla superficie del cono)
            # Approssimiamo come le celle sul bordo del cono
            cone_shell_thickness = cyl.shell_thickness
            R_cone_inner = cyl.r_shell * (1 - Z_rel / cyl.roof_height) - cone_shell_thickness
            R_cone_inner = np.maximum(R_cone_inner, 0)
            
            mask_cone_shell = in_cone_region & (R >= R_cone_inner) & (R < R_cone)
            mesh.material_id[mask_cone_shell] = MaterialID.STEEL
            mesh.k[mask_cone_shell] = steel_props.k
            mesh.rho[mask_cone_shell] = steel_props.rho
            mesh.cp[mask_cone_shell] = steel_props.cp
        
        # =====================================================================
        # 10. Elementi discreti (tubi e resistenze) - VERSIONE VETTORIZZATA
        # =====================================================================
        # OTTIMIZZAZIONE FASE 2: Broadcasting NumPy invece di loop Python
        # --------------------------------------------------------------------------
        # PROBLEMA ORIGINALE:
        # Il loop Python su N elementi richiede N*M operazioni dove M = mesh size
        # Per 100 elementi e mesh 100³ = 100M operazioni in Python (lento)
        #
        # SOLUZIONE:
        # 1. Estrai coordinate (x, y, z_bottom, z_top, radius) in array 1D
        # 2. Usa broadcasting 4D: (Nx, Ny, Nz, N_elements) per calcolare
        #    tutte le distanze in un'unica operazione vettorizzata
        # 3. any(axis=-1) per ridurre a maschera 3D
        #
        # SPEEDUP ATTESO: 5-20x rispetto al loop Python
        # TRADE-OFF: Usa più memoria temporanea (O(mesh_size * n_elements))
        # Per mesh molto grandi con molti elementi, potrebbe essere necessario
        # partizionare in batch per evitare memory overflow.
        # =====================================================================
        
        # -------------------------------------------------------------------------
        # RESISTENZE DISCRETE (vettorizzato)
        # -------------------------------------------------------------------------
        if heater_elements:
            n_heaters = len(heater_elements)
            
            # Estrai coordinate in array NumPy per broadcasting
            heater_x = np.array([h.x for h in heater_elements])        # (n_heaters,)
            heater_y = np.array([h.y for h in heater_elements])
            heater_r = np.array([h.radius for h in heater_elements])
            heater_zb = np.array([h.z_bottom for h in heater_elements])
            heater_zt = np.array([h.z_top for h in heater_elements])
            
            # Broadcasting 4D: X(Nx,Ny,Nz,1) - heater_x(1,1,1,n_heaters)
            # Risultato: (Nx, Ny, Nz, n_heaters)
            dx_h = X[:, :, :, np.newaxis] - heater_x[np.newaxis, np.newaxis, np.newaxis, :]
            dy_h = Y[:, :, :, np.newaxis] - heater_y[np.newaxis, np.newaxis, np.newaxis, :]
            dist_xy_h = np.sqrt(dx_h**2 + dy_h**2)  # (Nx, Ny, Nz, n_heaters)
            
            # Maschera per ogni elemento: distanza <= raggio AND z in range
            Z_4d = Z[:, :, :, np.newaxis]  # (Nx, Ny, Nz, 1)
            in_radius_h = dist_xy_h <= heater_r[np.newaxis, np.newaxis, np.newaxis, :]
            in_z_range_h = (Z_4d >= heater_zb[np.newaxis, np.newaxis, np.newaxis, :]) & \
                           (Z_4d <= heater_zt[np.newaxis, np.newaxis, np.newaxis, :])
            
            # Riduzione: any(axis=-1) -> True se il nodo appartiene ad almeno 1 heater
            mask_any_heater = np.any(in_radius_h & in_z_range_h, axis=-1)  # (Nx, Ny, Nz)
            
            # Applica proprietà a tutte le celle delle resistenze in una volta
            mesh.material_id[mask_any_heater] = MaterialID.HEATERS
            mesh.k[mask_any_heater] = storage_props.k
            mesh.rho[mask_any_heater] = storage_props.rho
            mesh.cp[mask_any_heater] = storage_props.cp
            mesh.Q[mask_any_heater] = Q_heaters
        
        # -------------------------------------------------------------------------
        # TUBI DISCRETI (vettorizzato)
        # -------------------------------------------------------------------------
        if tube_elements:
            n_tubes = len(tube_elements)
            
            # Estrai coordinate in array NumPy per broadcasting
            tube_x = np.array([t.x for t in tube_elements])        # (n_tubes,)
            tube_y = np.array([t.y for t in tube_elements])
            tube_r = np.array([t.radius for t in tube_elements])
            tube_zb = np.array([t.z_bottom for t in tube_elements])
            tube_zt = np.array([t.z_top for t in tube_elements])
            tube_h = np.array([t.h_fluid for t in tube_elements])
            tube_T = np.array([t.T_fluid for t in tube_elements])
            
            # Broadcasting 4D: X(Nx,Ny,Nz,1) - tube_x(1,1,1,n_tubes)
            dx_t = X[:, :, :, np.newaxis] - tube_x[np.newaxis, np.newaxis, np.newaxis, :]
            dy_t = Y[:, :, :, np.newaxis] - tube_y[np.newaxis, np.newaxis, np.newaxis, :]
            dist_xy_t = np.sqrt(dx_t**2 + dy_t**2)  # (Nx, Ny, Nz, n_tubes)
            
            # Maschera per ogni elemento
            Z_4d = Z[:, :, :, np.newaxis]  # (Nx, Ny, Nz, 1)
            in_radius_t = dist_xy_t <= tube_r[np.newaxis, np.newaxis, np.newaxis, :]
            in_z_range_t = (Z_4d >= tube_zb[np.newaxis, np.newaxis, np.newaxis, :]) & \
                           (Z_4d <= tube_zt[np.newaxis, np.newaxis, np.newaxis, :])
            
            # Maschera combinata per ogni tubo: (Nx, Ny, Nz, n_tubes)
            mask_per_tube = in_radius_t & in_z_range_t
            
            # Riduzione: any(axis=-1) -> True se il nodo appartiene ad almeno 1 tubo
            mask_any_tube = np.any(mask_per_tube, axis=-1)  # (Nx, Ny, Nz)
            
            # Applica proprietà comuni a tutti i tubi
            mesh.material_id[mask_any_tube] = MaterialID.TUBES
            mesh.k[mask_any_tube] = storage_props.k
            mesh.rho[mask_any_tube] = storage_props.rho
            mesh.cp[mask_any_tube] = storage_props.cp
            
            # Per le condizioni al contorno, serve sapere a QUALE tubo appartiene
            # ogni cella. Usiamo argmax per trovare il primo tubo che contiene il nodo.
            if self.tubes.active:
                # argmax restituisce l'indice del primo True lungo l'asse
                # Ma funziona solo dove mask_any_tube è True
                tube_indices = np.argmax(mask_per_tube, axis=-1)  # (Nx, Ny, Nz)
                
                # Costruisci array h_fluid e T_fluid per tutte le celle
                # h_fluid[i,j,k] = tube_h[tube_indices[i,j,k]] dove mask_any_tube[i,j,k]
                mesh.bc_h[mask_any_tube] = tube_h[tube_indices[mask_any_tube]]
                mesh.bc_T_inf[mask_any_tube] = tube_T[tube_indices[mask_any_tube]]
                mesh.boundary_type[mask_any_tube] = BoundaryType.CONVECTION
            else:
                mesh.bc_h[mask_any_tube] = 0.0
                mesh.boundary_type[mask_any_tube] = BoundaryType.INTERNAL
        
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
        
        # Volumi principali
        h_storage = cyl.z_storage_end - cyl.z_storage_start
        h_slab_bottom = cyl.insulation_slab_bottom
        h_slab_top = cyl.insulation_slab_top
        h_shell = cyl.z_shell_top - cyl.base_z
        
        volumes = {
            'storage': np.pi * cyl.r_storage**2 * h_storage,
            'slab_bottom': np.pi * cyl.r_storage**2 * h_slab_bottom,
            'slab_top': np.pi * cyl.r_storage**2 * h_slab_top,
            'insulation_radial': np.pi * (cyl.r_insulation**2 - cyl.r_storage**2) * h_shell,
            'shell': np.pi * (cyl.r_shell**2 - cyl.r_insulation**2) * h_shell,
        }
        
        # Volume del tetto conico (V = 1/3 * π * r² * h)
        if cyl.roof_height > 0:
            volumes['cone'] = (1/3) * np.pi * cyl.r_shell**2 * cyl.roof_height
        else:
            volumes['cone'] = 0.0
            
        # Slab acciaio opzionale
        if cyl.steel_slab_top > 0:
            volumes['steel_slab'] = np.pi * cyl.r_insulation**2 * cyl.steel_slab_top
        else:
            volumes['steel_slab'] = 0.0
        
        volumes['insulation_total'] = (
            volumes['slab_bottom'] + 
            volumes['slab_top'] + 
            volumes['insulation_radial']
        )
        
        volumes['total'] = (
            volumes['storage'] + 
            volumes['insulation_total'] + 
            volumes['shell'] + 
            volumes['cone'] + 
            volumes['steel_slab']
        )
        
        return volumes
    
    def get_zone_masses(self, mat_manager: MaterialManager) -> dict:
        """Calcola le masse di ogni zona [kg]"""
        volumes = self.get_zone_volumes()
        
        storage_props = mat_manager.compute_packed_bed_properties(
            self.storage_material, self.packing_fraction
        )
        insul_props = mat_manager.get(self.insulation_material)
        steel_props = mat_manager.get(self.shell_material)
        
        masses = {
            'storage': volumes['storage'] * storage_props.rho,
            'slab_bottom': volumes['slab_bottom'] * insul_props.rho,
            'slab_top': volumes['slab_top'] * insul_props.rho,
            'insulation_radial': volumes['insulation_radial'] * insul_props.rho,
            'shell': volumes['shell'] * steel_props.rho,
            'cone': volumes['cone'] * steel_props.rho,
            'steel_slab': volumes['steel_slab'] * steel_props.rho,
        }
        
        masses['insulation_total'] = (
            masses['slab_bottom'] + 
            masses['slab_top'] + 
            masses['insulation_radial']
        )
        
        masses['total'] = sum(v for k, v in masses.items() 
                              if k not in ['insulation_total'])
        
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
        storage_props = mat_manager.compute_packed_bed_properties(
            self.storage_material, self.packing_fraction
        )
        
        delta_T = T_high - T_low
        
        # Energia termica totale
        E_thermal_J = masses['storage'] * storage_props.cp * delta_T
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
    """
    Crea geometria piccola per test (8 MWh circa).
    
    Usa la nuova struttura a 4 zone:
    - STORAGE: r=2.0m (materiale di accumulo con tubi e resistenze)
    - INSULATION: spessore 0.3m
    - STEEL: spessore 0.02m
    """
    
    cylinder = CylinderGeometry(
        center_x=3.0,
        center_y=3.0,
        base_z=0.3,
        height=4.0,
        r_storage=2.0,
        insulation_thickness=0.3,
        shell_thickness=0.02,
        phase_offset_deg=15.0,
    )
    
    return BatteryGeometry(
        cylinder=cylinder,
        heaters=HeaterConfig(power_total=50),
        tubes=TubeConfig(n_tubes=6),
        storage_material="steatite",
        packing_fraction=0.63,
    )
