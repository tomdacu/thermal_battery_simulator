"""
transient_results_widget.py - Widget per visualizzazione risultati transitorio

=============================================================================
TRANSIENT RESULTS VISUALIZATION
=============================================================================

Widget per visualizzare i risultati della simulazione transitoria:
- Grafici temperatura vs tempo
- Grafici potenza vs tempo
- Animazione evoluzione campo T

=============================================================================
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox,
    QSlider, QCheckBox, QSplitter, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

import numpy as np

# Import matplotlib per i grafici (opzionale)
try:
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    FigureCanvas = None


class TransientPlotWidget(QWidget):
    """Widget per grafici risultati transitorio"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._results = None
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        if not HAS_MATPLOTLIB:
            layout.addWidget(QLabel("Matplotlib non disponibile.\nInstalla con: pip install matplotlib"))
            return
        
        # Sub-tabs per diversi grafici
        self.tabs = QTabWidget()
        
        # Tab Temperature
        temp_widget = QWidget()
        temp_layout = QVBoxLayout(temp_widget)
        
        self.fig_temp = Figure(figsize=(8, 5))
        self.canvas_temp = FigureCanvas(self.fig_temp)
        temp_layout.addWidget(self.canvas_temp)
        
        self.tabs.addTab(temp_widget, "Temperatura")
        
        # Tab Potenza/Energia
        power_widget = QWidget()
        power_layout = QVBoxLayout(power_widget)
        
        self.fig_power = Figure(figsize=(8, 5))
        self.canvas_power = FigureCanvas(self.fig_power)
        power_layout.addWidget(self.canvas_power)
        
        self.tabs.addTab(power_widget, "Potenza")
        
        # Tab Energia
        energy_widget = QWidget()
        energy_layout = QVBoxLayout(energy_widget)
        
        self.fig_energy = Figure(figsize=(8, 5))
        self.canvas_energy = FigureCanvas(self.fig_energy)
        energy_layout.addWidget(self.canvas_energy)
        
        self.tabs.addTab(energy_widget, "Energia")
        
        layout.addWidget(self.tabs)
    
    def set_results(self, results):
        """Imposta i risultati da visualizzare"""
        self._results = results
        self._update_plots()
    
    def _update_plots(self):
        """Aggiorna tutti i grafici"""
        if not HAS_MATPLOTLIB or self._results is None:
            return
        
        self._plot_temperature()
        self._plot_power()
        self._plot_energy()
    
    def _plot_temperature(self):
        """Grafico temperatura vs tempo"""
        r = self._results
        
        self.fig_temp.clear()
        ax = self.fig_temp.add_subplot(111)
        
        # Converti tempo in ore
        t_hours = np.array(r.times) / 3600
        
        # Converti temperature in Celsius
        T_mean_C = np.array(r.T_mean) - 273.15
        T_min_C = np.array(r.T_min) - 273.15
        T_max_C = np.array(r.T_max) - 273.15
        
        ax.plot(t_hours, T_mean_C, 'b-', linewidth=2, label='T media')
        ax.fill_between(t_hours, T_min_C, T_max_C, alpha=0.3, color='blue', label='Range T')
        
        ax.set_xlabel('Tempo [ore]')
        ax.set_ylabel('Temperatura [°C]')
        ax.set_title('Evoluzione Temperatura')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig_temp.tight_layout()
        self.canvas_temp.draw()
    
    def _plot_power(self):
        """Grafico potenza vs tempo"""
        r = self._results
        
        self.fig_power.clear()
        ax = self.fig_power.add_subplot(111)
        
        t_hours = np.array(r.times) / 3600
        
        # Se disponibili dati potenza
        if hasattr(r, 'P_heaters') and r.P_heaters:
            P_kW = np.array(r.P_heaters) / 1000
            ax.plot(t_hours, P_kW, 'r-', linewidth=2, label='P ingresso')
        
        if hasattr(r, 'Q_losses_total') and r.Q_losses_total:
            Q_kW = np.array(r.Q_losses_total) / 1000
            ax.plot(t_hours, Q_kW, 'g--', linewidth=2, label='Perdite')
        
        ax.set_xlabel('Tempo [ore]')
        ax.set_ylabel('Potenza [kW]')
        ax.set_title('Potenze')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig_power.tight_layout()
        self.canvas_power.draw()
    
    def _plot_energy(self):
        """Grafico energia cumulativa"""
        r = self._results
        
        self.fig_energy.clear()
        ax = self.fig_energy.add_subplot(111)
        
        t_hours = np.array(r.times) / 3600
        
        if hasattr(r, 'E_in_cumulative') and r.E_in_cumulative:
            E_kWh = np.array(r.E_in_cumulative) / 3.6e6
            ax.plot(t_hours, E_kWh, 'r-', linewidth=2, label='E ingresso')
        
        if hasattr(r, 'E_stored') and r.E_stored:
            E_stored_kWh = np.array(r.E_stored) / 3.6e6
            ax.plot(t_hours, E_stored_kWh, 'b-', linewidth=2, label='E immagazzinata')
        
        if hasattr(r, 'E_losses_cumulative') and r.E_losses_cumulative:
            E_loss_kWh = np.array(r.E_losses_cumulative) / 3.6e6
            ax.plot(t_hours, E_loss_kWh, 'g--', linewidth=2, label='Perdite cumulative')
        
        ax.set_xlabel('Tempo [ore]')
        ax.set_ylabel('Energia [kWh]')
        ax.set_title('Bilancio Energetico')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig_energy.tight_layout()
        self.canvas_energy.draw()


class AnimationControlWidget(QWidget):
    """Widget per controllare animazione campo T"""
    
    time_changed = pyqtSignal(int)  # Emette indice tempo
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_timer)
        self._playing = False
        self._max_frame = 0
        self._current_frame = 0
    
    def _init_ui(self):
        layout = QHBoxLayout(self)
        
        # Play/Pause
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_btn)
        
        # Stop
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._stop)
        layout.addWidget(self.stop_btn)
        
        # Slider tempo
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.time_slider, stretch=1)
        
        # Label tempo
        self.time_label = QLabel("t = 0 s")
        layout.addWidget(self.time_label)
        
        # Velocità
        layout.addWidget(QLabel("Velocità:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "1x", "2x", "4x", "10x"])
        self.speed_combo.setCurrentIndex(1)
        layout.addWidget(self.speed_combo)
    
    def set_n_frames(self, n: int, times: np.ndarray = None):
        """Imposta numero di frame e tempi"""
        self._max_frame = max(0, n - 1)
        self.time_slider.setRange(0, self._max_frame)
        self._times = times
    
    def _toggle_play(self):
        if self._playing:
            self._timer.stop()
            self.play_btn.setText("▶ Play")
            self._playing = False
        else:
            # Calcola intervallo in base a velocità
            speed_map = {0: 200, 1: 100, 2: 50, 3: 25, 4: 10}
            interval = speed_map.get(self.speed_combo.currentIndex(), 100)
            self._timer.start(interval)
            self.play_btn.setText("⏸ Pause")
            self._playing = True
    
    def _stop(self):
        self._timer.stop()
        self._playing = False
        self.play_btn.setText("▶ Play")
        self._current_frame = 0
        self.time_slider.setValue(0)
    
    def _on_timer(self):
        """Avanza di un frame"""
        self._current_frame += 1
        if self._current_frame > self._max_frame:
            self._current_frame = 0
        self.time_slider.setValue(self._current_frame)
    
    def _on_slider_changed(self, value: int):
        """Callback quando slider cambia"""
        self._current_frame = value
        
        # Aggiorna label
        if hasattr(self, '_times') and self._times is not None and len(self._times) > value:
            t = self._times[value]
            self.time_label.setText(f"t = {t:.0f} s")
        else:
            self.time_label.setText(f"Frame {value}")
        
        self.time_changed.emit(value)
