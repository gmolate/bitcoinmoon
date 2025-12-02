import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import shutil
import yfinance as yf
import warnings
import threading
import queue
import webbrowser

# --- Global Configuration --- 

# Matplotlib style
plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1e1e1e'
plt.rcParams['axes.facecolor'] = '#1e1e1e'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.labelcolor'] = '#aaaaaa'
plt.rcParams['text.color'] = '#dddddd'
plt.rcParams['xtick.color'] = '#aaaaaa'
plt.rcParams['ytick.color'] = '#aaaaaa'
plt.rcParams['grid.color'] = '#333333'

# API Availability Checks
try:
    import polygon
    POLYGON_AVAILABLE = True
except ImportError:
    POLYGON_AVAILABLE = False

try:
    from astropy.time import Time
    from astropy.coordinates import get_body, get_sun
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

# Emoji and Font configuration
try:
    warnings.filterwarnings("ignore", message=".*missing from font.*")
    warnings.filterwarnings("ignore", message=".*Glyph.*missing.*")
    plt.rcParams['font.family'] = ['Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    import matplotlib
    if matplotlib.get_backend() != 'TkAgg':
        matplotlib.use('TkAgg')
except Exception as e:
    print(f"Advertencia: No se pudo configurar fuentes emoji: {e}")

# Historical Data
HALVINGS = {
    '2012': {'date': '2012-11-28', 'price': 12.35, 'low_pre': 2, 'low_date': '2011-07', 'ath_post': 1150, 'months': 12, 'multi': 96, 'start': '2011-01-01', 'end': '2016-01-01'},
    '2016': {'date': '2016-07-09', 'price': 638.51, 'low_pre': 200, 'low_date': '2015-01', 'ath_post': 20000, 'months': 17, 'multi': 30, 'start': '2015-01-01', 'end': '2020-01-01'},
    '2020': {'date': '2020-05-11', 'price': 8475, 'low_pre': 3800, 'low_date': '2020-03', 'ath_post': 69000, 'months': 18, 'multi': 7.6, 'start': '2020-01-01', 'end': '2024-01-01'},
    '2024': {'date': '2024-04-20', 'price': 61067, 'low_pre': 15760, 'low_date': '2022-11', 'ath_post': None, 'months': None, 'multi': 'x3-4', 'start': '2021-01-01', 'end': '2026-01-01'}
}

# --- Helper Functions ---

def fetch_polygon_data(start='2012-01-01', end=None):
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    client = polygon.RESTClient()
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    aggs = client.get_aggs(ticker="X:BTCUSD", multiplier=1, timespan="day", from_=start_date, to=end_date)
    df = pd.DataFrame(aggs)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]

def calculate_cmf(df, period=20):
    df = df.copy()
    high_low = df['high'] - df['low']
    high_low = high_low.replace(0, np.nan)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / high_low
    mfm = mfm.fillna(0)
    mfv = mfm * df['volume']
    rolling_mfv = mfv.rolling(window=period, min_periods=1).sum()
    rolling_vol = df['volume'].rolling(window=period, min_periods=1).sum()
    cmf = rolling_mfv / rolling_vol.replace(0, np.nan)
    return cmf.fillna(0)

def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_free_crypto_data():
    import requests
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {'vs_currency': 'usd', 'days': 'max', 'interval': 'daily'}
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
    volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
    df = pd.merge(prices, volumes, on='timestamp')
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('date', inplace=True)
    df['open'] = df['high'] = df['low'] = df['close']
    return df[['open', 'high', 'low', 'close', 'volume']]

def fetch_yahoo_data(period="max"):
    ticker = yf.Ticker("BTC-USD")
    df = ticker.history(period=period)
    if df.empty: return None
    df.index = df.index.tz_localize(None)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    return df[['open', 'high', 'low', 'close', 'volume']]

def get_moon_phases(df):
    phases = []
    if not ASTROPY_AVAILABLE:
        lunar_cycle = 29.5
        reference_new_moon = pd.to_datetime('2024-01-11')
        for date in df.index:
            days_since_ref = (date - reference_new_moon).days
            cycle_position = (days_since_ref % lunar_cycle) / lunar_cycle
            if abs(cycle_position) < 0.03 or abs(cycle_position - 1) < 0.03:
                phases.append((date, 0.0))
            elif abs(cycle_position - 0.5) < 0.03:
                phases.append((date, 0.5))
        return phases
    for date in df.index[::1]:
        try:
            t = Time(str(date.date()))
            elongation = get_sun(t).separation(get_body('moon', t))
            if elongation.deg < 10 or elongation.deg > 350:
                phases.append((date, 0.0))
            elif 170 < elongation.deg < 190:
                phases.append((date, 0.5))
        except Exception:
            pass
    return phases

def generate_analysis_text(df, moon_phases, halvings=HALVINGS):
    if df.empty: return "No se han cargado datos."
    low_idx, low_price = df['close'].idxmin(), df['close'].min()
    current_price = df['close'].iloc[-1]
    current_cmf = df['cmf'].iloc[-1] if 'cmf' in df else np.nan
    current_rsi = df['rsi'].iloc[-1] if 'rsi' in df else np.nan
    volatility_30d = df['close'].pct_change().rolling(30).std().iloc[-1] * np.sqrt(365) if len(df) > 30 else 0
    days_since_halving = (df.index[-1] - pd.to_datetime(halvings['2024']['date'])).days
    halving_price = df.loc['2024-04-20':'2024-04-20', 'close'].iloc[0] if '2024-04-20' in df.index else halvings['2024']['price']
    proj_2026 = halving_price * 3.5
    valid_moon_dates = [p[0] for p in moon_phases if p[0] in df.index]
    vol_near_moon = df.loc[valid_moon_dates, 'close'].pct_change().std() if valid_moon_dates else np.nan
    
    table = """| Ciclo | Fecha Mínimo | Precio Mínimo | Fecha Halving | Precio Halving | ATH Post (Meses) | Multiplicador |
|-------|--------------|---------------|---------------|----------------|------------------|---------------|
"""
    for year, data in halvings.items():
        ath = f"Proy: ${proj_2026:.0f}" if year == '2024' else f"${data['ath_post']}"
        months = data['months'] or "18m proy"
        table += f"| {year}-{int(year)+4} | {data['low_date']} | ~${data['low_pre']} | {data['date']} | ~${data['price']} | {ath} ({months}) | x{data['multi']} |\n"
    
    analysis = f"""### Análisis Técnico y de Ciclos (al {df.index[-1].date()})
#### Métricas Actuales
- Precio Actual: ~${current_price:.0f} USD
- Mínimo del Ciclo Actual: {low_price:.0f} USD ({low_idx.date()})
- CMF Actual (20d): {current_cmf:.3f} (Salida de capital si < 0)
- RSI Actual (14d): {current_rsi:.2f}
- Volatilidad (30d): {volatility_30d:.2%}
- Días desde Halving 2024: {days_since_halving}
- Fases Lunares en Datos: {len(valid_moon_dates)} (Volatilidad ~ fases: {vol_near_moon:.2%})
#### Tabla de Ciclos Históricos
{table}"""
    return analysis

# --- Main GUI Class ---
class BTCGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BTC Analyzer - Bloomberg Style Terminal")
        self.root.geometry("1600x1200")
        
        self.data_queue = queue.Queue()
        self.loading_buttons = []

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.current_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.current_tab, text="Ciclo Actual (4 años)")
        self.setup_current_cycle_tab()
        
        self.total_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.total_tab, text="Total (12 años)")
        self.setup_total_tab()
        
        self.cycles_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.cycles_tab, text="Ciclos")
        self.setup_cycles_tab()
        
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Análisis (Asistido por IA)")
        self.setup_analysis_tab()
        
        log_frame = ttk.LabelFrame(root, text="Terminal Logs")
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=10)
        ttk.Button(log_frame, text="Limpiar Logs", command=self.clear_logs).pack(side=tk.RIGHT, padx=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=4, width=100)
        self.log_text.pack(fill=tk.X, padx=5, pady=5)
        
        self.df = None
        self.moon_phases = []
        self.current_zoom_start, self.current_zoom_end = None, None
        self.current_toolbar, self.total_toolbar, self.cycle_toolbar = None, None, None

        self.process_queue()

    def setup_analysis_tab(self):
        llm_frame = ttk.LabelFrame(self.analysis_tab, text="Asistente de Análisis con IA")
        llm_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(llm_frame, text="Selecciona un servicio de IA:").pack(side=tk.LEFT, padx=(5,0))
        
        self.llm_var = tk.StringVar()
        llm_options = ["Gemini", "ChatGPT", "Grok", "Kimi", "DeepSeek", "Qwen"]
        self.llm_dropdown = ttk.Combobox(llm_frame, textvariable=self.llm_var, values=llm_options, state="readonly", width=15)
        self.llm_dropdown.pack(side=tk.LEFT, padx=5)
        self.llm_dropdown.set(llm_options[0])
        
        ttk.Button(llm_frame, text="1. Acceder al Servicio", command=self.access_llm_service).pack(side=tk.LEFT, padx=5)
        ttk.Button(llm_frame, text="2. Copiar Prompt de Análisis", command=self.send_analysis_to_llm).pack(side=tk.LEFT, padx=5)
        
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_tab, height=20, width=100, wrap=tk.WORD, bg="#1e1e1e", fg="#dddddd")
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def access_llm_service(self):
        selected_llm = self.llm_var.get()
        if not selected_llm or selected_llm == "Seleccionar...":
            messagebox.showwarning("Advertencia", "Por favor, selecciona un servicio de IA.")
            return
            
        llm_urls = {
            "Grok": "https://grok.x.ai/",
            "Gemini": "https://gemini.google.com/",
            "ChatGPT": "https://chat.openai.com/",
            "Kimi": "https://kimi.moonshot.cn/",
            "DeepSeek": "https://chat.deepseek.com/",
            "Qwen": "https://tongyi.aliyun.com/qwen/"
        }
        
        url = llm_urls.get(selected_llm)
        if url:
            self.log(f"Abriendo navegador para acceder a {selected_llm}...")
            webbrowser.open(url, new=2)
            messagebox.showinfo("Acceso", f"Se ha abierto una pestaña en tu navegador para {selected_llm}.\n\nPor favor, inicia sesión si es necesario.")
        else:
            messagebox.showerror("Error", f"No se encontró la URL para {selected_llm}")
    
    def send_analysis_to_llm(self):
        analysis_content = self.analysis_text.get("1.0", tk.END).strip()
        if not analysis_content or "No se han cargado datos" in analysis_content:
            messagebox.showwarning("Advertencia", "No hay datos de análisis para copiar. Carga una fuente de datos primero.")
            return
            
        prompt = f"""Como analista financiero experto, por favor proporciona un análisis detallado y una estrategia de trading basada en los siguientes datos de mercado para Bitcoin.

### INFORME DE DATOS
{analysis_content}
### FIN DEL INFORME

Análisis Solicitado:
1.  **Resumen Ejecutivo:** ¿Cuál es el sentimiento general del mercado (alcista, bajista, neutral) y por qué?
2.  **Interpretación de Indicadores:** ¿Qué señalan específicamente el CMF y el RSI en este contexto?
3.  **Riesgos y Oportunidades:** Basado en los datos, ¿cuáles son los 1-2 riesgos más inmediatos y las 1-2 oportunidades más claras?
4.  **Estrategia Sugerida:** Proporciona 2-3 puntos de acción claros para un trader a corto/mediano plazo.
"""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(prompt)
            self.root.update()
            self.log(f"Prompt de análisis copiado al portapapeles.")
            messagebox.showinfo("Copiado al Portapapeles", "El prompt con el informe de datos se ha copiado.\n\nPégalo en la ventana del servicio de IA que abriste.")
        except Exception as e:
            self.log(f"Error copiando al portapapeles: {e}")
            messagebox.showerror("Error", f"No se pudo copiar al portapapeles: {e}")

    def setup_current_cycle_tab(self):
        frame_source = ttk.Frame(self.current_tab)
        frame_source.pack(pady=10)
        
        buttons_config = [
            ("Cargar Polygon", self.load_polygon, POLYGON_AVAILABLE),
            ("Cargar Kaggle", self.load_kaggle, KAGGLE_AVAILABLE),
            ("Cargar CSV", self.load_custom, True),
            ("Yahoo Finance", self.load_yahoo, True),
            ("CoinGecko", self.load_free_data, True),
        ]
        for text, command, available in buttons_config:
            btn = ttk.Button(frame_source, text=text, command=command)
            if not available: btn.config(state='disabled')
            btn.pack(side=tk.LEFT, padx=5)
            if available: self.loading_buttons.append(btn)

        ttk.Button(frame_source, text="Exportar Gráfico", command=self.export_current_plot).pack(side=tk.LEFT, padx=5)
        # Botón para inicializar archivos básicos del repositorio GitHub
        ttk.Button(frame_source, text="Inicializar GitHub", command=self.create_github_repo_files).pack(side=tk.LEFT, padx=5)
        
        zoom_frame = ttk.LabelFrame(self.current_tab, text="Zoom & Períodos")
        zoom_frame.pack(fill=tk.X, padx=10, pady=5)
        for period, days in {"1M": 30, "3M": 90, "6M": 180, "1A": 365, "2A": 730}.items():
            ttk.Button(zoom_frame, text=period, command=lambda d=days: self.set_period_current(d)).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Todo", command=self.reset_zoom_current).pack(side=tk.LEFT, padx=2)
        
        frame_ind = ttk.Frame(self.current_tab)
        frame_ind.pack(pady=5)
        self.show_cmf_current = tk.BooleanVar(value=True)
        self.show_rsi_current = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame_ind, text="CMF", variable=self.show_cmf_current, command=self.update_current_plot).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(frame_ind, text="RSI", variable=self.show_rsi_current, command=self.update_current_plot).pack(side=tk.LEFT, padx=5)
        
        self.current_plot_frame = ttk.Frame(self.current_tab)
        self.current_plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.current_fig = Figure(figsize=(14, 8))
        self.current_canvas = None

    def setup_total_tab(self):
        self.total_plot_frame = ttk.Frame(self.total_tab)
        self.total_plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.total_fig = Figure(figsize=(14, 8))
        self.total_canvas = None
        ttk.Button(self.total_tab, text="Mostrar Gráfico Completo", command=self.update_total_plot).pack(pady=10)

    def setup_cycles_tab(self):
        cycle_frame = ttk.LabelFrame(self.cycles_tab, text="Seleccionar Ciclo")
        cycle_frame.pack(fill=tk.X, padx=10, pady=5)
        self.cycle_var = tk.StringVar(value='2024')
        for year in ['2012', '2016', '2020', '2024']:
            ttk.Radiobutton(cycle_frame, text=f"Ciclo {year}", variable=self.cycle_var, value=year, command=self.update_cycle_plot).pack(side=tk.LEFT, padx=10)
        
        self.cycle_plot_frame = ttk.Frame(self.cycles_tab)
        self.cycle_plot_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.cycle_fig = Figure(figsize=(14, 8))
        self.cycle_canvas = None

    def set_period_current(self, days):
        if self.df is None: return
        end_date = self.df.index[-1]
        start_date = end_date - timedelta(days=days)
        self.current_zoom_start, self.current_zoom_end = start_date, end_date
        self.update_current_plot()

    def reset_zoom_current(self):
        self.current_zoom_start, self.current_zoom_end = None, None
        self.update_current_plot()

    def update_current_plot(self):
        if self.df is None: self.log("No hay datos para mostrar."); return
        df_plot = self.df.copy()
        if self.current_zoom_start and self.current_zoom_end:
            df_plot = df_plot.loc[self.current_zoom_start:self.current_zoom_end]
        else:
             df_plot = df_plot[df_plot.index >= pd.to_datetime(HALVINGS['2024']['start'])]
        
        title = f"Ciclo Actual BTC ({df_plot.index.min().year} - {df_plot.index.max().year})" if not df_plot.empty else "Ciclo Actual BTC"
        self.current_canvas = self._plot_data(df_plot, self.current_fig, self.current_plot_frame, self.current_canvas, self.show_cmf_current.get(), self.show_rsi_current.get(), title)

    def update_total_plot(self):
        if self.df is None: self.log("No hay datos para mostrar."); return
        self.total_canvas = self._plot_data(self.df, self.total_fig, self.total_plot_frame, self.total_canvas, True, False, "BTC Histórico Completo")

    def update_cycle_plot(self):
        if self.df is None: self.log("No hay datos para mostrar."); return
        cycle = self.cycle_var.get()
        cycle_data = HALVINGS[cycle]
        df_cycle = self.df.loc[cycle_data['start']:cycle_data['end']]
        if df_cycle.empty: self.log(f"No hay datos para el ciclo {cycle}."); return
        title = f"Ciclo {cycle} ({cycle_data['start'][:4]}-{cycle_data['end'][:4]})"
        self.cycle_canvas = self._plot_data(df_cycle, self.cycle_fig, self.cycle_plot_frame, self.cycle_canvas, True, True, title)

    def _plot_data(self, df, fig, frame, canvas, show_cmf, show_rsi, title):
        fig.clear()
        if df.empty:
            ax1 = fig.add_subplot(111)
            ax1.set_title(title, fontsize=12, weight='bold')
            ax1.text(0.5, 0.5, "No hay datos para el período seleccionado.", ha='center', va='center', color='yellow')
        else:
            ax1 = fig.add_subplot(211) if (show_cmf or show_rsi) else fig.add_subplot(111)
            ax1.plot(df.index, df['close'], label='BTC Close', color='cyan', linewidth=1.5)
            ax1.set_ylabel('Precio (USD)', fontsize=10)
            ax1.set_title(title, fontsize=12, weight='bold')
            ax1.legend(loc='upper left')
            ax1.grid(True, linestyle='--', alpha=0.3)
            y_min, y_max = df['low'].min(), df['high'].max()
            ax1.set_ylim(y_min * 0.95, y_max * 1.05)
            valid_moon_dates = [p[0] for p in self.moon_phases if p[0] in df.index]
            for date, phase in self.moon_phases:
                if date in valid_moon_dates:
                    emoji, color = ('🌑', 'lightblue') if phase == 0.0 else ('🌕', 'orange')
                    ax1.axvline(x=date, color=color, alpha=0.5, linestyle=':', linewidth=1)
            if show_cmf or show_rsi:
                ax2 = fig.add_subplot(212, sharex=ax1)
                if show_cmf and 'cmf' in df:
                    ax2.plot(df.index, df['cmf'], label='CMF (20)', color='red', linewidth=1)
                    ax2.axhline(0, color='white', linestyle='--', alpha=0.5)
                    ax2.set_ylabel('CMF', color='red')
                if show_rsi and 'rsi' in df:
                    ax2_rsi = ax2.twinx() if show_cmf else ax2
                    ax2_rsi.plot(df.index, df['rsi'], label='RSI (14)', color='purple', linewidth=1)
                    ax2_rsi.axhline(70, color='lightgray', linestyle=':', alpha=0.5); ax2_rsi.axhline(30, color='lightgray', linestyle=':', alpha=0.5)
                    ax2_rsi.set_ylabel('RSI', color='purple'); ax2_rsi.set_ylim(0, 100)
                ax2.grid(True, linestyle='--', alpha=0.2)
        fig.tight_layout()
        if canvas: canvas.get_tk_widget().destroy()
        new_canvas = FigureCanvasTkAgg(fig, frame); new_canvas.draw()
        new_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_map = {self.current_plot_frame: 'current_toolbar', self.total_plot_frame: 'total_toolbar', self.cycle_plot_frame: 'cycle_toolbar'}
        if frame in toolbar_map:
            toolbar_attr = toolbar_map[frame]
            if getattr(self, toolbar_attr): getattr(self, toolbar_attr).destroy()
            toolbar = NavigationToolbar2Tk(new_canvas, frame); toolbar.update()
            setattr(self, toolbar_attr, toolbar)
        return new_canvas

    def log(self, msg):
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see(tk.END); self.root.update_idletasks()

    def start_threaded_load(self, worker_func, *args):
        for btn in self.loading_buttons: btn.config(state='disabled')
        thread = threading.Thread(target=worker_func, args=args, daemon=True); thread.start()

    def process_queue(self):
        try:
            result = self.data_queue.get_nowait()
            for btn in self.loading_buttons: btn.config(state='normal')
            if isinstance(result, pd.DataFrame):
                self.df = result; self.df['cmf'] = calculate_cmf(self.df); self.df['rsi'] = calculate_rsi(self.df)
                self.moon_phases = get_moon_phases(self.df); self.update_all()
                self.log(f"Datos procesados: {len(result)} días.")
            elif isinstance(result, Exception):
                self.log(f"Error en la carga de datos: {result}"); messagebox.showerror("Error de Carga", str(result))
        except queue.Empty: pass
        finally: self.root.after(100, self.process_queue)

    def _load_worker(self, fetch_func, name, **kwargs):
        try:
            self.log(f"Descargando datos de {name}..."); df = fetch_func(**kwargs)
            if df is None or df.empty: raise ConnectionError(f"No se pudieron obtener datos de {name}.")
            self.data_queue.put(df)
        except Exception as e: self.data_queue.put(e)

    def load_free_data(self): self.start_threaded_load(self._load_worker, fetch_free_crypto_data, "CoinGecko")
    def load_yahoo(self): self.start_threaded_load(self._load_worker, fetch_yahoo_data, "Yahoo Finance")
    def load_polygon(self): self.start_threaded_load(self._load_worker, fetch_polygon_data, "Polygon", start='2012-01-01')

    def load_custom(self):
        file_path = filedialog.askopenfilename(title="Selecciona tu archivo CSV", filetypes=[("CSV files", "*.csv")])
        if not file_path: self.log("Carga de CSV cancelada."); return
        self.start_threaded_load(self._load_custom_worker, file_path)
    
    def _load_custom_worker(self, file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=[0], index_col=0); df.rename(columns={c: c.lower() for c in df.columns}, inplace=True)
            required = ['open', 'high', 'low', 'close']; missing = set(required) - set(df.columns)
            if missing: raise ValueError(f"Faltan columnas: {missing}")
            if 'volume' not in df.columns: df['volume'] = 0; self.log("Volumen no encontrado, usando 0.")
            self.data_queue.put(df)
        except Exception as e: self.data_queue.put(e)

    def load_kaggle(self):
        if not KAGGLE_AVAILABLE: messagebox.showerror("Error", "Kaggle API no disponible."); return
        json_file = filedialog.askopenfilename(title="Selecciona tu kaggle.json", filetypes=[("JSON files", "*.json")])
        if not json_file: self.log("Carga de Kaggle cancelada."); return
        self.start_threaded_load(self._load_kaggle_worker, json_file)

    def _load_kaggle_worker(self, json_file):
        try:
            kaggle_dir = os.path.expanduser("~/.kaggle"); os.makedirs(kaggle_dir, exist_ok=True)
            shutil.copy(json_file, os.path.join(kaggle_dir, "kaggle.json")); os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
            api = KaggleApi(); api.authenticate()
            self.log("Descargando dataset de Bitcoin (puede tardar)...")
            api.dataset_download_file('mczielinski/bitcoin-historical-data', 'btcusd_1-min_data.csv', path='./')
            df_raw = pd.read_csv('btcusd_1-min_data.csv'); os.remove('btcusd_1-min_data.csv')
            df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], unit='s'); df_raw.set_index('Timestamp', inplace=True)
            
            agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
            rename_dict = {'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
            if 'Volume_(Currency)' in df_raw.columns:
                agg_dict['Volume_(Currency)'] = 'sum'; rename_dict['Volume_(Currency)'] = 'volume'
            elif 'Volume_(BTC)' in df_raw.columns:
                agg_dict['Volume_(BTC)'] = 'sum'; rename_dict['Volume_(BTC)'] = 'volume'
            df = df_raw.resample('D').agg(agg_dict).dropna(); df = df.rename(columns=rename_dict)
            if 'volume' not in df.columns: df['volume'] = 0; self.log("Volumen no encontrado, usando 0.")
            self.data_queue.put(df[df.index >= '2012-01-01'])
        except Exception as e: self.data_queue.put(e)

    def export_current_plot(self): self._export_plot(self.current_fig, "ciclo_actual")
    def export_total_plot(self): self._export_plot(self.total_fig, "total")
    def export_cycle_plot(self): self._export_plot(self.cycle_fig, f"ciclo_{self.cycle_var.get()}")

    def _export_plot(self, fig, name_suffix):
        if fig is None: messagebox.showwarning("Advertencia", "No hay gráfico para exportar."); return
        file_path = filedialog.asksaveasfilename(title=f"Guardar gráfico {name_suffix}", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight'); self.log(f"Gráfico exportado: {file_path}")
            except Exception as e:
                self.log(f"Error exportando: {e}"); messagebox.showerror("Error", f"No se pudo guardar el gráfico: {e}")

    def clear_logs(self):
        self.log_text.delete(1.0, tk.END)
        self.log("Logs limpiados.")

    def update_analysis(self):
        if self.df is None: self.analysis_text.delete(1.0, tk.END); self.analysis_text.insert(1.0, "No hay datos para analizar."); return
        try:
            analysis_text = generate_analysis_text(self.df, self.moon_phases)
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, analysis_text)
        except Exception as e: self.log(f"Error generando análisis: {e}")

    def update_all(self):
        self.log("Actualizando todos los gráficos y análisis...")
        self.update_current_plot(); self.update_total_plot(); self.update_cycle_plot(); self.update_analysis()
        self.notebook.select(0); self.log("¡Actualización completa!")

    def create_github_repo_files(self):
        """Crea archivos básicos para subir a GitHub (.gitignore, README, LICENSE, workflow)."""
        project_root = os.path.dirname(__file__)
        try:
            # .gitignore
            gitignore_path = os.path.join(project_root, ".gitignore")
            with open(gitignore_path, "w", encoding="utf-8") as f:
                f.write(
                    "# Python\n__pycache__/\n*.py[cod]\n*.so\n*.egg-info/\ndist/\nbuild/\n\n# Environments\n.env\n.venv/\nvenv/\n\n# IDEs\n.vscode/\n.idea/\n\n# OS\n.DS_Store\nThumbs.db\n\n# Testing\n.pytest_cache/\n"
                )

            # README.md
            readme_path = os.path.join(project_root, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(
                    "<!-- filepath: c:\\bitcoinmoon\\README.md -->\n# BitcoinMoon\n\nInterfaz para análisis histórico de BTC (CoinGecko / Yahoo / Polygon / Kaggle).\n\nInstrucciones rápidas para subir a GitHub:\n\n```bash\ngit init\ngit add .\ngit commit -m \"Initial commit\"\n# crear repo vacío en GitHub y luego:\ngit remote add origin https://github.com/<TU_USUARIO>/<TU_REPO>.git\ngit branch -M main\ngit push -u origin main\n```\n"
                )

            # LICENSE (MIT)
            license_path = os.path.join(project_root, "LICENSE")
            with open(license_path, "w", encoding="utf-8") as f:
                f.write(
                    "# MIT License\n\nCopyright (c) [YEAR] [COPYRIGHT HOLDER]\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the \"Software\"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\n[...shortened for brevity...]\n"
                )

            # GitHub Actions workflow
            workflows_dir = os.path.join(project_root, ".github", "workflows")
            os.makedirs(workflows_dir, exist_ok=True)
            workflow_path = os.path.join(workflows_dir, "python-app.yml")
            with open(workflow_path, "w", encoding="utf-8") as f:
                f.write(
                    "# filepath: c:\\bitcoinmoon\\.github\\workflows\\python-app.yml\nname: Python CI\n\non:\n  push:\n    branches: [ main ]\n  pull_request:\n    branches: [ main ]\n\njobs:\n  lint:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v3\n      - name: Set up Python\n        uses: actions/setup-python@v4\n        with:\n          python-version: '3.11'\n      - name: Install flake8\n        run: python -m pip install flake8\n      - name: Run flake8 (exit 0)\n        run: flake8 . || true\n"
                )

            self.log("Archivos de GitHub creados (.gitignore, README.md, LICENSE, workflow).")
            self.log("Sigue las instrucciones en README.md para inicializar y subir el repo.")
            if messagebox.askyesno("Abrir crear repo en GitHub", "¿Abrir la página para crear un nuevo repositorio en GitHub?"):
                webbrowser.open("https://github.com/new", new=2)
        except Exception as e:
            self.log(f"Error creando archivos de GitHub: {e}")
            messagebox.showerror("Error", f"No se pudieron crear los archivos: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        print("Iniciando BTC Analyzer...")
        root = tk.Tk(); app = BTCGUI(root)
        app.log("¡Bienvenido! Selecciona una fuente de datos para comenzar.")
        root.mainloop()
    except Exception as e:
        print(f"Error fatal iniciando la aplicacion: {e}"); input("Presiona Enter para salir...")
