import numpy as np
import tkinter as tk
import io
from histogram import open_histogram_window
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from image_processing_gui import ImageProcessingGUI, create_mask
from image_selection import ImageSelection
from config import METHODS, ANOMALY_TYPES, LOGO_CHANGES_PATH,LOGO_CHANGES_TRASP_PATH,LOGO_ISPC_PATH,LOGO_ISPC_TRASP_PATH, PCA_COMPONENTS, ocSVM_KERNELS, ocSVM_DEFAULT_KERNEL, ocSVM_DEFAULT_C, MULTISPECTRAL_BANDS
from tooltip import Tooltip, get_tooltip
from raster_calculator import  open_raster_calculator_via_assignment
from image_processing import find_anomalies
from anomaly_editor import AnomalyEditor



class ImageAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        screen_width = int(self.winfo_screenwidth() * 0.9)
        screen_height = int(self.winfo_screenheight() * 0.9)
        self.geometry(f"{screen_width}x{screen_height}")
        self.title("Smart Anomaly Detection Assistant (S.A.D.A.)")
        self.image_path = None
        self.image = None             # np.array (H,W,C)
        self.wip_image = None
        self.image_histo_processing = None
        self.original_image = None    # np.array (H,W,C)
        self.analyzed_image = None    # np.array (H,W,C)
        self.zoomed_selection_coords = None
        self.zoomed_in = False
        self.effective_scale_factor = 1.0


        self.x_offset = 0  # Inizializza l'offset orizzontale
        self.y_offset = 0  # Inizializza l'offset verticale

        self.history = []
        self.history_index = -1

        self.threshold = tk.IntVar(value=100)
        self.color = tk.StringVar(value="Red")
        self.anomaly_type = tk.StringVar(value="Darker Pixels")
        self.method = tk.StringVar(value="Standard")
        self.band = tk.StringVar(value="Band 1")
        self.geotransform = None
        self.crs = None

        self.logo_image = None
        self.logo_image_trasp = None
        self.second_logo_image = None
        self.second_logo_image_white = None
        self.logo_id = None
        self.second_logo_id = None

        self.image_selection = ImageSelection(self)
        self.image_processing_gui = ImageProcessingGUI(self)

        self.lisa_results = None

        self.pca_mode = tk.StringVar(value="PC1 and PC2")
        self.pca_colormap = tk.StringVar(value="Grayscale")

        self.create_gui()

        self.load_logo()
        self.update_anomaly_type_state()

        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.canvas.bind("<Configure>", self.image_processing_gui.on_main_resize)

    def set_combobox_width(self, combobox, values):
        max_width = max(len(str(value)) for value in values)
        combobox.config(width=max_width)

    def load_logo(self):
        self.logo_image = Image.open(LOGO_CHANGES_PATH)  # Primo logo
        self.logo_image_trasp = Image.open(LOGO_CHANGES_TRASP_PATH)  # Primo logo trasparente
        self.second_logo_image = Image.open(LOGO_ISPC_PATH)  # Secondo logo scuro
        self.second_logo_image_white = Image.open(LOGO_ISPC_TRASP_PATH)  # Secondo logo chiaro (bianco)
        self.after(100, self.resize_logo)
        self.bind("<Configure>", self.on_resize)

    def resize_logo(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width > 0 and canvas_height > 0:
            # Calcoliamo un fattore di dimensione generale
            logo_size = min(canvas_width, canvas_height) // 2
            if logo_size > 0:
                #
                # 1) Ridimensionamento del PRIMO LOGO (logo_image) a dimensione quadrata (logo_size x logo_size)
                #
                self.logo_image_resized = self.logo_image.resize((logo_size, logo_size), Image.LANCZOS)
                self.tk_logo_image = ImageTk.PhotoImage(self.logo_image_resized)

                #
                # 2) Ridimensionamento del SECONDO LOGO (second_logo_image) mantenendo il rapporto d’aspetto
                #
                w2, h2 = self.second_logo_image.size
                aspect2 = w2 / h2  # rapporto larghezza/altezza
                new_h2 = logo_size
                new_w2 = int(new_h2 * aspect2)  # larghezza proporzionata
                self.second_logo_resized = self.second_logo_image.resize((new_w2, new_h2), Image.LANCZOS)
                self.tk_second_logo = ImageTk.PhotoImage(self.second_logo_resized)

                #
                # 3) Cancelliamo eventuali loghi precedenti sul canvas
                #
                if self.logo_id:
                    self.canvas.delete(self.logo_id)
                if self.second_logo_id:
                    self.canvas.delete(self.second_logo_id)

                #
                # 4) Posizioniamo i due loghi affiancati al centro del canvas
                #
                #   - second_logo_id a sinistra
                #   - logo_id a destra
                #
                center_x = canvas_width // 2
                center_y = canvas_height // 2

                # Calcoliamo la “somma” delle larghezze
                total_width = new_w2 + logo_size
                # La X di partenza per il secondo logo (ancorato in alto-sinistra dell’immagine)
                second_x = center_x - total_width // 2
                # La X per il primo logo è second_x + new_w2
                first_x = second_x + new_w2

                # Creiamo le immagini sul canvas (anchor=NW in alto-sinistra, puoi usare anchor=CENTER se preferisci)
                self.second_logo_id = self.canvas.create_image(
                    second_x, center_y - (new_h2 // 2),
                    anchor='nw', image=self.tk_second_logo
                )
                self.logo_id = self.canvas.create_image(
                    first_x, center_y - (logo_size // 2),
                    anchor='nw', image=self.tk_logo_image
                )

    def move_logo_to_top_right(self):
        if self.logo_image_trasp:
            # Mantieni l'aspect ratio impostando l'altezza a 100 e calcolando la larghezza
            original_w, original_h = self.logo_image_trasp.size
            aspect = original_w / original_h
            new_h = 100
            new_w = int(new_h * aspect)

            # Ridimensiona e converti in RGBA per mantenere il canale alpha
            logo_image_trasp_resized = self.logo_image_trasp.resize((new_w, new_h), Image.LANCZOS).convert(
                "RGBA")

            self.tk_logo_image_trasp = ImageTk.PhotoImage(logo_image_trasp_resized)
            self.logo_label_right.config(image=self.tk_logo_image_trasp)
            self.logo_label_right.image = self.tk_logo_image
            self.logo_label_right.grid(row=0, column=2, padx=10, sticky='e')

    def move_logo_to_top_left(self):
        if self.second_logo_image_white:
            # Mantieni l'aspect ratio impostando l'altezza a 100 e calcolando la larghezza
            original_w, original_h = self.second_logo_image_white.size
            aspect = original_w / original_h
            new_h = 100
            new_w = int(new_h * aspect)

            # Ridimensiona e converti in RGBA per mantenere il canale alpha
            second_logo_resized_white = self.second_logo_image_white.resize((new_w, new_h), Image.LANCZOS).convert(
                "RGBA")

            self.tk_second_logo_white = ImageTk.PhotoImage(second_logo_resized_white)

            # Se desideri che il background del label si uniformi a quello del frame, ad esempio:
            # self.logo_label_left.config(bg=self.cget("bg"))

            self.logo_label_left.config(image=self.tk_second_logo_white)
            self.logo_label_left.image = self.tk_second_logo_white
            self.logo_label_left.grid(row=0, column=0, padx=10, sticky='w')
    def hide_center_logo(self):
        if self.logo_id:
            self.canvas.delete(self.logo_id)
            self.logo_id = None

    def on_resize(self, event):
        if self.image is None:
            self.resize_logo()

    def create_gui(self):
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Configura 3 colonne con weight=1 per distribuire lo spazio
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        top_frame.columnconfigure(2, weight=1)

        # Label per secondo logo (sinistro)
        self.logo_label_left = tk.Label(top_frame)
        self.logo_label_left.grid(row=0, column=0, padx=10, sticky='w')

        # Pulsante "Load Image" al centro
        self.select_button = tk.Button(top_frame, text="Load Image", command=self.load_image)
        self.select_button.grid(row=0, column=1, padx=10)

        # Label per primo logo (destro)
        self.logo_label_right = tk.Label(top_frame)
        self.logo_label_right.grid(row=0, column=2, padx=10, sticky='e')

        config_canvas = tk.Canvas(self, height=90, bg=self.cget("bg"), highlightthickness=0, bd=0, relief=tk.FLAT)
        config_scrollbar = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=config_canvas.xview)
        config_frame = tk.Frame(config_canvas, bg=self.cget("bg"), highlightthickness=0, bd=0, relief=tk.FLAT)

        config_frame.bind(
            "<Configure>",
            lambda e: config_canvas.configure(
                scrollregion=config_canvas.bbox("all")
            )
        )
        config_canvas.bind_all("<Shift-MouseWheel>",
                               lambda e: config_canvas.xview_scroll(int(-1 * (e.delta / 120)), "units"))

        config_canvas.create_window((0, 0), window=config_frame, anchor="nw")
        config_canvas.configure(xscrollcommand=config_scrollbar.set)

        config_canvas.pack(side=tk.TOP, fill=tk.X, padx=5)
        config_scrollbar.pack(side=tk.TOP, fill=tk.X)
        self.controls_frame = tk.Frame(self)
        self.controls_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(config_frame, text="Method:").pack(side=tk.LEFT)
        self.method_combobox = ttk.Combobox(config_frame, textvariable=self.method, values=METHODS, state="readonly")
        max_width_methods = max(len(method) for method in METHODS)
        self.method_combobox.config(width=max_width_methods)
        self.method_combobox.set("Standard")
        self.method_combobox.pack(side=tk.LEFT, padx=5)
        self.method_combobox.bind("<<ComboboxSelected>>", self.update_anomaly_type_state)
        Tooltip(self.method_combobox, get_tooltip("method_combobox"))

        self.eps_label = tk.Label(config_frame, text="Eps:")
        self.eps_label.pack_forget()
        self.eps_slider = tk.Scale(config_frame, from_=0.1, to=10, resolution=0.1, orient=tk.HORIZONTAL)
        self.eps_slider.set(0.5)
        self.eps_slider.pack_forget()
        Tooltip(self.eps_slider, get_tooltip("eps_slider"))

        self.min_samples_label = tk.Label(config_frame, text="Min Samples:")
        self.min_samples_label.pack_forget()
        self.min_samples_slider = tk.Scale(config_frame, from_=1, to=20, orient=tk.HORIZONTAL)
        self.min_samples_slider.set(5)
        self.min_samples_slider.pack_forget()
        Tooltip(self.min_samples_slider, get_tooltip("min_samples_slider"))

        self.ocsvm_kernel_label = tk.Label(config_frame, text="ocSVM Kernel:")
        self.ocsvm_kernel_combobox = ttk.Combobox(config_frame, values=ocSVM_KERNELS, state="readonly")
        max_width = max(len(kernel) for kernel in ocSVM_KERNELS)
        self.ocsvm_kernel_combobox.config(width=max_width)
        self.ocsvm_kernel_combobox.set(ocSVM_DEFAULT_KERNEL)

        self.ocsvm_c_label = tk.Label(config_frame, text="ocSVM C:")
        self.ocsvm_c_slider = tk.Scale(config_frame, from_=0.1, to=10.0, orient=tk.HORIZONTAL, resolution=0.1)
        self.ocsvm_c_slider.set(ocSVM_DEFAULT_C)

        self.ocsvm_kernel_label.pack_forget()
        self.ocsvm_kernel_combobox.pack_forget()
        self.ocsvm_c_label.pack_forget()
        self.ocsvm_c_slider.pack_forget()
        Tooltip(self.ocsvm_kernel_combobox, get_tooltip("ocSVM_kernel"))
        Tooltip(self.ocsvm_c_slider, get_tooltip("ocSVM_c_slider"))

        tk.Label(config_frame, text="Threshold:").pack(side=tk.LEFT)
        self.threshold_slider = tk.Scale(config_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.threshold,
                                         state=tk.DISABLED, fg="gray", troughcolor="gray")
        self.threshold_slider.pack(side=tk.LEFT, padx=5)
        Tooltip(self.threshold_slider, get_tooltip("threshold_slider"))

        tk.Label(config_frame, text="Anomaly Type:").pack(side=tk.LEFT)
        self.anomaly_combobox = ttk.Combobox(config_frame, textvariable=self.anomaly_type, values=ANOMALY_TYPES,
                                             state="readonly")
        Tooltip(self.anomaly_combobox, get_tooltip("anomaly_combobox"))
        self.anomaly_combobox.set("Darker Pixels")
        self.set_combobox_width(self.anomaly_combobox, ANOMALY_TYPES)
        self.anomaly_combobox.pack(side=tk.LEFT, padx=5)

        color_options = ["Red", "White", "Blue", "Green", "Black"]
        self.color_combobox = ttk.Combobox(config_frame, textvariable=self.color, values=color_options,
                                           state="readonly")
        self.color_combobox.set("Red")
        self.set_combobox_width(self.color_combobox, color_options)
        self.color_combobox.pack(side=tk.LEFT, padx=5)
        Tooltip(self.color_combobox, get_tooltip("color_combobox"))

        tk.Label(config_frame, text="Brightness:").pack(side=tk.LEFT)
        self.brightness_slider = tk.Scale(config_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                                          command=self.adjust_brightness, state=tk.DISABLED, fg="gray",
                                          troughcolor="gray")
        self.brightness_slider.set(0)
        self.brightness_slider.pack(side=tk.LEFT, padx=5)
        Tooltip(self.brightness_slider, get_tooltip("brightness_slider"))

        tk.Label(config_frame, text="Contrast:").pack(side=tk.LEFT)
        self.contrast_slider = tk.Scale(config_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
                                        command=self.adjust_contrast, state=tk.DISABLED, fg="gray", troughcolor="gray")
        self.contrast_slider.set(0)
        self.contrast_slider.pack(side=tk.LEFT, padx=5)
        Tooltip(self.contrast_slider, get_tooltip("contrast_slider"))
        button_canvas = tk.Canvas(self, height=50)
        button_scrollbar = tk.Scrollbar(self, orient=tk.HORIZONTAL, command=button_canvas.xview)
        button_frame = tk.Frame(self,bg=self.cget("bg"), highlightthickness=0, bd=0, relief=tk.FLAT)

        button_frame.bind(
            "<Configure>",
            lambda e: button_canvas.configure(
                scrollregion=button_canvas.bbox("all")
            )
        )
        button_canvas.bind_all("<Shift-MouseWheel>",
                               lambda e: button_canvas.xview_scroll(int(-1 * (e.delta / 120)), "units"))

        button_canvas.create_window((0, 0), window=button_frame, anchor="nw")
        button_canvas.configure(xscrollcommand=button_scrollbar.set)

        button_canvas.pack(side=tk.BOTTOM, fill=tk.X, padx=5)
        button_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.select_area_button = tk.Button(button_frame, text="Select by points", command=self.enable_selection,
                                            state=tk.DISABLED)
        self.select_area_button.pack(side=tk.LEFT, padx=5)
        # Tooltip for "Select by Points"
        Tooltip(self.select_area_button, "Draw a polygon to select an area of interest.")

        self.standard_selection_button = tk.Button(button_frame, text="Standard Selection",
                                                   command=self.enable_standard_selection, state=tk.DISABLED)
        self.standard_selection_button.pack(side=tk.LEFT, padx=5)
        # Tooltip for "Standard Selection"
        Tooltip(self.standard_selection_button, "Draw a rectangle to select an area of interest.")

        self.raster_calc_button = tk.Button(button_frame, text="Raster Calculator",
                                            command=lambda: open_raster_calculator_via_assignment(self,
                                                                                                  self.update_image_from_raster_calc),
                                            state=tk.DISABLED)
        self.raster_calc_button.pack(side=tk.LEFT, padx=5)
        # Tooltip for "Raster Calculator"
        Tooltip(self.raster_calc_button, "Open the raster calculator to compute indices or custom expressions.")

        self.analyze_button = tk.Button(button_frame, text="Analyze Image", command=self.analyze_image,
                                        state=tk.DISABLED)
        self.analyze_button.pack(side=tk.LEFT, padx=5)
        # Tooltip for "Analyze Image"
        Tooltip(self.analyze_button, "Run anomaly detection using the selected method.")

        self.anomaly_editor_button = tk.Button(button_frame, text="Anomaly Editor",
                                               command=self.open_anomaly_editor,
                                               state=tk.DISABLED)
        self.anomaly_editor_button.pack(side=tk.LEFT, padx=5)
        # Tooltip for "Anomaly Editor"
        Tooltip(self.anomaly_editor_button, "Open the editor to refine or correct detected anomalies.")

        self.pca_viewer_button = tk.Button(button_frame, text="PCA Viewer",
                                           command=self.image_processing_gui.open_pca_viewer, state=tk.DISABLED)
        self.pca_viewer_button.pack(side=tk.LEFT, padx=5)
        # Tooltip for "PCA Editor"
        Tooltip(self.pca_viewer_button, "View and manage the results of Principal Component Analysis.")

        self.save_button = tk.Button(button_frame, text="Export Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.save_wip_button = tk.Button(button_frame, text="Save WIP", command=self.save_wip, state=tk.DISABLED)
        self.save_wip_button.pack(side=tk.LEFT, padx=2, pady=2)
        Tooltip(self.save_wip_button, "Save current image state as work-in-progress")

        self.save_plot_button = tk.Button(button_frame, text="Export Plot", command=self.save_plot, state=tk.DISABLED)
        self.save_plot_button.pack(side=tk.LEFT, padx=5)
        Tooltip(self.save_plot_button, get_tooltip("save_plot_button"))

        self.undo_button = tk.Button(button_frame, text="Undo", command=self.undo, state=tk.DISABLED)
        self.undo_button.pack(side=tk.LEFT, padx=5)

        self.redo_button = tk.Button(button_frame, text="Redo", command=self.redo, state=tk.DISABLED)
        self.redo_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(button_frame, text="Reset", command=self.go_home, state=tk.DISABLED)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        Tooltip(self.reset_button, "Reset the interface. If a Work In Progress image exists, you can choose to restore it instead of the original image.")

        self.zoom_in_button = tk.Button(button_frame, text="Zoom In", command=self.zoom_in, state=tk.DISABLED)
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)

        self.zoom_out_button = tk.Button(button_frame, text="Zoom Out", command=self.zoom_out, state=tk.DISABLED)
        self.zoom_out_button.pack(side=tk.LEFT, padx=5)

        self.zoom_to_selection_button = tk.Button(button_frame, text="Zoom to Selection",
                                                  command=self.zoom_to_selection,
                                                  state=tk.DISABLED)
        self.zoom_to_selection_button.pack(side=tk.LEFT, padx=5)

        self.pan_button = tk.Button(button_frame, text="Move", command=self.start_pan, state=tk.DISABLED)
        self.pan_button.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self, width=800, height=600, bg="white")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        self.canvas.bind("<Double-Button-1>", self.on_button_double_click)

        self.status_bar = tk.Label(self, text="Status: Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.load_logo()
        self.update_anomaly_type_state()

        tk.Label(config_frame, text="Select Band:").pack(side=tk.LEFT)
        self.band_combobox = ttk.Combobox(config_frame, textvariable=self.band, values=MULTISPECTRAL_BANDS,
                                          state="readonly")
        self.band_combobox.set("Band 1")
        self.band_combobox.pack(side=tk.LEFT, padx=5)
        self.band_combobox.bind("<<ComboboxSelected>>", self.update_band_display)
        self.band_combobox.config(state="disabled")

        self.pca_selector_label = tk.Label(config_frame, text="PCA Component:")
        self.pca_selector_label.pack(side=tk.LEFT)
        self.pca_selector_label.pack_forget()

        self.pca_component_selector = ttk.Combobox(config_frame, state="disabled", width=10)
        self.pca_component_selector.bind("<<ComboboxSelected>>", self.image_processing_gui.update_pca_highlight)
        self.pca_component_selector.pack(side=tk.LEFT, padx=5)
        self.pca_component_selector.pack_forget()

        self.lisa_index_label = tk.Label(config_frame, text="Select LISA index:")
        self.lisa_index_combobox = ttk.Combobox(
            config_frame,
            values=["Moran's I", "Getis-Ord", "INDIC (Moran n Getis)"],
            state="disabled"  # disabilitato all’inizio
        )
        self.lisa_index_combobox.set("Moran's I")  # default stringa
        Tooltip(self.lisa_index_combobox, get_tooltip("lisa_index_combobox"))
        self.lisa_index_combobox.bind("<<ComboboxSelected>>", self.update_lisa_display)

        self.histogram_button = tk.Button(
            button_frame,
            text="Histogram",
            command=lambda: setattr(
                self,
                'hist_win',
                open_histogram_window(
                    self,
                    self.image,
                    self.original_image_pil,
                    self.update_image_from_histogram,
                    selection_mask=self.image_selection.get_selection_mask(self.original_image_np.shape[:2])
                )
            ),
            state=tk.DISABLED
        )
        self.histogram_button.pack(side=tk.LEFT, padx=5)
        # Tooltip for "Histogram"
        Tooltip(self.histogram_button, "View and modify histogram of the current image or selected area.")

        self.histo_pixel_button = tk.Button(button_frame, text="Histo by Pixel",
                                            command=self.activate_histo_pixel_mode, state=tk.DISABLED)
        self.histo_pixel_button.pack(side=tk.LEFT, padx=5)
        # Tooltip for "Histo by Pixel"
        Tooltip(self.histo_pixel_button, "View spectral profile of a selected pixel.")

        self.select_band_combobox = self.band_combobox
        self.select_band_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_band_display())


    def update_band_display(self, event=None):
        if self.original_image_np is None:
            return

        # Usa sempre lo STACK COMPLETO se disponibile, altrimenti fallback a original_image_np
        full_stack = self.multispectral_stack if hasattr(self, 'multispectral_stack') and isinstance(
            self.multispectral_stack, np.ndarray) else self.original_image_np

        # Mostra tutta l'immagine salvo zoom selezione
        if self.zoomed_in and self.zoomed_selection_coords is not None:
            left, top, right, bottom = self.zoomed_selection_coords
            img_h, img_w = full_stack.shape[:2]
            left = max(int(left), 0)
            top = max(int(top), 0)
            right = min(int(right), img_w)
            bottom = min(int(bottom), img_h)
            image_array = full_stack[top:bottom, left:right, :]
        else:
            image_array = full_stack
            self.zoomed_in = False
            self.zoomed_selection_coords = None

        selected_band_name = self.band.get()
        import re

        # Determina il numero di canali
        if image_array.ndim < 3:
            n_channels = 1
        else:
            n_channels = image_array.shape[2]

        if n_channels >= 5:
            # Per immagini multispettrali a 5+ canali:
            if selected_band_name == "RGB":
                # Applica stretching 2‑98% su false RGB
                pseudo_rgb = image_array[..., [2, 1, 0]].astype(np.float32)
                pseudo_rgb[pseudo_rgb < -9999] = np.nan
                low = np.nanpercentile(pseudo_rgb, 2)
                high = np.nanpercentile(pseudo_rgb, 98)
                print(f"[DEBUG] update_band_display (RGB stretch): 2-98 percentile range: {low} .. {high}")
                if high - low < 1e-6:
                    arr_f = np.zeros_like(pseudo_rgb)
                else:
                    arr_f = (pseudo_rgb - low) / (high - low) * 255.0
                arr_f = np.nan_to_num(arr_f, nan=0)
                single_band = np.clip(arr_f, 0, 255).astype('uint8')
            else:
                # Per una singola banda: estraiamo la banda desiderata
                if hasattr(self, 'logic_to_index') and self.logic_to_index:
                    if selected_band_name in self.logic_to_index:
                        idx = self.logic_to_index[selected_band_name]
                        single_band = image_array[..., idx]
                    else:
                        match = re.search(r"Band\s+(\d+)", selected_band_name)
                        if match:
                            band_index = int(match.group(1)) - 1
                            single_band = image_array[..., band_index]
                        else:
                            single_band = image_array[..., 0]
                else:
                    match = re.search(r"Band\s+(\d+)", selected_band_name)
                    if match:
                        band_index = int(match.group(1)) - 1
                        single_band = image_array[..., band_index]
                    else:
                        single_band = image_array[..., 0]
                if single_band.ndim == 3:
                    single_band = single_band[..., 0]
                if not np.issubdtype(single_band.dtype, np.floating):
                    single_band = single_band.astype(np.float32)
                single_band[single_band < -9999] = np.nan
                low = np.nanpercentile(single_band, 2)
                high = np.nanpercentile(single_band, 98)
                print(f"[DEBUG] update_band_display (single band): 2-98 percentile range: {low} .. {high}")
                if high - low < 1e-6:
                    arr_f = np.zeros_like(single_band)
                else:
                    arr_f = (single_band - low) / (high - low) * 255.0
                arr_f = np.nan_to_num(arr_f, nan=0)
                single_band = np.clip(arr_f, 0, 255).astype('uint8')
                single_band = single_band[..., np.newaxis]

        else:
            # Per immagini con meno di 5 canali (1, 3 o 4)
            if selected_band_name == "RGB":
                single_band = image_array[..., :3]
            else:
                match = re.search(r"Band\s+(\d+)", selected_band_name)
                if match:
                    band_index = int(match.group(1)) - 1
                    if image_array.ndim == 3 and band_index < image_array.shape[2]:
                        single_band = image_array[..., band_index]
                    else:
                        single_band = image_array[..., 0] if image_array.ndim == 3 else image_array
                else:
                    single_band = image_array[..., 0] if image_array.ndim == 3 else image_array

            # Applica stretch lineare
            if single_band.ndim == 2:
                band_min, band_max = single_band.min(), single_band.max()
                if band_max - band_min > 0:
                    single_band = (single_band - band_min) / (band_max - band_min) * 255
                else:
                    single_band = np.zeros_like(single_band)
                single_band = single_band.astype('uint8')[..., np.newaxis]
            elif single_band.ndim == 3:
                band_min, band_max = single_band.min(), single_band.max()
                if band_max - band_min > 0:
                    single_band = (single_band - band_min) / (band_max - band_min) * 255
                else:
                    single_band = np.zeros_like(single_band)
                single_band = single_band.astype('uint8')

        self.image = single_band
        self.analyzed_image = None
        self.image_processing_gui.display_image_on_canvas(single_band)
        self.update_status(f"Displaying {selected_band_name}")

    def update_status(self, message: object) -> object:
        self.status_bar.config(text=f"Status: {message}")

    def update_anomaly_type_state(self, *args):
        self.eps_label.pack_forget()
        self.eps_slider.pack_forget()
        self.min_samples_label.pack_forget()
        self.min_samples_slider.pack_forget()
        self.ocsvm_kernel_label.pack_forget()
        self.ocsvm_kernel_combobox.pack_forget()
        self.ocsvm_c_label.pack_forget()
        self.ocsvm_c_slider.pack_forget()

        if self.method.get() == "Standard":
            self.anomaly_combobox.config(state="readonly")
        elif self.method.get() == "PCA":
            self.anomaly_combobox.config(state="readonly")
        elif self.method.get() == "K-means":
            self.anomaly_combobox.config(state="disabled")
        elif self.method.get() == "DBSCAN":
            self.anomaly_combobox.config(state="disabled")
            self.eps_label.pack(side=tk.LEFT)
            self.eps_slider.pack(side=tk.LEFT, padx=5)
            self.min_samples_label.pack(side=tk.LEFT)
            self.min_samples_slider.pack(side=tk.LEFT, padx=5)
        elif self.method.get() == "ocSVM":
            self.anomaly_combobox.config(state="disabled")
            self.ocsvm_kernel_label.pack(side=tk.LEFT)
            self.ocsvm_kernel_combobox.pack(side=tk.LEFT, padx=5)
            self.ocsvm_c_label.pack(side=tk.LEFT)
            self.ocsvm_c_slider.pack(side=tk.LEFT, padx=5)

    def load_image(self):
        # Reset della variabile wip_image PRIMA di qualsiasi operazione
        self.wip_image = None
        self.image_selection.load_image()
        if self.image_selection.image is not None:
            # Reset totali per evitare che la vecchia immagine rimanga
            self.history.clear()
            self.history_index = -1
            self.analyzed_image = None
            self.band_map = {}  # ⚠️ Reset della band assignment per Raster Calculator

            # Reset manuale dei riferimenti (NON azzeriamo original_image_np: la preview 3‑canali potrebbe già essere stata impostata da ImageSelection)
            # Manteniamo self.original_image_np se già presente; azzeriamo solo la PIL
            self.original_image_pil = None

            # Imposta la nuova immagine
            new_arr = self.image_selection.image

            if isinstance(new_arr, np.ndarray):
                print(
                    f"[DEBUG] load_image: new_arr shape={new_arr.shape}, dtype={new_arr.dtype}, min={new_arr.min()}, max={new_arr.max()}")
            else:
                print(f"[DEBUG] load_image: new_arr is a PIL.Image, size={new_arr.size}, mode={new_arr.mode}")

            self.image_histo_processing = None

            # Non sovrascrivere la preview 3‑canali se il dato è multispettrale (>3 bande)
            if isinstance(new_arr, np.ndarray) and new_arr.ndim == 3 and new_arr.shape[2] > 3:
                pass
            else:
                self.original_image_np = np.copy(new_arr)

            # Determina l'immagine da visualizzare
            if isinstance(new_arr, np.ndarray) and new_arr.ndim == 3:
                if new_arr.shape[2] >= 5:
                    # Stretch percentuale solo per multispettrali (usa preview RGB 3‑2‑1)
                    pseudo_rgb = self.original_image_np[..., [2, 1, 0]].astype(np.float32)
                    pseudo_rgb[pseudo_rgb < -9999] = np.nan
                    low = np.nanpercentile(pseudo_rgb, 2)
                    high = np.nanpercentile(pseudo_rgb, 98)
                    if high - low < 1e-6:
                        arr_f = np.zeros_like(pseudo_rgb)
                    else:
                        arr_f = (pseudo_rgb - low) / (high - low) * 255.0
                    arr_f = np.nan_to_num(arr_f, nan=0)
                    stretched = np.clip(arr_f, 0, 255).astype('uint8')
                    self.image = stretched
                    self.band.set("RGB")
                elif new_arr.shape[2] >= 3:
                    # 3 o 4 bande: visualizza sempre come RGB a 3 canali
                    if hasattr(self, 'original_image_np') and isinstance(self.original_image_np, np.ndarray) \
                            and self.original_image_np.ndim == 3 and self.original_image_np.shape[2] == 3:
                        self.image = np.copy(self.original_image_np)
                    else:
                        self.image = new_arr[:, :, :3].astype('uint8')
                    self.band.set("RGB")
                else:
                    self.image = np.copy(new_arr)
            else:
                # PIL o array 2D
                self.image = np.copy(new_arr) if isinstance(new_arr, np.ndarray) else np.array(new_arr)

            # Prepara la PIL corretta (evita sempre RGBA per la visualizzazione di base)
            if isinstance(self.image_selection.image, Image.Image):
                pil_img = self.image_selection.image
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                self.original_image_pil = pil_img
                # Mantieni la preview 3‑canali già presente; aggiorna solo se mancante
                if not (hasattr(self, 'original_image_np') and isinstance(self.original_image_np, np.ndarray)):
                    self.original_image_np = np.array(pil_img)
            else:
                arr = self.image_selection.image
                # Se abbiamo già una preview 3 canali, usiamola per costruire la PIL
                if hasattr(self, 'original_image_np') and isinstance(self.original_image_np, np.ndarray):
                    src = self.original_image_np
                else:
                    src = arr
                    self.original_image_np = arr

                if src.ndim == 2:
                    self.original_image_pil = Image.fromarray(src.astype('uint8'), 'L').convert('RGB')
                elif src.ndim == 3:
                    c = src.shape[2]
                    if c == 1:
                        self.original_image_pil = Image.fromarray(src[:, :, 0].astype('uint8'), 'L').convert('RGB')
                    else:
                        # Forza sempre RGB usando le prime 3 bande per la sola visualizzazione
                        self.original_image_pil = Image.fromarray(src[:, :, :3].astype('uint8'), 'RGB')
                else:
                    self.original_image_pil = Image.fromarray(src.astype('uint8')).convert('RGB')

            # --- ATTENZIONE QUI ---
            self.go_home(reset_image=False)

            # NON sovrascrivere di nuovo self.image!
            # self.image_processing_gui.display_image_on_canvas(self.image)  # <-- RIMOSSO per evitare doppio rendering

            self.brightness_slider.config(state=tk.NORMAL)
            self.brightness_slider.set(0)
            self.contrast_slider.config(state=tk.NORMAL)
            self.contrast_slider.set(0)
            self.histogram_button.config(state=tk.NORMAL)
            self.raster_calc_button.config(state=tk.NORMAL)

            if self.crs is not None and self.geotransform is not None:
                self.save_plot_button.config(state=tk.NORMAL)
            else:
                self.save_plot_button.config(state=tk.DISABLED)

            self.histo_pixel_button.config(state=tk.NORMAL)

            self.move_logo_to_top_right()
            self.move_logo_to_top_left()
            self.pca_viewer_button.config(state=tk.DISABLED)


    def enable_selection(self):
        self.image_selection.enable_selection()
        self.pan_button.config(state=tk.DISABLED)


    def enable_standard_selection(self):
        self.image_selection.enable_standard_selection()
        self.pan_button.config(state=tk.DISABLED)


    def on_button_press(self, event):
        self.image_selection.on_button_press(event)


    def on_move_press(self, event):
        self.image_selection.on_move_press(event)
        # Prima unbind di selezione
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        # Poi start pan
        self.image_processing_gui.start_pan()


    def on_button_release(self, event):
        self.image_selection.on_button_release(event)


    def on_button_double_click(self, event):
        self.image_selection.on_button_double_click(event)


    def analyze_image(self):
        self.image_processing_gui.analyze_image()
        self.update_status("Image analyzed")
        if self.image is not None:
            self.analyzed_image = np.copy(self.image)

    def save_image(self):
        self.image_processing_gui.save_image()

    def save_wip(self):
        if self.image is not None:
            self.wip_image = np.copy(self.image)
            self.update_status("Work-in-progress saved.")

    def undo(self):
        self.image_processing_gui.undo()

    def redo(self):
        self.image_processing_gui.redo()

    def go_home(self, reset_image=True):
        self.x_offset = 0  # Reimposta l'offset orizzontale
        self.y_offset = 0  # Reimposta l'offset verticale
        # --- Elimina qualsiasi selezione attiva (poligono o rettangolo) ---
        self.image_selection.reset_selection()

        if hasattr(self, "wip_image") and self.wip_image is not None:
            self.open_reset_choice_window()
            return

        if hasattr(self, 'hist_win') and self.hist_win is not None:
            self.hist_win.destroy()
            self.hist_win = None

        if reset_image:
            if self.original_image_np.ndim == 3 and self.original_image_np.shape[2] >= 5:
                try:
                    pseudo_rgb = self.original_image_np[..., [2, 1, 0]].astype(np.float32)
                except Exception:
                    pseudo_rgb = self.original_image_np[..., :3].astype(np.float32)

                pseudo_rgb[pseudo_rgb < -9999] = np.nan
                low = np.nanpercentile(pseudo_rgb, 2)
                high = np.nanpercentile(pseudo_rgb, 98)
                if high - low < 1e-6:
                    arr_f = np.zeros_like(pseudo_rgb)
                else:
                    arr_f = (pseudo_rgb - low) / (high - low) * 255.0
                arr_f = np.nan_to_num(arr_f, nan=0)
                stretched = np.clip(arr_f, 0, 255).astype('uint8')

                self.image = stretched
                self.original_image_pil = Image.fromarray(stretched, 'RGB')
                self.band.set("RGB")
            else:
                self.image = np.copy(self.original_image_np)
                # Forza sempre visualizzazione RGB se possibile
                if self.image.ndim == 3 and self.image.shape[2] >= 3:
                    self.image = self.image[..., :3]
                    self.original_image_pil = Image.fromarray(self.image.astype('uint8'), 'RGB')
                    self.band.set("RGB")
                elif self.image.ndim == 2 or self.image.shape[2] == 1:
                    self.original_image_pil = Image.fromarray(self.image[..., 0].astype('uint8'), 'L')
                    self.band.set("Band 1")
                else:
                    self.original_image_pil = None
        else:
            self.image = np.copy(self.image)

        self.image_histo_processing = None
        self.image_processing_gui.go_home(reset_image=False)

        self.zoom_to_selection_button.config(state=tk.DISABLED)
        self.pan_button.config(state=tk.DISABLED)
        self.select_area_button.config(state=tk.NORMAL)
        self.brightness_slider.set(0)
        self.contrast_slider.set(0)
        self.zoomed_in = False
        self.update_idletasks()

        self.method.set("Standard")
        self.update_anomaly_type_state()
        self.update_status("Original image")

        self.threshold_slider.config(state=tk.DISABLED, fg="gray", troughcolor="gray")
        self.threshold.set(100)
        self.anomaly_editor_button.config(state=tk.DISABLED)

        self.lisa_index_label.pack_forget()
        self.lisa_index_combobox.pack_forget()
        self.lisa_index_combobox.config(state="disabled")
        self.lisa_results = None

        self.pca_viewer_button.config(state=tk.DISABLED)

    def open_reset_choice_window(self):
        win = tk.Toplevel(self)
        win.title("Reset Image")

        tk.Label(win, text="Select reset mode:").pack(side=tk.TOP, padx=10, pady=10)

        choice = tk.StringVar(value="original")

        tk.Radiobutton(win, text="Original Image", variable=choice, value="original").pack(anchor=tk.W, padx=15)
        tk.Radiobutton(win, text="Work In Progress (WIP)", variable=choice, value="wip").pack(anchor=tk.W, padx=15)

        def on_choice():
            import numpy as np
            if choice.get() == "wip":
                # Ripristina l’immagine WIP e imposta la baseline per brightness/contrast
                self.image = self.wip_image.copy()
                self.image_histo_processing = np.copy(self.image)
            else:
                # Ritorna all’immagine originale
                self.image = self.original_image_np.copy()
                # Azzeri eventuali trasformazioni di istogramma precedenti
                self.image_histo_processing = None
            self.brightness_slider.set(0)
            self.contrast_slider.set(0)
            # Forza un ridisegno passando una copia (nuovo id di array) per aggirare l'early‑return “Skipping redraw”
            import numpy as np
            self.image_processing_gui.display_image_on_canvas(np.copy(self.image))
            self.update_status("Image reset")
            # Se l'utente ha scelto di tornare all'immagine originale, la WIP non serve più.
            # Rimuovendola evitiamo di riaprire la finestra di scelta a ogni successivo reset
            # e, soprattutto, consentiamo a go_home di completare il reset standard in un solo passaggio.
            if choice.get() == "original":
                self.wip_image = None
                # ri‑utilizzando la logica già presente in go_home senza ripresentare la finestra.
                self.after(10, lambda: self.go_home(reset_image=False))
            # Azzera completamente la selezione attiva
            self.image_selection.reset_selection()
            # Chiudi eventuali finestre Histogram aperte per evitare che conservino la vecchia mask
            for child in self.winfo_children():
                if isinstance(child, tk.Toplevel) and child.title() in ("Histogram", "Pixel Histogram"):
                    try:
                        child.destroy()
                    except Exception:
                        pass
            self.zoom_to_selection_button.config(state=tk.DISABLED)
            self.pan_button.config(state=tk.DISABLED)
            win.destroy()

        ok_btn = tk.Button(win, text="OK", command=on_choice)
        ok_btn.pack(side=tk.BOTTOM, pady=10)

    def zoom_in(self):
        self.image_processing_gui.zoom_in()
        self.pan_button.config(state=tk.NORMAL)

    def zoom_out(self):
        self.image_processing_gui.zoom_out()
        self.pan_button.config(state=tk.NORMAL)

    def zoom_to_selection(self):
        self.image_processing_gui.zoom_to_selection()
        self.zoomed_in = True
        self.reset_button.config(state=tk.NORMAL)

    def start_pan(self):
        self.image_processing_gui.start_pan()

    def adjust_brightness(self, value):
        self.update_brightness_contrast()

    def adjust_contrast(self, value):
        self.update_brightness_contrast()

    def update_brightness_contrast(self):
        if self.original_image_np is None:
            return

        # Usa sempre l'immagine processata dall'istogramma, se presente
        if hasattr(self, "image_histo_processing") and self.image_histo_processing is not None:
            base = self.image_histo_processing.astype(np.float32)
        elif self.original_image_np.ndim == 3 and self.original_image_np.shape[2] >= 5:
            base = self.image.astype(np.float32)
        else:
            base = self.original_image_np.astype(np.float32)

        # Se è selezionata una singola banda (non "RGB"), estrai il canale corrispondente.
        selected_band = self.band.get()
        if selected_band != "RGB":
            import re
            match = re.search(r"Band\s*(\d+)", selected_band)
            if match:
                ch = int(match.group(1)) - 1
            else:
                ch = 0
            if base.ndim == 3 and base.shape[2] > ch:
                base = base[..., ch]
            else:
                if base.ndim == 3:
                    base = base[..., 0]
            if base.ndim == 2:
                base = base[..., np.newaxis]

        # Stretch base tra 0–255
        band_min, band_max = base.min(), base.max()
        if band_max - band_min > 0:
            base = (base - band_min) / (band_max - band_min) * 255
        else:
            base = np.zeros_like(base)
        base = base.astype('uint8')

        # Salva la versione già stretchata in self.image (che fungerà da "base" per le trasformazioni successive)
        self.image = base

        # Applica la trasformazione di luminosità
        b = float(self.brightness_slider.get())
        if b < 0:
            bright = base * (1 + b / 100.0)
        elif b > 0:
            bright = base + (255 - base) * (b / 100.0)
        else:
            bright = base

        # Applica la trasformazione di contrasto
        c = float(self.contrast_slider.get())
        mean = bright.mean()
        if c < 0:
            final = mean + (bright - mean) * (1 + c / 100.0)
        elif c > 0:
            final = mean + (bright - mean) * (1 + c / 75.0)
        else:
            final = bright

        final = np.clip(final, 0, 255).astype(np.uint8)

        # Se esiste una selezione attiva (controllo anche per polygon_points e rect_coords)
        if ((self.zoomed_in and self.zoomed_selection_coords is not None) or
                (hasattr(self.image_selection, 'polygon_points') and self.image_selection.polygon_points and len(
                    self.image_selection.polygon_points) > 0) or
                (self.image_selection.rect_coords is not None)):
            self.apply_partial_update(final, "Brightness/Contrast adjusted")
        else:
            self.image = final
            self.image_processing_gui.display_image_on_canvas(self.image)
            self.save_wip_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.update_status("Brightness/Contrast adjusted")

    def apply_partial_update(self, enhanced_image, status_message):
        if self.image is None:
            self.image = np.copy(enhanced_image)
            self.image_processing_gui.display_image_on_canvas(self.image)
            self.reset_button.config(state=tk.NORMAL)
            self.update_status(status_message)
            return

        # Per immagini multispettrali (5+ canali) usiamo le dimensioni della versione attualmente visualizzata;
        # altrimenti usiamo quelle dell'immagine originale.
        if self.original_image_np.ndim == 3 and self.original_image_np.shape[2] >= 5:
            img_h, img_w = self.image.shape[:2]
        else:
            img_h, img_w = self.original_image_np.shape[:2]

        if self.zoomed_in and self.zoomed_selection_coords is not None:
            left, top, right, bottom = self.zoomed_selection_coords
            left = max(int(left), 0)
            right = min(int(right), img_w)
            top = max(int(top), 0)
            bottom = min(int(bottom), img_h)
            cropped_image = enhanced_image[top:bottom, left:right]
            if cropped_image.ndim == 2:
                cropped_image = cropped_image[..., np.newaxis]
            # Ridimensiona cropped_image alle dimensioni attuali di self.image
            cur_h, cur_w = self.image.shape[:2]
            pil_cropped = Image.fromarray(cropped_image)
            pil_zoomed = pil_cropped.resize((cur_w, cur_h), Image.LANCZOS)
            zoomed_image = np.array(pil_zoomed)
            if zoomed_image.ndim == 2:
                zoomed_image = zoomed_image[..., np.newaxis]
            self.image = zoomed_image

        elif self.image_selection.polygon_points:
            x_coords, y_coords = zip(*self.image_selection.polygon_points)
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            min_x = max(int(min_x - self.x_offset), 0)
            max_x = min(int(max_x - self.x_offset), img_w)
            min_y = max(int(min_y - self.y_offset), 0)
            max_y = min(int(max_y - self.y_offset), img_h)
            cropped_image = enhanced_image[min_y:max_y, min_x:max_x]
            self.image[min_y:max_y, min_x:max_x] = cropped_image

        elif self.image_selection.rect_coords:
            left, top, right, bottom = self.image_selection.rect_coords
            left = max(int(left), 0)
            right = min(int(right), img_w)
            top = max(int(top), 0)
            bottom = min(int(bottom), img_h)
            cropped_image = enhanced_image[top:bottom, left:right]
            self.image[top:bottom, left:right] = cropped_image
        else:
            self.image = enhanced_image

        self.image_processing_gui.display_image_on_canvas(self.image)
        self.reset_button.config(state=tk.NORMAL)
        self.update_status(status_message)

    def update_image_from_histogram(self, new_image):
        self.image = new_image
        self.image_histo_processing = np.copy(new_image)
        self.image_processing_gui.display_image_on_canvas(self.image)
        self.save_wip_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
        self.update_status("Image updated from histogram transformation")

    def update_image_from_raster_calc(self, new_image, band_name=None):
        # ------------------------------------------------------------------
        # GUARD CLAUSE: gestione rimozione banda di anteprima (__preview__)
        # ------------------------------------------------------------------
        if new_image is None:
            # Se la band da rimuovere è nota ed esiste nelle strutture interne
            if band_name and hasattr(self, "logic_to_index") and band_name in self.logic_to_index:
                idx_to_delete = self.logic_to_index.pop(band_name)

                # 1) Rimuovi la banda dall'array originale
                if (hasattr(self, "original_image_np") and self.original_image_np is not None
                        and self.original_image_np.ndim == 3
                        and self.original_image_np.shape[2] > idx_to_delete):
                    self.original_image_np = np.delete(self.original_image_np, idx_to_delete, axis=2)

                # 2) Aggiorna gli indici rimanenti
                for key in list(self.logic_to_index.keys()):
                    if self.logic_to_index[key] > idx_to_delete:
                        self.logic_to_index[key] -= 1

                # 3) Elimina eventuale mapping band_map → logical name
                if hasattr(self, "band_map") and self.band_map:
                    for k in list(self.band_map.keys()):
                        if self.band_map[k] == band_name:
                            del self.band_map[k]

                # 4) Aggiorna la combobox delle bande
                self.update_band_combobox(None)
                self.update_band_display()
                self.update_status("Preview band removed.")
            # Esci comunque: nessuna ulteriore elaborazione se new_image è None
            return

        self.image = new_image

        if new_image.ndim == 2:
            new_image = new_image[..., np.newaxis]

        # === ⚠️ EVITA di ri-aggiungere la banda se è già stata aggiunta da add_result_as_band
        if band_name is not None:
            # Verifica se la banda esiste già in logic_to_index OPPURE se l'immagine ha già abbastanza bande
            if hasattr(self, 'logic_to_index') and band_name in self.logic_to_index:
                self.update_band_combobox(band_name)
                self.update_band_display()
                self.update_status(f"Band '{band_name}' already exists, updated display only.")
                return
            if self.original_image_np is not None and new_image.shape[2] == 1:
                last_band = self.original_image_np[..., -1]
                if np.array_equal(last_band, new_image[..., 0]):
                    # Se il nome precedente era '__preview__', aggiornalo con quello reale
                    if hasattr(self, 'logic_to_index') and '__preview__' in self.logic_to_index:
                        preview_index = self.logic_to_index.pop('__preview__')
                        self.logic_to_index[band_name] = preview_index
                        band_idx = preview_index
                        self.band_map[f"B{band_idx + 1}"] = band_name
                        self.update_status(f"Preview band renamed as '{band_name}'")
                    self.update_band_combobox(band_name)
                    self.update_band_display()
                    return

        # === Usa/aggiorna lo STACK COMPLETO (preferisci multispectral_stack) ===
        base = None
        if hasattr(self, 'multispectral_stack') and isinstance(self.multispectral_stack,
                                                               np.ndarray) and self.multispectral_stack.ndim >= 2:
            base = self.multispectral_stack
        elif self.original_image_np is not None:
            base = self.original_image_np

        if base is not None:
            if base.ndim == 2:
                base = base[..., np.newaxis]
            if new_image.shape[:2] == base.shape[:2]:
                combined = np.concatenate((base, new_image), axis=2)
            else:
                messagebox.showerror("Error", "Dimension mismatch when adding image band.")
                return
        else:
            combined = new_image

        # Aggiorna sia lo stack multispettrale che l'array originale visibile
        self.multispectral_stack = combined
        self.original_image_np = combined

        # === Aggiorna self.band_map se band_name è definito (persistenza nomi logici)
        if band_name is not None:
            if not hasattr(self, 'band_map') or self.band_map is None:
                self.band_map = {}
            n_bands = self.original_image_np.shape[2]
            self.band_map[f"B{n_bands}"] = band_name  # assegna nome logico all'ultima banda

        self.update_band_combobox(band_name)
        self.update_band_display()
        self.update_status(f"Image band '{band_name}' added from Raster Calculator")

    def update_band_combobox(self, band_name):
        # Inizializza strutture
        if not hasattr(self, 'logic_to_index') or self.logic_to_index is None:
            self.logic_to_index = {}
        if not hasattr(self, 'band_map') or self.band_map is None:
            self.band_map = {}

        base_arr = self.multispectral_stack if hasattr(self, 'multispectral_stack') and isinstance(
            self.multispectral_stack, np.ndarray) else self.original_image_np
        n_bands = base_arr.shape[2] if (isinstance(base_arr, np.ndarray) and base_arr.ndim == 3) else 1

        # Escludi bande ignorate (persistite dal Raster Calculator)
        ignored = getattr(self, 'ignored_bands', set()) or set()
        valid_indices = [i for i in range(n_bands) if i not in ignored]

        # Se esiste una preview temporanea, rinominala
        if band_name and "__preview__" in self.logic_to_index:
            preview_index = self.logic_to_index.pop("__preview__")
            self.logic_to_index[band_name] = preview_index

        # Costruisci una mappa indice->nome partendo dai nomi già assegnati
        prev_idx_to_name = {}
        for name, idx in list(self.logic_to_index.items()):
            if isinstance(idx, int) and idx in valid_indices:
                prev_idx_to_name[idx] = name

        for i in valid_indices:
            key = f"B{i + 1}"
            if key in self.band_map and self.band_map[key]:
                prev_idx_to_name[i] = self.band_map[key]

        # Applica override da band_map (es. quando alcune bande sono state nominate esplicitamente B1..Bn)
        for i in range(n_bands):
            key = f"B{i + 1}"
            if key in self.band_map and self.band_map[key]:
                prev_idx_to_name[i] = self.band_map[key]

        # Se stiamo aggiungendo una nuova banda, assicuriamoci che l'ultima prenda il suo nome logico
        if band_name:
            last_idx = valid_indices[-1] if valid_indices else (n_bands - 1)
            prev_idx_to_name[last_idx] = band_name

        # Riempimento dei “buchi” con nomi sensati
        default_rgb = ["R", "G", "B"]
        used_names = set(prev_idx_to_name.values())
        for i in valid_indices:
            if i not in prev_idx_to_name:
                candidate = default_rgb[i] if i < 3 and default_rgb[i] not in used_names else f"Band {i + 1}"
                prev_idx_to_name[i] = candidate
                used_names.add(candidate)

        # Ricostruisci logic_to_index **ordinato per indice** per mantenere l'ordine B1..Bn
        self.logic_to_index.clear()
        for i in valid_indices:
            self.logic_to_index[prev_idx_to_name[i]] = i

        # Prepara valori per la combobox: RGB (se R,G,B presenti) + lista ordinata per indice
        band_values = []
        if all(x in self.logic_to_index for x in ["R", "G", "B"]):
            band_values.append("RGB")
        band_values.extend(prev_idx_to_name[i] for i in valid_indices)

        # Mantieni la selezione corrente se possibile
        current = self.band_combobox.get() if hasattr(self, 'band_combobox') else None
        self.band_combobox['values'] = band_values
        self.band_combobox.config(state="readonly")

        if band_name and band_name in band_values:
            self.band_combobox.set(band_name)
            self.band.set(band_name)
        elif current in band_values and current:
            self.band_combobox.set(current)
            self.band.set(current)
        else:
            # fallback: seleziona l'ultima voce (tipicamente la nuova banda)
            self.band_combobox.set(band_values[-1])
            self.band.set(band_values[-1])

    def open_anomaly_editor(self):
        if self.analyzed_image is None:
            messagebox.showinfo("Anomaly Editor", "No anomalies to edit.")
            return

        if hasattr(self, "analyzed_clean_image"):
            clean_array = self.analyzed_clean_image
        else:
            clean_array = self.analyzed_image  # fallback di sicurezza

        if clean_array.ndim == 2:
            clean_image_pil = Image.fromarray(clean_array.astype('uint8'), 'L').convert('RGB')
        elif clean_array.ndim == 3:
            if clean_array.shape[2] == 1:
                clean_img = np.repeat(clean_array, 3, axis=2)
                clean_image_pil = Image.fromarray(clean_img.astype('uint8'), 'RGB')
            else:
                clean_image_pil = Image.fromarray(clean_array.astype('uint8'))
        else:
            messagebox.showerror("Error", "Unsupported image format for anomaly editor.")
            return

        editor = AnomalyEditor(self, self.image, clean_image_pil, self.update_image_from_anomaly_editor)

    def update_image_from_anomaly_editor(self, new_image):
        self.image = new_image
        self.image_processing_gui.display_image_on_canvas(self.image)
        self.update_status("Image updated from Anomaly Editor")

    def get_selected_image(self):
        """
        Restituisce la porzione dell'immagine corrente relativa alla selezione,
        se presente; altrimenti restituisce l'intera immagine.
        """
        # Ottieni le dimensioni dell'immagine originale
        img_h, img_w = self.original_image_np.shape[:2]

        if self.image_selection.polygon_points:
            try:
                x_coords, y_coords = zip(*self.image_selection.polygon_points)
            except Exception:
                return self.image
            sel_left = max(int(min(x_coords) - self.x_offset), 0)
            sel_right = min(int(max(x_coords) - self.x_offset), img_w)
            sel_top = max(int(min(y_coords) - self.y_offset), 0)
            sel_bottom = min(int(max(y_coords) - self.y_offset), img_h)
            return self.image[sel_top:sel_bottom, sel_left:sel_right, :]
        elif self.image_selection.rect_coords:
            left, top, right, bottom = self.image_selection.rect_coords
            left = max(int(left), 0)
            top = max(int(top), 0)
            right = min(int(right), img_w)
            bottom = min(int(bottom), img_h)
            return self.image[top:bottom, left:right, :]
        else:
            return self.image

    def save_plot(self):
        self.canvas.update()

        if not hasattr(self.image_processing_gui, "canvas_image"):
            self.update_status("Error: no canvas_image found.")
            return

        bbox = self.canvas.bbox(self.image_processing_gui.canvas_image)
        if bbox is None:
            self.update_status("Error: no image to save.")
            return

        x1, y1, x2, y2 = bbox

        top_margin = 120
        side_margin = 120
        bottom_margin = 60
        x1 = max(0, x1 - side_margin)
        y1 = max(0, y1 - top_margin)

        scalebar_extra = 0
        if hasattr(self.image_processing_gui, "display_w") and hasattr(self.image_processing_gui, "zoom_factor") and \
                self.geotransform is not None:
            try:
                pixel_size = abs(self.geotransform.a)
                target_length_m = 100
                scale_px = int(target_length_m / pixel_size * self.image_processing_gui.zoom_factor)
                scalebar_extra = scale_px // 2 + 100
            except Exception:
                scalebar_extra = 200  # fallback di sicurezza
        else:
            scalebar_extra = 200

        x2 = x2 + side_margin + scalebar_extra
        y2 = y2 + bottom_margin

        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        scale = 3

        ps = self.canvas.postscript(
            colormode='color',
            x=0, y=0, width=w, height=h,
            pagewidth=w * scale, pageheight=h * scale
        )

        try:
            img = Image.open(io.BytesIO(ps.encode("utf-8")))
            cropped = img.crop((x1 * scale, y1 * scale, x2 * scale, y2 * scale))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture canvas: {e}")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            cropped.save(file_path, dpi=(300, 300))
            self.update_status(f"Plot saved to {file_path}")

    def update_lisa_display(self, event=None):
        if self.lisa_results is None or not isinstance(self.lisa_results, dict):
            return

        index_name = self.lisa_index_combobox.get()
        key_map = {
            "Moran's I": "moran",
            "Getis-Ord": "getis",
            "INDIC (Moran n Getis)": "indic"
        }

        if index_name not in key_map:
            return

        chosen_key = key_map[index_name]
        lisa_map = self.lisa_results[chosen_key]

        # Applica maschera della selezione se presente
        selection_mask = None
        if self.image_selection.polygon_points:
            img_h, img_w = self.original_image_np.shape[:2]
            normalized = [(x / img_w, y / img_h) for (x, y) in self.image_selection.polygon_points]
            selection_mask = create_mask((img_h, img_w), normalized, normalize=True)
        elif self.image_selection.rect_coords:
            left, top, right, bottom = self.image_selection.rect_coords
            img_h, img_w = self.original_image_np.shape[:2]
            selection_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            selection_mask[top:bottom, left:right] = 1

        # Visualizzazione limitata alla selezione
        arr = np.array(lisa_map, dtype=float)
        if selection_mask is not None:
            arr[selection_mask == 0] = np.nan

        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mx - mn > 1e-9:
            arr = (arr - mn) / (mx - mn) * 255.0
        arr = np.nan_to_num(arr, nan=0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        arr_3d = arr[..., np.newaxis]

        # Combina con immagine originale
        if hasattr(self, 'image_histo_processing') and self.image_histo_processing is not None:
            base_image = np.array(self.image_histo_processing)
        else:
            base_image = np.array(self.original_image_np)

        if base_image.ndim == 2 or base_image.shape[2] == 1:
            base_image = np.repeat(base_image, 3, axis=2)

        from image_processing import highlight_anomalies
        overlay = highlight_anomalies(base_image, arr > 0, self.color.get())
        self.image = overlay
        self.image_processing_gui.display_image_on_canvas(overlay)
        self.update_status(f"Displaying LISA index: {index_name}")

    def activate_histo_pixel_mode(self):
        self.update_status("Click on the image to view pixel histogram")
        self.canvas.bind("<Button-1>", self.on_pixel_click_for_histogram)

    def on_pixel_click_for_histogram(self, event):
        self.canvas.unbind("<Button-1>")
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        offset_x = (canvas_width - self.display_w) // 2
        offset_y = (canvas_height - self.display_h) // 2

        x_img = int((event.x - offset_x) / self.effective_scale_factor)
        y_img = int((event.y - offset_y) / self.effective_scale_factor)

        # Controllo bounds
        if (x_img < 0 or y_img < 0
                or x_img >= self.image.shape[1]
                or y_img >= self.image.shape[0]):
            self.update_status("Clicked outside image area")
            return

        # Estrai il pixel dalla self.image (non dall'original_image_np).
        # self.image potrebbe essere (H, W) se 1 canale, (H, W, 1) se single band, (H, W, 3) se RGB
        import numpy as np
        if self.image.ndim == 2:
            pixel_array = self.image[y_img:y_img + 1, x_img:x_img + 1]
        else:
            pixel_array = self.image[y_img:y_img + 1, x_img:x_img + 1, :]

        # Apri la finestra dell'istogramma in modalità pixel
        from histogram import open_histogram_window
        open_histogram_window(self, pixel_array, self.original_image_pil, self.update_image_from_histogram,
                              pixel_mode=True)

    def open_lisa_config_window(self, selection_mask=None):
        # Crea una finestra Toplevel per chiedere il tipo di immagine per LISA
        config_win = tk.Toplevel(self)
        config_win.title("LISA Configuration")

        tk.Label(config_win, text="Select image type for LISA:").pack(side=tk.TOP, padx=10, pady=10)

        # Variabile per memorizzare la scelta; default "RGB"
        self.lisa_image_type = tk.StringVar(value="RGB")

        # Tre opzioni: "RGB (8 neighbors)", "Single Band/Index", "NDVI"
        tk.Radiobutton(config_win, text="RGB (8 neighbors)", variable=self.lisa_image_type, value="RGB").pack(
            anchor=tk.W)
        tk.Radiobutton(config_win, text="Single Band/Index", variable=self.lisa_image_type, value="Single").pack(
            anchor=tk.W)

        # Pulsante OK per confermare la scelta
        ok_btn = tk.Button(config_win, text="OK", command=lambda: self.on_lisa_config_ok(config_win))
        ok_btn.pack(side=tk.BOTTOM, pady=10)

    def on_lisa_config_ok(self, config_win):
        config_win.destroy()
        image_type = self.lisa_image_type.get()
        threshold = self.threshold.get()

        image_array = np.array(self.image)
        if image_array.ndim == 2:
            image_array = image_array[..., np.newaxis]
        if image_array.shape[2] == 1:
            image_array = np.repeat(image_array, 3, axis=2)

        img_width, img_height = self.original_image_pil.size
        mask_size = (img_height, img_width)

        selection_mask = None

        if self.image_selection.polygon_points:
            normalized_polygon_points = [(x / img_width, y / img_height)
                                         for x, y in self.image_selection.polygon_points]
            selection_mask = create_mask(mask_size, normalized_polygon_points, normalize=True)

        elif self.image_selection.rect_coords:
            left, top, right, bottom = self.image_selection.rect_coords
            left = max(int(left), 0)
            right = min(int(right), img_width)
            top = max(int(top), 0)
            bottom = min(int(bottom), img_height)
            selection_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            selection_mask[top:bottom, left:right] = 1

        # Applica la maschera se esiste
        masked_image = image_array.copy()
        if selection_mask is not None:
            masked_image[selection_mask == 0] = 0

        # Esegui LISA
        anomalies = find_anomalies(
            masked_image,
            threshold,
            self.anomaly_type.get(),
            "LISA",
            image_type,
            selection_mask  # ← passaggio fondamentale
        )

        self.lisa_results = anomalies
        self.lisa_index_label.pack(side=tk.LEFT)
        self.lisa_index_combobox.pack(side=tk.LEFT, padx=5)
        self.lisa_index_combobox.config(state="readonly")
        self.lisa_index_combobox.set("Moran's I")
        # Visualizza immediatamente l'indice di default con overlay nella selezione
        selected_key = "moran"  # default iniziale
        if self.lisa_index_combobox.get().lower() == "getis":
            selected_key = "getis"
        elif self.lisa_index_combobox.get().lower().startswith("indic"):
            selected_key = "indic"

        selected_mask = self.lisa_results.get(selected_key)

        # Recupera immagine base
        image_array = np.array(self.image)
        if image_array.ndim == 2:
            image_array = image_array[..., np.newaxis]
        if image_array.shape[2] == 1:
            image_array = np.repeat(image_array, 3, axis=2)

        # Applica overlay SOLO sulla selezione
        img_width, img_height = image_array.shape[1], image_array.shape[0]
        if self.image_selection.polygon_points:
            points = [(x / img_width, y / img_height) for (x, y) in self.image_selection.polygon_points]
            selection_mask = create_mask((img_height, img_width), points, normalize=True)
        elif self.image_selection.rect_coords:
            left, top, right, bottom = self.image_selection.rect_coords
            selection_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            selection_mask[top:bottom, left:right] = 1
        else:
            selection_mask = np.ones((img_height, img_width), dtype=np.uint8)

        # Limita il risultato alla sola area selezionata
        visible_mask = selected_mask & (selection_mask.astype(bool))

        from image_processing import composite_anomaly_overlay
        composite = composite_anomaly_overlay(image_array, visible_mask, self.color.get(), opacity=200)

        self.image = composite
        self.image_processing_gui.display_image_on_canvas(self.image)
        # Abilita pulsanti di editing e salvataggio
        self.save_button.config(state=tk.NORMAL)
        self.save_wip_button.config(state=tk.NORMAL)
        self.undo_button.config(state=tk.NORMAL)
        self.redo_button.config(state=tk.DISABLED)

        # Allinea al comportamento degli altri metodi: attiva l'anomaly editor
        self.analyzed_image = np.copy(self.image)
        self.anomaly_editor_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = ImageAnalyzer()
    app.mainloop()
