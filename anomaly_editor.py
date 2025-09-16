import tkinter as tk
import numpy as np
import io
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
from config import LOGO_CHANGES_TRASP_PATH, LOGO_ISPC_TRASP_PATH
from scipy.ndimage import generic_filter
from PIL import Image
from tooltip import Tooltip, get_tooltip
from skimage import measure



class AnomalyEditor(tk.Toplevel):
    def __init__(self, parent, anomaly_image, clean_image, update_callback):
        super().__init__(parent)
        self.anomaly_image = anomaly_image
        screen_width = int(self.winfo_screenwidth() * 0.9)
        screen_height = int(self.winfo_screenheight() * 0.9)
        self.geometry(f"{screen_width}x{screen_height}")
        self.parent = parent
        self.update_callback = update_callback
        self.title("Anomaly Editor")

        # Ottieni dimensioni canvas principale
        parent.update_idletasks()
        canvas_width = parent.canvas.winfo_width()
        canvas_height = parent.canvas.winfo_height()

        # Aggiungi margine verticale extra
        extra_margin = 240  # margine per coordinate sopra/sotto
        if canvas_width > 1 and canvas_height > 1:
            self.geometry(f"{canvas_width}x{canvas_height + extra_margin}")

        # Parametri di zoom e pan (per la working image)
        self.zoom_factor = 1.0
        self.x_offset = 0
        self.y_offset = 0

        # Crea il frame per i loghi in alto
        self.logo_frame = tk.Frame(self)
        self.logo_frame.pack(fill=tk.X)
        self.logo_frame.columnconfigure(0, weight=1)
        self.logo_frame.columnconfigure(1, weight=1)
        self.logo_label_left = tk.Label(self.logo_frame)
        self.logo_label_left.pack(side=tk.LEFT, padx=10)
        self.logo_label_right = tk.Label(self.logo_frame)
        self.logo_label_right.pack(side=tk.RIGHT, padx=10)
        self.load_logos()

        # Converti le immagini in RGBA
        if isinstance(anomaly_image, np.ndarray):
            self.anomaly_img = Image.fromarray(anomaly_image.astype('uint8')).convert("RGBA")
        else:
            self.anomaly_img = anomaly_image.copy().convert("RGBA")

        if isinstance(clean_image, np.ndarray):
            self.clean_img = Image.fromarray(clean_image.astype('uint8')).convert("RGBA")
        else:
            self.clean_img = clean_image.copy().convert("RGBA")

        # Copie di lavoro
        self.original_clean_img = self.clean_img.copy()
        self.current_img = self.anomaly_img.copy()

        # Salva maschera di selezione se esiste
        self.selection_mask = getattr(parent, "selection_mask", None)
        self.selection_bbox = getattr(parent, "selection_bbox", None)

        # Storico per undo/redo
        self.history = [self.current_img.copy()]
        self.history_index = 0

        # Canvas principale
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Toolframe in basso
        scroll_frame = tk.Frame(self)
        scroll_frame.pack(fill=tk.X, padx=10)

        x_scroll = tk.Scrollbar(scroll_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.tools_canvas = tk.Canvas(scroll_frame, height=80, xscrollcommand=x_scroll.set)
        self.tools_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)

        x_scroll.config(command=self.tools_canvas.xview)

        self.tools_frame = tk.Frame(self.tools_canvas)
        self.tools_canvas.create_window((0, 0), window=self.tools_frame, anchor='nw')
        self.tools_frame.bind("<Configure>",
                              lambda e: self.tools_canvas.config(scrollregion=self.tools_canvas.bbox("all")))

        self.update_idletasks()
        self.tools_canvas.config(scrollregion=self.tools_canvas.bbox("all"))

        # Brush
        tk.Label(self.tools_frame, text="Brush size:").pack(side=tk.LEFT)
        self.brush_size = tk.IntVar(value=10)
        self.brush_slider = tk.Scale(self.tools_frame, from_=1, to=50,
                                     orient=tk.HORIZONTAL, variable=self.brush_size)
        self.brush_slider.pack(side=tk.LEFT, padx=5)

        self.erase_btn = tk.Button(self.tools_frame, text="Erase", command=self.activate_erase)
        self.erase_btn.pack(side=tk.LEFT, padx=5)
        self.erase_selection_btn = tk.Button(self.tools_frame, text="Erase by Selection", command=self.erase_by_selection)
        self.erase_selection_btn.pack(side=tk.LEFT, padx=5)
        Tooltip(self.erase_selection_btn, "Erase highlights inside the selected polygonal area.")
        # Polygon selection state
        self.polygon_points = []
        self.poly_line = None
        self.selection_in_progress = False
        self.canvas.bind("<ButtonPress-1>", self.on_polygon_click)
        self.canvas.bind("<Double-Button-1>", self.on_polygon_finish)

        self.zoom_in_btn = tk.Button(self.tools_frame, text="Zoom In", command=self.zoom_in)
        self.zoom_in_btn.pack(side=tk.LEFT, padx=5)

        self.zoom_out_btn = tk.Button(self.tools_frame, text="Zoom Out", command=self.zoom_out, state=tk.DISABLED)
        self.zoom_out_btn.pack(side=tk.LEFT, padx=5)

        self.pan_button = tk.Button(self.tools_frame, text="Move", command=self.start_pan, state=tk.DISABLED)
        self.pan_button.pack(side=tk.LEFT, padx=5)

        # Outlier
        self.outlier_method = tk.StringVar(value="Standard")

        tk.Label(self.tools_frame, text="Outlier Method:").pack(side=tk.LEFT, padx=5)
        self.outlier_method_menu = ttk.Combobox(
            self.tools_frame, textvariable=self.outlier_method,
            values=["Standard", "Z-score", "Local Contrast"], state="readonly", width=12
        )
        self.outlier_method_menu.current(0)
        self.outlier_method_menu.pack(side=tk.LEFT)
        self.outlier_method_menu.bind("<<ComboboxSelected>>", self.on_outlier_method_change)
        # Tooltip for outlier method dropdown
        Tooltip(self.outlier_method_menu, "Select the method used to detect outliers: Standard, Z-Score or Local Contrast.")

        self.outlier_param_frame = tk.Frame(self.tools_frame)
        self.outlier_param_frame.pack(side=tk.LEFT, padx=5)

        self.outlier_slider = tk.Scale(self.outlier_param_frame, from_=80, to=99, orient=tk.HORIZONTAL,
                                       label="Threshold (%)", length=120)
        Tooltip(self.outlier_slider, "Set the pixel intensity threshold for the Standard outlier detection method.")
        self.outlier_slider.set(90)

        self.zscore_slider = tk.Scale(self.outlier_param_frame, from_=0.5, to=3.0, resolution=0.1,
                                      orient=tk.HORIZONTAL, label="Z-Score", length=120)
        self.zscore_slider.set(2.0)
        # Tooltip for Z-score slider
        Tooltip(self.zscore_slider, "Adjust the Z-score threshold. Higher values make detection stricter.")

        self.window_size = tk.StringVar(value="3x3")
        self.window_menu = ttk.Combobox(self.outlier_param_frame, textvariable=self.window_size,
                                        values=["1x1", "3x3", "5x5"], state="readonly", width=6)
        # Tooltip for window size menu
        Tooltip(self.window_menu, "Choose the window size for local contrast detection.")

        self.outlier_slider.pack()

        self.remove_outlier_btn = tk.Button(self.tools_frame, text="Remove Outlier", command=self.remove_outlier)
        self.remove_outlier_btn.pack(side=tk.LEFT, padx=5)
        # Tooltip for Remove Outlier button
        Tooltip(self.remove_outlier_btn, "Apply the selected outlier detection method to clean the image.")

        self.undo_btn = tk.Button(self.tools_frame, text="Undo", command=self.undo, state=tk.DISABLED)
        self.undo_btn.pack(side=tk.LEFT, padx=5)

        self.redo_btn = tk.Button(self.tools_frame, text="Redo", command=self.redo, state=tk.DISABLED)
        self.redo_btn.pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(self.tools_frame, text="Reset Editor", command=self.reset_editor)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(self.tools_frame, text="Export Edited Image", command=self.save_edited)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.save_plot_btn = tk.Button(self.tools_frame, text="Save Plot", command=self.save_plot)
        self.save_plot_btn.pack(side=tk.LEFT, padx=5)
        if not hasattr(self.parent, "crs") or not hasattr(self.parent, "geotransform") or \
                self.parent.crs is None or self.parent.geotransform is None:
            self.save_plot_btn.config(state=tk.DISABLED)
        Tooltip(self.save_plot_btn, get_tooltip("save_plot_button"))


        self.status_label = tk.Label(self, text="Use 'Erase' tool to remove anomalies.",
                                     bd=1, relief=tk.SUNKEN, anchor="w")
        self.status_label.pack(fill=tk.X)

        self.eraser_active = False
        self.manual_zoom = False
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

        self.display_image()
        self.adapt_brush_to_image_size()
        self.bind("<Configure>", self.on_main_resize)

        self.export_button = tk.Button(self.tools_frame, text="Export Shapefile", command=self._on_export_shapefile)
        self.export_button.pack(side=tk.LEFT, padx=5)
        Tooltip(self.export_button,
                "Export detected highlights as a line shapefile in geographic coordinates.\nNearby points will be grouped based on the distance threshold.")

        # Gap UI for shapefile export
        export_frame = self.tools_frame
        self.gap_label = tk.Label(export_frame, text="Gap:")
        self.gap_entry = tk.Entry(export_frame, width=4)
        self.gap_entry.insert(0, "3")
        self.gap_tooltip = Tooltip(self.gap_entry, "Maximum pixel distance to group highlights")
        # Use pack instead of grid for gap_label and gap_entry
        self.gap_label.pack(side=tk.LEFT, padx=(10, 0))
        self.gap_entry.pack(side=tk.LEFT, padx=(5, 0))

        # Enable/disable export button and gap fields based on georeferencing
        if self.parent.geotransform is not None and self.parent.crs is not None:
            self.export_button.config(state=tk.NORMAL)
            self.gap_label.pack(side=tk.LEFT, padx=(10, 0))
            self.gap_entry.pack(side=tk.LEFT, padx=(5, 0))
        else:
            self.export_button.config(state=tk.DISABLED)
            self.gap_label.pack_forget()
            self.gap_entry.pack_forget()

    def on_main_resize(self, event):
        self.display_image()

    def load_logos(self):
        # Logo a sinistra
        try:
            logo_left = Image.open(LOGO_ISPC_TRASP_PATH)
            orig_w, orig_h = logo_left.size
            new_h = 100
            new_w = int(new_h * (orig_w / orig_h))
            logo_left_resized = logo_left.resize((new_w, new_h), Image.LANCZOS).convert("RGBA")
            self.tk_logo_left = ImageTk.PhotoImage(logo_left_resized)
            self.logo_label_left.config(image=self.tk_logo_left)
        except Exception:
            pass

        # Logo a destra
        try:
            logo_right = Image.open(LOGO_CHANGES_TRASP_PATH)
            orig_w, orig_h = logo_right.size
            new_h = 100
            new_w = int(new_h * (orig_w / orig_h))
            logo_right_resized = logo_right.resize((new_w, new_h), Image.LANCZOS).convert("RGBA")
            self.tk_logo_right = ImageTk.PhotoImage(logo_right_resized)
            self.logo_label_right.config(image=self.tk_logo_right)
        except Exception:
            pass

    def display_image(self):
        self.update_idletasks()

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.after(100, self.display_image)
            return

        margin = 20
        # Calcola lo spazio effettivamente disponibile sottraendo altezza fissa dei widget (es. ~140px totali)
        widget_margin = 140  # Altezza stimata: logo_frame (100) + tools_frame (30) + status_label (10)
        available_w = canvas_width - 2 * margin
        available_h = canvas_height - widget_margin - 2 * margin

        orig_w, orig_h = self.current_img.size

        scale_w = available_w / orig_w
        scale_h = available_h / orig_h
        auto_scale = min(scale_w, scale_h, 1.0)

        if not hasattr(self, "manual_zoom") or not self.manual_zoom:
            self.zoom_factor = auto_scale

        zoomed_w = int(orig_w * self.zoom_factor)
        zoomed_h = int(orig_h * self.zoom_factor)
        zoomed_img = self.current_img.resize((zoomed_w, zoomed_h), Image.LANCZOS)

        final_img = zoomed_img
        self.display_scale = self.zoom_factor
        final_w, final_h = final_img.size

        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(final_img)

        x = (canvas_width - final_w) // 2 + self.x_offset
        y = (canvas_height - final_h) // 2 + self.y_offset

        self.image_x_offset = x
        self.image_y_offset = y

        self.canvas_image = self.canvas.create_image(x, y, anchor='nw', image=self.tk_image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        border_thickness = 2
        self.canvas.create_rectangle(
            x - border_thickness, y - border_thickness,
            x + final_w + border_thickness, y + final_h + border_thickness,
            outline="black", width=border_thickness
        )

        self.display_w = final_w
        self.display_h = final_h

        # ===== Coordinate geografiche (ticks, nord, scalebar) =====
        if hasattr(self.parent, "crs") and hasattr(self.parent, "geotransform") and \
                self.parent.crs is not None and self.parent.geotransform is not None:
            from rasterio.transform import xy
            from pyproj import Transformer, CRS

            crs = self.parent.crs
            geotransform = self.parent.geotransform
            current_crs = CRS(crs)
            transformer_to_geographic = Transformer.from_crs(current_crs, "EPSG:4326", always_xy=True)

            img_w, img_h = final_img.size
            h, w = self.current_img.size

            corners = [(0, 0), (0, w), (h, 0), (h, w)]
            lats, lons = [], []

            for row_c, col_c in corners:
                x_c, y_c = xy(geotransform, row_c, col_c)
                lon_c, lat_c = transformer_to_geographic.transform(x_c, y_c)
                lats.append(lat_c)
                lons.append(lon_c)

            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)

            def latlon_to_canvas(lat, lon):
                x_g, y_g = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
                col = (x_g - geotransform.c) / geotransform.a
                row = (y_g - geotransform.f) / geotransform.e
                x_canvas = self.image_x_offset + (col * (img_w / w))
                y_canvas = self.image_y_offset + (row * (img_h / h))
                return x_canvas, y_canvas

            def to_dms(value, is_lat=False):
                degrees = int(value)
                minutes = int((abs(value) - abs(degrees)) * 60)
                seconds = (abs(value) - abs(degrees)) * 3600 - minutes * 60
                hemi = 'N' if is_lat and value >= 0 else 'S' if is_lat else 'E' if value >= 0 else 'W'
                return f"{abs(degrees)}°{minutes}'{seconds:.1f}\"{hemi}"

            tick_len = 10
            lat_ticks = [lat_min, lat_max]
            lon_ticks = [lon_min, (lon_min + lon_max) / 2, lon_max]

            for lat_val in lat_ticks:
                x_c, y_c = latlon_to_canvas(lat_val, (lon_min + lon_max) / 2)
                lat_dms = to_dms(lat_val, is_lat=True)
                self.canvas.create_line(self.image_x_offset, y_c, self.image_x_offset - tick_len, y_c, fill="black")
                self.canvas.create_text(self.image_x_offset - tick_len - 40, y_c, text=lat_dms, fill="black",
                                        font=("Arial", 12, "bold"))
                self.canvas.create_line(self.image_x_offset + img_w, y_c, self.image_x_offset + img_w + tick_len, y_c,
                                        fill="black")
                self.canvas.create_text(self.image_x_offset + img_w + tick_len + 40, y_c, text=lat_dms, fill="black",
                                        font=("Arial", 12, "bold"))

            for lon_val in lon_ticks:
                x_c, y_c = latlon_to_canvas((lat_min + lat_max) / 2, lon_val)
                lon_dms = to_dms(lon_val, is_lat=False)
                self.canvas.create_line(x_c, self.image_y_offset, x_c, self.image_y_offset - tick_len, fill="black")
                self.canvas.create_text(x_c, self.image_y_offset - tick_len - 10, text=lon_dms, fill="black",
                                        font=("Arial", 12, "bold"))
                self.canvas.create_line(x_c, self.image_y_offset + img_h, x_c, self.image_y_offset + img_h + tick_len,
                                        fill="black")
                self.canvas.create_text(x_c, self.image_y_offset + img_h + tick_len + 10, text=lon_dms, fill="black",
                                        font=("Arial", 12, "bold"))

            # Freccia del Nord
            north_x = self.image_x_offset + img_w + 40
            north_y = self.image_y_offset + 60
            self.canvas.create_polygon(
                [(north_x, north_y), (north_x - 10, north_y + 30), (north_x + 10, north_y + 30)],
                fill="black", outline="black"
            )
            self.canvas.create_text(north_x, north_y - 15, text="N", font=("Arial", 15, "bold"), fill="black")

            # Barra di scala
            pixel_width = abs(geotransform.a)
            scale_factor = self.zoom_factor
            pixel_width_m_displayed = pixel_width / scale_factor
            gui_width = self.winfo_width()
            scale_length_px_gui = int((gui_width / 20) * 1.5)
            proposed_length_m = scale_length_px_gui * pixel_width_m_displayed

            base_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            chosen_scale_m = min(base_values, key=lambda x: abs(x - proposed_length_m))
            scalebar_length_px = int(chosen_scale_m / pixel_width_m_displayed)

            segment_count = 5
            segment_px = scalebar_length_px // segment_count
            scalebar_x = self.image_x_offset + img_w + 20
            scalebar_y = self.image_y_offset + img_h - 50

            for seg in range(segment_count):
                seg_x1 = scalebar_x + seg * segment_px
                seg_x2 = seg_x1 + segment_px
                fill_color = "black" if seg % 2 == 0 else "white"
                self.canvas.create_rectangle(seg_x1, scalebar_y, seg_x2, scalebar_y + 10,
                                             fill=fill_color, outline="black")

            self.canvas.create_text(scalebar_x, scalebar_y + 20, text="0", fill="black", font=("Arial", 15, "bold"))
            label_text = f"{chosen_scale_m} m" if chosen_scale_m < 1000 else f"{chosen_scale_m / 1000:.1f} km"
            self.canvas.create_text(scalebar_x + scalebar_length_px, scalebar_y + 20,
                                    text=label_text, fill="black", font=("Arial", 15, "bold"))

    def activate_erase(self):
        # Disabilita eventuali binding di pan
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.eraser_active = True
        self.display_image()  # Assicura che l'immagine sia aggiornata e centrata
        # Salva i valori correnti come fissi per la modalità erase
        self.fixed_offset_x = self.image_x_offset
        self.fixed_offset_y = self.image_y_offset
        self.fixed_zoom = self.zoom_factor
        self.status_label.config(text="Erase mode: drag to restore original pixels.")
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def on_canvas_drag(self, event):
        if not self.eraser_active:
            return
        x_img = int((event.x - self.image_x_offset) / self.zoom_factor)
        y_img = int((event.y - self.image_y_offset) / self.zoom_factor)
        r = self.brush_size.get()
        box = (
            max(0, x_img - r),
            max(0, y_img - r),
            min(self.current_img.width, x_img + r),
            min(self.current_img.height, y_img + r)
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            return
        mask = Image.new("L", (box[2] - box[0], box[3] - box[1]), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, box[2] - box[0], box[3] - box[1]), fill=255)
        clean_resized = self.clean_img.resize(self.current_img.size, Image.LANCZOS)
        region = clean_resized.crop(box)
        self.current_img.paste(region, box, mask)
        self.display_image()
        self._pending_erase = True

    def on_mouse_release(self, event):
        if getattr(self, "_pending_erase", False):
            if self.history_index == len(self.history) - 1:
                self.history.append(self.current_img.copy())
            else:
                self.history = self.history[:self.history_index + 1]
                self.history.append(self.current_img.copy())
            self.history_index += 1
            self.update_undo_redo_buttons()
            self._pending_erase = False

    def adapt_brush_to_image_size(self):
        img_w, img_h = self.current_img.size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        scaling_ratio = max(img_w / canvas_w, img_h / canvas_h)

        # Definizione base
        base_min = 1
        base_max = 50
        base_default = 10

        if scaling_ratio > 1:
            factor = min(max(scaling_ratio, 1.0), 5.0)  # Clamp tra 1.0 e 5.0
            new_min = int(base_min * factor)
            new_max = int(base_max * factor)
            new_default = int(base_default * factor)

            new_min = max(new_min, 1)
            new_max = min(new_max, 300)
            new_default = min(new_default, new_max)
        else:
            new_min = base_min
            new_max = base_max
            new_default = base_default

        self.brush_slider.config(from_=new_min, to=new_max)
        self.brush_size.set(new_default)

    def zoom_in(self):
        self.manual_zoom = True
        self.zoom_factor *= 1.2
        # Se andiamo oltre 1.0, abilitiamo lo Zoom Out
        if self.zoom_factor > 1.0:
            self.zoom_out_btn.config(state=tk.NORMAL)

        self.display_image()
        self.pan_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Zoom In => factor={self.zoom_factor:.2f}")

    def zoom_out(self):
        self.manual_zoom = True
        new_factor = self.zoom_factor / 1.2
        if new_factor < 1.0:
            new_factor = 1.0

        self.zoom_factor = new_factor
        # Se torniamo a 1.0, disabilitiamo il pulsante Zoom Out
        if abs(self.zoom_factor - 1.0) < 1e-9:
            self.zoom_out_btn.config(state=tk.DISABLED)

        self.display_image()
        self.status_label.config(text=f"Zoom Out => factor={self.zoom_factor:.2f}")

    def start_pan(self):
        self.eraser_active = False  # Disattiva eventuale modalità erase
        self.canvas.config(cursor="fleur")
        self.canvas.bind("<ButtonPress-1>", self.start_pan_drag)
        self.canvas.bind("<B1-Motion>", self.pan_image)
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)

    def start_pan_drag(self, event):
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        print(f"[DEBUG] Start pan: {self.pan_start_x}, {self.pan_start_y}")

    def pan_image(self, event):
        if not self.panning:
            return
        # Log prima dell'aggiornamento
        print("[DEBUG] pan_image - prima: pan_start_x =", self.pan_start_x,
              "pan_start_y =", self.pan_start_y)
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        print("[DEBUG] pan_image - delta: dx =", dx, "dy =", dy)
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.x_offset += dx
        self.y_offset += dy
        print("[DEBUG] pan_image - nuovi offset: x_offset =", self.x_offset,
              "y_offset =", self.y_offset)
        self.display_image()

    def end_pan(self, event):
        self.panning = False
        self.canvas.config(cursor="")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self._pending_erase = False

    def reset_editor(self):
        self.current_img = self.anomaly_img.copy()
        self.zoom_factor = 1.0
        self.x_offset = 0
        self.y_offset = 0
        self.eraser_active = False
        self.canvas.update_idletasks()
        self.display_image()
        self.status_label.config(text="Editor reset to original anomaly image.")
        self.zoom_out_btn.config(state=tk.DISABLED)
        self.pan_button.config(state=tk.DISABLED)
        self.adapt_brush_to_image_size()

        self.outlier_slider.set(90)
        self.zscore_slider.set(2.0)
        self.window_size.set("3x3")
        self.outlier_method.set("Standard")
        self.outlier_method_menu.set("Standard")
        self.update_outlier_method_gui()

        self.history = [self.current_img.copy()]
        self.history_index = 0
        self.update_undo_redo_buttons()

    def save_edited(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Edited Image",
            filetypes=[
                ("PNG files", "*.png"),
                ("GeoTIFF files", "*.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not file_path:
            return

        # Correggi estensione se l'utente non l'ha specificata
        if not any(file_path.lower().endswith(ext) for ext in [".png", ".tif", ".tiff"]):
            # Imposta ".png" come default, se nulla specificato
            file_path += ".png"

        if file_path:
            img_array = np.array(self.current_img)

            # Se l'immagine è georiferita salva in GeoTIFF
            if file_path.lower().endswith((".tif", ".tiff")) and hasattr(self.parent, "crs") and hasattr(self.parent,
                                                                                                         "geotransform"):
                import rasterio
                from rasterio.transform import Affine

                height, width = img_array.shape[:2]
                transform = self.parent.geotransform
                if isinstance(transform, Affine):
                    aff = transform
                else:
                    aff = Affine(transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)

                with rasterio.open(
                        file_path,
                        "w",
                        driver="GTiff",
                        height=height,
                        width=width,
                        count=4,
                        dtype=img_array.dtype,
                        crs=self.parent.crs,
                        transform=aff
                ) as dst:
                    for i in range(4):  # RGBA
                        dst.write(img_array[..., i], i + 1)
            else:
                self.current_img.save(file_path)

            self.status_label.config(text=f"Edited image saved to {file_path}")
            if hasattr(self, "update_callback") and self.update_callback:
                self.update_callback(img_array)

    def update_outlier_method_gui(self, event=None):
        # Nasconde tutto
        for widget in self.outlier_param_frame.winfo_children():
            widget.pack_forget()

        method = self.outlier_method.get()

        # Mostra solo il parametro del metodo selezionato
        if method == "Standard":
            self.outlier_slider.pack(side=tk.LEFT)
        elif method == "Z-score":
            self.zscore_slider.pack(side=tk.LEFT)
        elif method == "Local Contrast":
            self.window_menu.pack(side=tk.LEFT)

        self.status_label.config(text=f"Outlier method: {method}")

    def on_outlier_method_change(self, event=None):
        method = self.outlier_method.get()

        # Nasconde tutti i controlli
        for widget in self.outlier_param_frame.winfo_children():
            widget.pack_forget()

        # Mostra solo quello del metodo selezionato
        if method == "Standard":
            self.outlier_slider.pack(side=tk.LEFT)
        elif method == "Z-score":
            self.zscore_slider.pack(side=tk.LEFT)
        elif method == "Local Contrast":
            self.window_menu.pack(side=tk.LEFT)

        self.status_label.config(text=f"Outlier method: {method}")

    def remove_outlier(self):
        # Salva stato per undo
        if self.history_index == len(self.history) - 1:
            self.history.append(self.current_img.copy())
        else:
            self.history = self.history[:self.history_index + 1]
            self.history.append(self.current_img.copy())
        self.history_index += 1
        self.update_undo_redo_buttons()

        current_np = np.array(self.current_img)
        clean_np = np.array(self.clean_img.resize(self.current_img.size, Image.LANCZOS))

        method = self.outlier_method.get()

        if self.selection_mask is not None:
            mask = self.selection_mask
            if mask.shape != current_np.shape[:2]:
                mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(current_np.shape[:2][::-1], Image.NEAREST)) > 0
        elif self.selection_bbox is not None:
            mask = np.zeros(current_np.shape[:2], dtype=bool)
            x1, y1, x2, y2 = self.selection_bbox
            mask[y1:y2, x1:x2] = True
        else:
            mask = None

        if method == "Standard":
            val = self.outlier_slider.get()
            filtered = remove_outliers_standard(current_np, clean_np, percentile_threshold=val, mask=mask)
        elif method == "Z-score":
            val = self.zscore_slider.get()
            filtered = remove_outliers_zscore(current_np, clean_np, z_threshold=val, mask=mask)
        elif method == "Local Contrast":
            val_str = self.window_size.get().replace("x", "")
            filtered = remove_outliers_local_contrast(current_np, clean_np, window_size=int(val_str), mask=mask)
        else:
            self.status_label.config(text="Unknown method selected.")
            return

        self.current_img = Image.fromarray(filtered)
        self.display_image()
        self.status_label.config(text=f"Outlier removal ({method}) applied.")

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.current_img = self.history[self.history_index].copy()
            self.display_image()
            self.update_undo_redo_buttons()
            self.status_label.config(text="Undo applied")

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.current_img = self.history[self.history_index].copy()
            self.display_image()
            self.update_undo_redo_buttons()
            self.status_label.config(text="Redo applied")

    def update_undo_redo_buttons(self):
        self.undo_btn.config(state=tk.NORMAL if self.history_index > 0 else tk.DISABLED)
        self.redo_btn.config(state=tk.NORMAL if self.history_index < len(self.history) - 1 else tk.DISABLED)

    def save_plot(self):
        self.canvas.update()

        # 1. Ottieni bbox dell’immagine + coordinate nel canvas
        bbox = self.canvas.bbox(self.canvas_image)  # (x1, y1, x2, y2)
        if bbox is None:
            self.status_label.config(text="Error: no image to save.")
            return

        # 2. Espandi il bbox per includere ticks, freccia nord e scalebar
        top_margin = 120
        side_margin = 120
        bottom_margin = 60  # ridotto per evitare bordo nero
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 -side_margin)
        y1 = max(0, y1 - top_margin)
        # Espandi a destra di più se la scalebar è lunga
        scalebar_extra = 0
        if hasattr(self, "display_w") and hasattr(self, "zoom_factor"):
            pixel_size = abs(self.parent.geotransform.a)
            target_length_m = 100  # oppure 50 m se più adatto
            scale_px = int(target_length_m / pixel_size * self.zoom_factor)
            scalebar_extra = scale_px // 2 + 100

        x2 = x2 + side_margin + scalebar_extra
        y2 = y2 + bottom_margin

        # 3. Cattura intero canvas in PostScript con larghezza/altezza pagina aumentata
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        scale = 3  # Fattore di scala per migliorare qualità testo/grafica
        ps_data = self.canvas.postscript(
            colormode='color',
            x=0, y=0, width=w, height=h,
            pagewidth=w * scale, pageheight=h * scale
        )

        try:
            img = Image.open(io.BytesIO(ps_data.encode("utf-8")))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to capture canvas: {e}")
            return

        # 4. Ritaglia immagine alla bounding box utile
        cropped = img.crop((x1 * scale, y1 * scale, x2 * scale, y2 * scale))

        # 5. Salva immagine finale con risoluzione elevata
        file_path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            cropped.save(file_path, dpi=(300, 300))
            self.status_label.config(text=f"Plot saved to {file_path}")

    # ====== EXPORT HIGHLIGHTS AS SHAPEFILE ======
    # --- EXPORT HIGHLIGHTS AS SHAPEFILE ---

    def export_highlights_as_shapefile(self, path, max_gap=3, geometry_type="line"):
        if not hasattr(self.parent, "geotransform") or self.parent.geotransform is None:
            tk.messagebox.showerror("Export Error", "No geotransform found. The image must be georeferenced.")
            return

        transform = self.parent.geotransform
        crs = self.parent.crs

        image_array = np.array(self.current_img.convert("RGB"))
        clean_array = np.array(self.clean_img.convert("RGB"))

        # Crea maschera binaria degli highlight
        diff_mask = np.any(image_array != clean_array, axis=2)

        # Estrai coordinate dei pixel evidenziati
        coords = np.argwhere(diff_mask)  # (row, col)

        if coords.size == 0:
            tk.messagebox.showinfo("Export Highlights", "No highlights to export.")
            return

        coords_sorted = sorted([tuple(coord) for coord in coords], key=lambda x: (x[0], x[1]))

        import shapely.geometry
        import geopandas as gpd
        from rasterio.transform import xy

        if geometry_type == "point":
            # Export each highlight pixel as a Point
            features = [shapely.geometry.Point(*xy(transform, r, c)) for r, c in coords_sorted]
            gdf = gpd.GeoDataFrame(geometry=features, crs=crs)
            gdf.to_file(path)
            tk.messagebox.showinfo("Export Highlights", f"Shapefile exported with {len(gdf)} points.")

        elif geometry_type == "line":
            # Group pixels into lines using max_gap
            lines = []
            current_line = [coords_sorted[0]]
            for i in range(1, len(coords_sorted)):
                r1, c1 = coords_sorted[i - 1]
                r2, c2 = coords_sorted[i]
                if abs(r2 - r1) <= max_gap and abs(c2 - c1) <= max_gap:
                    current_line.append((r2, c2))
                else:
                    if len(current_line) > 1:
                        lines.append(current_line)
                    current_line = [(r2, c2)]
            if len(current_line) > 1:
                lines.append(current_line)

            geo_lines = [shapely.geometry.LineString([xy(transform, r, c) for r, c in line]) for line in lines]
            gdf = gpd.GeoDataFrame(geometry=geo_lines, crs=crs)
            gdf.to_file(path)
            tk.messagebox.showinfo("Export Highlights", f"Shapefile exported with {len(gdf)} lines.")

        elif geometry_type == "polygon":
            # Trova contorni chiusi nella maschera binaria per creare poligoni


            contours = measure.find_contours(diff_mask.astype(float), level=0.5)
            polygons = []

            for contour in contours:
                coords_poly = []
                for y_img, x_img in contour:
                    x_geo = transform.c + x_img * transform.a
                    y_geo = transform.f + y_img * transform.e
                    coords_poly.append((x_geo, y_geo))
                if len(coords_poly) >= 3:
                    poly = shapely.geometry.Polygon(coords_poly)
                    if poly.is_valid:
                        polygons.append(poly)

            if not polygons:
                tk.messagebox.showwarning("Export Error", "Could not extract valid polygons from highlights.")
                return

            gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
            gdf.to_file(path)
            tk.messagebox.showinfo("Export Highlights", f"Shapefile exported with {len(gdf)} polygons.")

    def _on_export_shapefile(self):
        # Use PCA-style geometry selection popup
        if not hasattr(self, "geotransform") or not hasattr(self, "crs"):
            self.geotransform = getattr(self.parent, "geotransform", None)
            self.crs = getattr(self.parent, "crs", None)
        if not self.geotransform or not self.crs:
            messagebox.showwarning("Export Shapefile", "The image is not georeferenced.")
            return

        def on_geometry_choice(choice):
            popup.destroy()
            path = filedialog.asksaveasfilename(defaultextension=".shp", filetypes=[("Shapefiles", "*.shp")])
            if path:
                try:
                    self.export_highlights_as_shapefile(path, max_gap=int(self.gap_entry.get()), geometry_type=choice)
                    messagebox.showinfo("Export successful", f"Shapefile saved at:\n{path}")
                except Exception as e:
                    messagebox.showerror("Export failed", f"An error occurred:\n{e}")

        popup = tk.Toplevel(self)
        popup.title("Select geometry type")
        tk.Label(popup, text="Choose export geometry:").pack(padx=10, pady=10)
        tk.Button(popup, text="Points", command=lambda: on_geometry_choice("point")).pack(padx=10, pady=5)
        tk.Button(popup, text="Lines", command=lambda: on_geometry_choice("line")).pack(padx=10, pady=5)
        tk.Button(popup, text="Polygons", command=lambda: on_geometry_choice("polygon")).pack(padx=10, pady=5)

    def erase_by_selection(self):
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<Button-1>")
        self.selection_in_progress = True
        self.polygon_points = []
        self.poly_line = None
        self.canvas.delete("polygon_preview")
        self.canvas.config(cursor="cross")
        self.status_label.config(text="Click to define a polygon. Double-click to confirm and erase anomalies inside.")
        self.canvas.bind("<ButtonPress-1>", self.on_polygon_click)
        self.canvas.bind("<Double-Button-1>", self.on_polygon_finish)
        self.erase_polygon_mask = None

    def on_polygon_click(self, event):
        if not self.selection_in_progress:
            self.selection_in_progress = True
            self.polygon_points = []
            self.poly_line = None
            self.canvas.delete("polygon_preview")
            self.status_label.config(text="Select area: click to add points, double-click to confirm.")
            self.erase_polygon_mask = None

        self.polygon_points.append((event.x, event.y))

        radius = 3
        self.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius,
                                fill='red', tags="polygon_preview")

        if self.poly_line:
            self.canvas.delete(self.poly_line)

        if len(self.polygon_points) > 1:
            self.poly_line = self.canvas.create_line(
                *sum(self.polygon_points, ()),
                fill='red', tags="polygon_preview"
            )

    def on_polygon_finish(self, event):
        if len(self.polygon_points) < 3:
            self.status_label.config(text="Need at least 3 points to form a polygon.")
            return

        self.canvas.create_polygon(self.polygon_points, outline='red', fill='', width=2, tags="polygon_preview")
        self.canvas.config(cursor="")
        self.selection_in_progress = False

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_w, img_h = self.current_img.size

        center_x = (canvas_w - self.display_w) // 2
        center_y = (canvas_h - self.display_h) // 2
        disp_x_offset = center_x + self.x_offset
        disp_y_offset = center_y + self.y_offset

        converted_points = []
        for x_canvas, y_canvas in self.polygon_points:
            x_in_disp = x_canvas - disp_x_offset
            y_in_disp = y_canvas - disp_y_offset
            x_orig = x_in_disp / self.zoom_factor
            y_orig = y_in_disp / self.zoom_factor
            converted_points.append((x_orig, y_orig))

        mask = self._create_erase_polygon_mask((img_h, img_w), converted_points)
        self.erase_polygon_mask = mask

        # ===== Applica erase direttamente =====
        clean_resized = self.clean_img.resize(self.current_img.size, Image.LANCZOS)
        current_np = np.array(self.current_img)
        clean_np = np.array(clean_resized)
        for c in range(4):
            current_np[..., c][mask] = clean_np[..., c][mask]
        self.current_img = Image.fromarray(current_np)
        self.display_image()

        # Aggiunge stato a history
        if self.history_index == len(self.history) - 1:
            self.history.append(self.current_img.copy())
        else:
            self.history = self.history[:self.history_index + 1]
            self.history.append(self.current_img.copy())
        self.history_index += 1
        self.update_undo_redo_buttons()

        self.status_label.config(text="Polygon selection applied and anomalies erased.")

    def _create_erase_polygon_mask(self, shape, polygon_points):
        from matplotlib.path import Path
        y, x = np.mgrid[:shape[0], :shape[1]]
        coords = np.stack((x.ravel(), y.ravel()), axis=-1)
        path = Path(polygon_points)
        mask = path.contains_points(coords).reshape(shape)
        return mask

# === FUNZIONI OUTLIER ESTERNE ===

def remove_outliers_standard(image_array, clean_image, percentile_threshold=90, mask=None):
    """
    Rimuove gli outlier in base alla differenza sui canali RGB tra immagine e clean image.
    La differenza è calcolata come somma assoluta dei 3 canali.
    """
    diff_rgb = np.abs(image_array[..., :3].astype(np.float32) - clean_image[..., :3].astype(np.float32))
    diff_sum = np.sum(diff_rgb, axis=2)

    threshold = np.percentile(diff_sum, percentile_threshold)
    outlier_mask = diff_sum > threshold

    if mask is not None:
        outlier_mask = outlier_mask & mask

    filtered = np.copy(image_array)
    for c in range(4):
        filtered[..., c][outlier_mask] = clean_image[..., c][outlier_mask]

    return filtered

def remove_outliers_zscore(image_array, clean_image, z_threshold=2.0, mask=None):
    """
    Rimuove outlier in base allo Z-score calcolato sui canali RGB.
    """
    rgb = image_array[..., :3].astype(np.float32)
    mean = np.mean(rgb)
    std = np.std(rgb)
    z_scores = np.abs((rgb - mean) / (std + 1e-8))
    z_sum = np.sum(z_scores, axis=2)
    outlier_mask = z_sum > z_threshold * 3
    if mask is not None:
        outlier_mask = outlier_mask & mask

    filtered = np.copy(image_array)
    for c in range(4):
        filtered[..., c][outlier_mask] = clean_image[..., c][outlier_mask]
    return filtered

def remove_outliers_local_contrast(image_array, clean_image, window_size=3, mask=None):
    """
    Rimuove outlier in base al contrasto locale (finestra NxN) sui canali RGB.
    """
    def local_contrast_filter(values):
        center = values[len(values) // 2]
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0
        z = np.abs(center - mean) / std
        return 1 if z > 2 else 0

    rgb_gray = np.mean(image_array[..., :3], axis=2).astype(np.float32)
    outlier_mask = generic_filter(rgb_gray, local_contrast_filter, size=window_size, mode='reflect') > 0
    if mask is not None:
        outlier_mask = outlier_mask & mask
    filtered = np.copy(image_array)
    for c in range(4):
        filtered[..., c][outlier_mask] = clean_image[..., c][outlier_mask]
    return filtered


