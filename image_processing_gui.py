import numpy as np
import tkinter as tk
import os
from pyproj import CRS, Transformer
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox
from image_processing import find_anomalies, highlight_anomalies, create_mask, composite_anomaly_overlay
from rasterio.transform import xy
from pca_editor import PCAEditor
from sklearn.decomposition import PCA

class ImageProcessingGUI:
    def __init__(self, parent):
        self.parent = parent
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.scale_factor = 1.0
        self.base_display_image = None  # PIL.Image resized-to-fit used as zoom baseline
        self.parent.tick_ids = []  # Per memorizzare gli ID dei tick e delle etichette
        self.parent.scalebar_ids = []  # Per memorizzare gli ID della barra di scala
        self.selection_mask = None
        self.selection_bbox = None

    def display_image_on_canvas(self, image_array):
        # Evita di visualizzare immagini multispettrali (verrà gestita da pseudoRGB logic)
        if isinstance(image_array, np.ndarray) and image_array.ndim == 3 and image_array.shape[2] > 3:
            print("[DEBUG] display_image_on_canvas: Skipping multispectral (>3 bands) image.")
            return
        # Costanti per margini
        tick_label_margin = 50  # Margine per etichette dei tick
        north_arrow_margin = 40  # Margine per la freccia del Nord
        scalebar_margin = 50  # Margine per la barra di scala
        border_thickness = 2  # Spessore del bordo nero

        # Log di debug sull'input
        if isinstance(image_array, np.ndarray):
            # Check to avoid redundant redraws
            if hasattr(self.parent, "displayed_image"):
                if np.array_equal(image_array, self.parent.displayed_image) and not self.parent.zoomed_selection_coords:
                    print("[DEBUG] display_image_on_canvas: Skipping redraw, image already displayed.")
                    return
            self.parent.displayed_image = np.copy(image_array)
            print(f"[DEBUG] display_image_on_canvas: input shape={image_array.shape}, "
                  f"dtype={image_array.dtype}, min={image_array.min()}, max={image_array.max()}")
        else:
            print("[DEBUG] display_image_on_canvas: input is not a NumPy array.")

        # Converti l’immagine in PIL per la visualizzazione
        if image_array.ndim == 2:
            pil_img = Image.fromarray(image_array.astype('uint8'), 'L')
            print("[DEBUG] display_image_on_canvas: 2D array => grayscale mode (L)")
        else:
            c = image_array.shape[2]
            print(f"[DEBUG] display_image_on_canvas: channels={c}")
            if c == 1:
                gray = image_array[:, :, 0].astype('uint8')
                pil_img = Image.fromarray(gray, 'L')
            elif c == 3:
                pil_img = Image.fromarray(image_array.astype('uint8'), 'RGB')
            elif c == 4:
                pil_img = Image.fromarray(image_array.astype('uint8'), 'RGBA')
            elif c >= 5:
                print("[DEBUG] display_image_on_canvas: 5+ channels => pseudoRGB logic or spectral logic.")
                try:
                    import spectral
                    # Seleziona le bande per pseudo-RGB; ad esempio banda 3, banda 2, banda 1
                    pseudo_rgb = image_array[..., [2, 1, 0]].copy()
                except ImportError:
                    pseudo_rgb = image_array[..., :3].copy()

                # Se l'array è di tipo floating point, applica lo stretch (sostituisci i valori di nodata con NaN, ecc.)
                if np.issubdtype(image_array.dtype, np.floating):
                    pseudo_rgb[pseudo_rgb < -9999] = np.nan
                    low = np.nanmin(pseudo_rgb)
                    high = np.nanmax(pseudo_rgb)
                    print(f"[DEBUG] pseudoRGB stretch range: {low} .. {high}")
                    if high - low < 1e-6:
                        arr_f = np.zeros_like(pseudo_rgb)
                    else:
                        arr_f = (pseudo_rgb - low) / (high - low) * 255.0
                    arr_f = np.nan_to_num(arr_f, nan=0)
                    arr_f = np.clip(arr_f, 0, 255).astype('uint8')
                    pil_img = Image.fromarray(arr_f, 'RGB')
                else:
                    # Se l'array non è floating, ad esempio già uint8, usa direttamente pseudo_rgb
                    pil_img = Image.fromarray(pseudo_rgb, 'RGB')
            print(f"[DEBUG] PIL image size={pil_img.size}, mode={pil_img.mode}")

        # Salva l'immagine originale (PIL) SOLO se non esiste già
        if not hasattr(self.parent, "original_image_pil") or self.parent.original_image_pil is None:
            self.parent.original_image_pil = pil_img.copy()
        self.parent.scale_factor = 1.0
        # Reset zoom baseline/state every time we render a fresh image on the main canvas
        self.scale_factor = 1.0

        canvas_width = self.parent.canvas.winfo_width()
        canvas_height = self.parent.canvas.winfo_height()

        # Margini aggiuntivi per tick, barra di scala e Nord
        total_margin_x = tick_label_margin * 2
        total_margin_y = tick_label_margin + north_arrow_margin + scalebar_margin

        # Dimensioni del canvas considerando i margini
        effective_canvas_width = canvas_width - total_margin_x
        effective_canvas_height = canvas_height - total_margin_y

        img_width, img_height = pil_img.size

        # Ridimensiona l'immagine considerando i margini
        if img_width > effective_canvas_width or img_height > effective_canvas_height:
            scale_factor = min(effective_canvas_width / img_width, effective_canvas_height / img_height)
            new_size = (int(img_width * scale_factor), int(img_height * scale_factor))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            self.parent.canvas.update_idletasks()
            img_width, img_height = new_size
        else:
            scale_factor = 1.0

        self.parent.effective_scale_factor = scale_factor

        # Baseline image for zoom/pan: the exact resized-to-fit image currently displayed
        self.base_display_image = pil_img.copy()

        # Crea l'oggetto Tkinter PhotoImage
        self.parent.tk_image = ImageTk.PhotoImage(pil_img)

        # Svuota il canvas e ridisegna
        self.parent.canvas.delete("all")

        # Forza aggiornamento della geometria del canvas e calcola il centro correttamente
        self.parent.canvas.update_idletasks()
        canvas_width = self.parent.canvas.winfo_width()
        canvas_height = self.parent.canvas.winfo_height()

        x_offset = max((canvas_width - img_width) // 2, 0)
        y_offset = max((canvas_height - img_height) // 2, 0)

        self.parent.x_offset = x_offset
        self.parent.y_offset = y_offset

        # Mostra l'immagine sul canvas
        self.canvas_image = self.parent.canvas.create_image(x_offset, y_offset, anchor='nw', image=self.parent.tk_image)
        self.parent.canvas.config(scrollregion=self.parent.canvas.bbox("all"))

        self.parent.display_w = img_width
        self.parent.display_h = img_height

        print(f"[DEBUG] display_image_on_canvas: final display_w={self.parent.display_w}, "
              f"display_h={self.parent.display_h}, x_offset={self.parent.x_offset}, y_offset={self.parent.y_offset}")

        if self.parent.logo_id:
            self.parent.canvas.delete(self.parent.logo_id)
        self.parent.logo_id = None

        self.reset_pan()

        # Aggiungi il bordo nero intorno all'immagine
        self.parent.canvas.create_rectangle(
            x_offset - border_thickness, y_offset - border_thickness,
            x_offset + img_width + border_thickness, y_offset + img_height + border_thickness,
            outline="black", width=border_thickness
        )

        # (1) Imposta current_image SEMPRE (così lo zoom/pan funziona anche su non-geotiff)
        self.parent.current_image = self.base_display_image.copy() if self.base_display_image is not None else pil_img.copy()

        # --- Remove legacy offset computation ---
        # self.parent.x_offset = x_offset - (canvas_width - img_width) // 2
        # self.parent.y_offset = y_offset - (canvas_height - img_height) // 2

        def debug_log(message):
            print(f"[DEBUG] {message}")

        # (2) Se l'immagine NON ha CRS e geotransform, salta freccia Nord, ticks, scalebar
        if self.parent.crs is None or self.parent.geotransform is None:
            debug_log("No CRS or geotransform. Skipping arrow, ticks and scale.")
            return

        # Da qui in poi SOLO se c’è georeferenziazione
        current_crs = CRS(self.parent.crs)
        debug_log(f"Image CRS: {current_crs}")

        # Freccia del nord personalizzata
        north_x = x_offset + img_width + north_arrow_margin
        north_y = y_offset + 60  # Manteniamo la freccia sotto il bordo superiore

        # Freccia del nord migliorata
        self.parent.canvas.create_polygon(
            [
                (north_x, north_y),  # Punta
                (north_x - 10, north_y + 30),  # Angolo sinistro
                (north_x + 10, north_y + 30)  # Angolo destro
            ],
            fill="black", outline="black"
        )
        self.parent.canvas.create_text(
            north_x, north_y - 15, text="N", font=("Arial", 14, "bold"), fill="black"
        )

        # Determina la risoluzione del pixel in metri
        scale_factor = scale_factor  # già definito
        if current_crs.is_geographic:
            debug_log("CRS is geographic. Converting to UTM for accurate measurements.")
            utm_crs = current_crs.to_utm()
            transformer_to_utm = Transformer.from_crs(current_crs, utm_crs, always_xy=True)
            pixel_width_degrees = abs(self.parent.geotransform.a)
            lon1, lat1 = self.parent.geotransform.c, self.parent.geotransform.f
            lon2, lat2 = lon1 + pixel_width_degrees, lat1
            x1, y1 = transformer_to_utm.transform(lon1, lat1)
            x2, y2 = transformer_to_utm.transform(lon2, lat2)
            pixel_width_m = abs(x2 - x1)
            pixel_width_m_displayed = pixel_width_m / scale_factor
            debug_log(f"Pixel Width (m, UTM): {pixel_width_m}")
        else:
            debug_log("CRS is projected. Using pixel width directly.")
            pixel_width_m = abs(self.parent.geotransform.a)
            pixel_width_m_displayed = pixel_width_m / scale_factor

        from rasterio.transform import xy
        transformer_to_geographic = Transformer.from_crs(self.parent.crs, "EPSG:4326", always_xy=True)

        # Ottieni i 4 angoli dell'immagine in pixel originali
        img_w, img_h = self.parent.original_image_pil.size

        # Angoli in pixel (riga, colonna)
        corners = [
            (0, 0),  # top-left
            (0, img_w),  # top-right
            (img_h, 0),  # bottom-left
            (img_h, img_w)  # bottom-right
        ]

        # Trasforma gli angoli in lat/lon
        lats = []
        lons = []
        for row_c, col_c in corners:
            x_c, y_c = xy(self.parent.geotransform, row_c, col_c)
            lon_c, lat_c = transformer_to_geographic.transform(x_c, y_c)
            lats.append(lat_c)
            lons.append(lon_c)

        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        debug_log(f"Image Lat/Lon bounds: lat_min={lat_min}, lat_max={lat_max}, lon_min={lon_min}, lon_max={lon_max}")

        # Tick lat/lon
        lat_ticks = [lat_min, lat_max]
        lon_ticks = [lon_min, (lon_min + lon_max) / 2, lon_max]

        def latlon_to_canvas(lat, lon):
            x_g, y_g = Transformer.from_crs("EPSG:4326", self.parent.crs, always_xy=True).transform(lon, lat)
            col = (x_g - self.parent.geotransform.c) / self.parent.geotransform.a
            row = (y_g - self.parent.geotransform.f) / self.parent.geotransform.e
            x_canvas = x_offset + (col * (img_width / img_w))
            y_canvas = y_offset + (row * (img_height / img_h))
            debug_log(
                f"lat: {lat}, lon: {lon} -> x_g: {x_g}, y_g: {y_g}, col: {col}, row: {row}, canvas: ({x_canvas}, {y_canvas})"
            )
            return x_canvas, y_canvas

        def to_dms(value, is_lat=False):
            degrees = int(value)
            minutes = int((abs(value) - abs(degrees)) * 60)
            seconds = (abs(value) - abs(degrees)) * 3600 - minutes * 60
            hemi = ''
            if is_lat:
                hemi = 'N' if value >= 0 else 'S'
            else:
                hemi = 'E' if value >= 0 else 'W'
            return f"{abs(degrees)}°{minutes}'{seconds:.1f}\"{hemi}"

        tick_len = 10

        # Cancella gli ID precedenti
        for tick_id in self.parent.tick_ids:
            self.parent.canvas.delete(tick_id)
        self.parent.tick_ids.clear()

        # Cancella gli ID della barra di scala precedente
        for scalebar_id in self.parent.scalebar_ids:
            self.parent.canvas.delete(scalebar_id)
        self.parent.scalebar_ids.clear()

        # Disegno i tick LAT (lati sinistro/destro)
        for lat_val in lat_ticks:
            x_canvas_mid, y_canvas_mid = latlon_to_canvas(lat_val, (lon_min + lon_max) / 2)
            lat_dms = to_dms(lat_val, is_lat=True)

            # Lato sinistro
            tick_id = self.parent.canvas.create_line(x_offset, y_canvas_mid, x_offset - tick_len, y_canvas_mid,
                                                     fill="black")
            self.parent.tick_ids.append(tick_id)
            label_id = self.parent.canvas.create_text(x_offset - tick_len - 40, y_canvas_mid,
                                                      text=lat_dms, fill="black",font=("Arial", 15, "bold"))
            self.parent.tick_ids.append(label_id)

            # Lato destro
            tick_id = self.parent.canvas.create_line(x_offset + img_width, y_canvas_mid,
                                                     x_offset + img_width + tick_len, y_canvas_mid,
                                                     fill="black")
            self.parent.tick_ids.append(tick_id)
            label_id = self.parent.canvas.create_text(x_offset + img_width + tick_len + 40, y_canvas_mid,
                                                      text=lat_dms, fill="black",font=("Arial", 15, "bold"))
            self.parent.tick_ids.append(label_id)

        # Disegno i tick LON (lati superiore/inferiore)
        for lon_val in lon_ticks:
            x_canvas_mid, y_canvas_mid = latlon_to_canvas((lat_min + lat_max) / 2, lon_val)
            lon_dms = to_dms(lon_val, is_lat=False)

            # Lato superiore
            tick_id = self.parent.canvas.create_line(x_canvas_mid, y_offset, x_canvas_mid, y_offset - tick_len,
                                                     fill="black")
            self.parent.tick_ids.append(tick_id)
            label_id = self.parent.canvas.create_text(x_canvas_mid, y_offset - tick_len - 10,
                                                      text=lon_dms, fill="black",font=("Arial", 15, "bold"))
            self.parent.tick_ids.append(label_id)

            # Lato inferiore
            tick_id = self.parent.canvas.create_line(x_canvas_mid, y_offset + img_height, x_canvas_mid,
                                                     y_offset + img_height + tick_len, fill="black")
            self.parent.tick_ids.append(tick_id)
            label_id = self.parent.canvas.create_text(x_canvas_mid, y_offset + img_height + tick_len + 10,
                                                      text=lon_dms, fill="black",font=("Arial", 15, "bold"))
            self.parent.tick_ids.append(label_id)

        # Barra di scala
        gui_width = self.parent.winfo_width()
        scale_length_px_gui = int((gui_width / 20) * 1.5)
        proposed_length_m = scale_length_px_gui * pixel_width_m_displayed
        debug_log(f"Proposed Scale Length (m): {proposed_length_m}")

        base_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
        chosen_scale_m = min(base_values, key=lambda x: abs(x - proposed_length_m))
        debug_log(f"Chosen Scale Value: {chosen_scale_m} m")

        scalebar_length_px = int(chosen_scale_m / pixel_width_m_displayed)
        debug_log(f"Scale Bar Length (px): {scalebar_length_px}, Value (m): {chosen_scale_m}")

        scalebar_x = x_offset + self.parent.display_w + 20
        scalebar_y = y_offset + self.parent.display_h - 50
        self.parent.scalebar_x = scalebar_x
        self.parent.scalebar_y = scalebar_y

        debug_log(f"Scalebar position: ({scalebar_x}, {scalebar_y})")

        segment_count = 5
        segment_px = scalebar_length_px // segment_count

        # Segmenti della barra di scala
        for seg in range(segment_count):
            seg_x1 = scalebar_x + seg * segment_px
            seg_x2 = seg_x1 + segment_px
            fill_color = "black" if seg % 2 == 0 else "white"
            segment_id = self.parent.canvas.create_rectangle(seg_x1, scalebar_y, seg_x2, scalebar_y + 10,
                                                             fill=fill_color,
                                                             outline="black")
            self.parent.scalebar_ids.append(segment_id)

        # Etichette della barra di scala
        scalebar_label_id_0 = self.parent.canvas.create_text(scalebar_x, scalebar_y + 20, text="0", fill="black", font=("Arial", 15, "bold"))
        self.parent.scalebar_ids.append(scalebar_label_id_0)

        if chosen_scale_m < 1000:
            scalebar_label_id_end = self.parent.canvas.create_text(scalebar_x + scalebar_length_px,
                                                                   scalebar_y + 20,
                                                                   text=f"{chosen_scale_m} m", fill="black", font=("Arial", 15, "bold"))
        else:
            km_value = chosen_scale_m / 1000.0
            scalebar_label_id_end = self.parent.canvas.create_text(scalebar_x + scalebar_length_px,
                                                                   scalebar_y + 20,
                                                                   text=f"{km_value:.1f} km", fill="black", font=("Arial", 15, "bold"))
        self.parent.scalebar_ids.append(scalebar_label_id_end)

    def update_anomalies(self, *args):
        threshold = self.parent.threshold.get()
        color = self.parent.color.get()

        img_width, img_height = self.parent.original_image_pil.size  # PIL.Image.size restituisce (width, height)
        canvas_width = self.parent.canvas.winfo_width()
        canvas_height = self.parent.canvas.winfo_height()

        # Converte l'immagine PIL in NumPy per l'analisi
        base_image = np.array(
            self.parent.analyzed_image if self.parent.analyzed_image is not None else self.parent.image
        )

        if base_image.ndim == 2:  # Immagine in scala di grigi
            base_image = base_image[..., np.newaxis]

        if base_image.shape[2] == 1:
            base_image = np.repeat(base_image, 3, axis=2)

        # Dimensioni per create_mask (width, height)
        mask_size = (img_height, img_width)

        if self.parent.image_selection.polygon_points:
            normalized_polygon_points = [
                (x / img_width, y / img_height)
                for (x, y) in self.parent.image_selection.polygon_points
            ]

            mask = create_mask(mask_size, normalized_polygon_points, normalize=True)
            masked_image = base_image.copy()
            masked_image[mask == 0] = 0

            anomalies = find_anomalies(masked_image, threshold, self.parent.anomaly_type.get(),
                                       self.parent.method.get())

            highlighted_image = base_image.copy()
            if self.parent.method.get() == "Standard":
                temp_highlight = highlight_anomalies(base_image, anomalies, color)
                highlighted_image[mask == 1] = temp_highlight[mask == 1]
            else:
                temp_highlight = highlight_anomalies(base_image, anomalies, color)
                highlighted_image = temp_highlight
                highlighted_image[mask == 0] = base_image[mask == 0]

            self.parent.image = highlighted_image
            self.display_image_on_canvas(self.parent.image)

        elif self.parent.image_selection.rect_coords:
            left, top, right, bottom = self.parent.image_selection.rect_coords

            left = max(int(left), 0)
            right = min(int(right), img_width)
            top = max(int(top), 0)
            bottom = min(int(bottom), img_height)

            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            mask[top:bottom, left:right] = 1

            masked_image = base_image.copy()
            masked_image[mask == 0] = 0

            anomalies = find_anomalies(masked_image, threshold, self.parent.anomaly_type.get(),
                                       self.parent.method.get())

            highlighted_image = base_image.copy()
            temp_highlight = highlight_anomalies(base_image, anomalies, color)
            highlighted_image[mask == 1] = temp_highlight[mask == 1]

            self.parent.image = highlighted_image
            self.display_image_on_canvas(self.parent.image)

        else:
            anomalies = find_anomalies(base_image, threshold, self.parent.anomaly_type.get(), self.parent.method.get())
            highlighted_image = highlight_anomalies(base_image, anomalies, color)
            self.parent.image = highlighted_image
            self.display_image_on_canvas(self.parent.image)

        self.parent.update_status("Threshold adjusted")

    def update_pca_highlight(self, event=None):
        if not hasattr(self.parent, "pca_result") or self.parent.pca_result is None:
            return
        try:
            idx = int(self.parent.pca_component_selector.get()[2:]) - 1
            h, w = self.parent.pca_shape
            comp = self.parent.pca_result[:, idx].reshape(h, w)
            color = self.parent.color.get()

            # Applica la maschera se presente
            if hasattr(self.parent, "pca_mask") and self.parent.pca_mask is not None:
                valid_pixels = self.parent.pca_mask
            else:
                valid_pixels = ~np.isnan(comp)

            if np.sum(valid_pixels) == 0:
                tk.messagebox.showerror("PCA", "No valid data in selection area for PCA.")
                return

            comp_clean = comp[valid_pixels]
            threshold = np.percentile(comp_clean, 98)
            anomalies = comp > threshold

            composite = composite_anomaly_overlay(self.parent.pca_input_image, anomalies, color, opacity=200)
            self.parent.image = composite
            self.display_image_on_canvas(self.parent.image)
            self.parent.analyzed_image = np.copy(self.parent.image)
            self.parent.update_status(f"{self.parent.pca_component_selector.get()} shown with highlights")
        except Exception as e:
            print("[DEBUG] Errore in update_pca_highlight:", e)

    def analyze_image(self):
        self.parent.hide_center_logo()
        self.update_threshold_slider()

        threshold = self.parent.threshold.get()
        color = self.parent.color.get()
        eps = self.parent.eps_slider.get()
        min_samples = int(self.parent.min_samples_slider.get())

        if hasattr(self.parent, 'image_histo_processing') and self.parent.image_histo_processing is not None:
            image_array = np.copy(self.parent.image_histo_processing)
        else:
            image_array = np.copy(self.parent.image)
            if image_array.ndim == 2:
                image_array = image_array[..., np.newaxis]
            if image_array.shape[2] == 3 and np.array_equal(image_array, self.parent.original_image_np[..., :3]):
                image_array = np.copy(self.parent.original_image_np)

        # === Caso LISA ===
        if self.parent.method.get() == "LISA":
            selection_mask = None
            img_width, img_height = self.parent.original_image_pil.size
            mask_size = (img_height, img_width)

            if self.parent.image_selection.polygon_points:
                normalized_points = [(x / img_width, y / img_height)
                                     for (x, y) in self.parent.image_selection.polygon_points]
                selection_mask = create_mask(mask_size, normalized_points, normalize=True)
            elif self.parent.image_selection.rect_coords:
                left, top, right, bottom = self.parent.image_selection.rect_coords
                selection_mask = np.zeros(mask_size, dtype=np.uint8)
                selection_mask[top:bottom, left:right] = 1

            self.parent.open_lisa_config_window(selection_mask=selection_mask)
            return

        # === Caso PCA ===
        elif self.parent.method.get() == "PCA":
            def on_pca_selection(mode):
                image_copy = np.copy(self.parent.original_image_np)

                # Rimuove canale singolo se presente
                if image_copy.ndim == 3 and image_copy.shape[2] == 1:
                    image_copy = image_copy[:, :, 0]

                # Estrai dimensioni in modo sicuro
                if image_copy.ndim == 2:
                    h, w = image_copy.shape
                    c = 1
                else:
                    h, w, c = image_copy.shape

                # Calcolo maschera se esiste selezione
                selection_mask = None
                if self.parent.image_selection.polygon_points:
                    normalized_points = [(x / w, y / h)
                                         for (x, y) in self.parent.image_selection.polygon_points]
                    selection_mask = create_mask((h, w), normalized_points, normalize=True).astype(bool)
                elif self.parent.image_selection.rect_coords:
                    left, top, right, bottom = self.parent.image_selection.rect_coords
                    selection_mask = np.zeros((h, w), dtype=bool)
                    selection_mask[top:bottom, left:right] = True

                # Applica la maschera per calcolare la PCA solo sull’area selezionata
                if selection_mask is not None:
                    flat_mask = selection_mask.flatten()
                    reshaped = image_copy.reshape((-1, c))
                    masked_reshaped = reshaped[flat_mask]
                else:
                    reshaped = image_copy.reshape((-1, c))
                    masked_reshaped = reshaped

                if mode == "PC1-2":
                    n_components = 2
                else:
                    max_components = min(masked_reshaped.shape[0], masked_reshaped.shape[1])
                    n_components = max_components
                    print(
                        f"[DEBUG] PCA mode=ALL, available_pixels={masked_reshaped.shape[0]}, channels={masked_reshaped.shape[1]}")
                    print(f"[DEBUG] Will compute {n_components} PCA components")

                pca = PCA(n_components=n_components)
                pca_result = np.full((h * w, n_components), np.nan, dtype=np.float32)

                # Rimuove eventuali righe con NaN
                valid_rows = ~np.isnan(masked_reshaped).any(axis=1)
                masked_reshaped_clean = masked_reshaped[valid_rows]

                if masked_reshaped_clean.shape[0] == 0:
                    tk.messagebox.showerror("PCA Error", "No valid pixels available (all NaN).")
                    return

                # Esegui PCA solo sui pixel selezionati e validi
                pca_values = pca.fit_transform(masked_reshaped_clean)
                if selection_mask is not None:
                    valid_flat_mask = flat_mask.copy()
                    valid_flat_mask[flat_mask] = valid_rows
                    pca_result[valid_flat_mask] = pca_values
                else:
                    pca_result = pca_values

                explained_variance_ratio = pca.explained_variance_ratio_


                # Salva risultati per il viewer
                self.parent.pca_result = pca_result
                self.parent.pca_shape = (h, w)
                self.parent.pca_mode_used = mode
                self.parent.pca_input_image = image_copy
                self.parent.pca_mask = selection_mask  # salva la maschera per uso successivo
                self.parent.explained_variance_ratio = explained_variance_ratio

                # Visualizza PC1 con highlight nel canvas
                pc1_img = pca_result[:, 0].reshape(h, w)
                valid_pixels = ~np.isnan(pc1_img)
                if np.sum(valid_pixels) == 0:
                    tk.messagebox.showerror("PCA", "No valid data in selection area for PCA.")
                    return

                pc1_clean = pc1_img[valid_pixels]
                pc1_thresh = np.percentile(pc1_clean, 98)
                anomalies = pc1_img > pc1_thresh

                self.parent.analyzed_clean_image = np.copy(image_array)
                composite = composite_anomaly_overlay(image_copy, anomalies, color, opacity=200)
                self.parent.image = composite
                self.display_image_on_canvas(composite)

                self.parent.analyzed_image = np.copy(composite)
                self.parent.anomaly_editor_button.config(state=tk.NORMAL)
                self.parent.pca_viewer_button.config(state=tk.NORMAL)
                self.parent.save_button.config(state=tk.NORMAL)
                self.parent.undo_button.config(state=tk.NORMAL)
                self.parent.redo_button.config(state=tk.DISABLED)
                self.parent.update_status("PCA1 shown with highlights")

                self.parent.pca_selector_label.pack(side=tk.LEFT, padx=(10, 0))
                self.parent.pca_component_selector.pack(side=tk.LEFT, padx=5)
                self.parent.pca_component_selector.config(state=tk.NORMAL)
                self.parent.pca_component_selector['values'] = [f"PC{i + 1}" for i in range(n_components)]
                self.parent.pca_component_selector.current(0)

            # Popup per selezione modalità PCA
            popup = tk.Toplevel(self.parent)
            popup.title("PCA Mode")
            tk.Label(popup, text="Select PCA mode:").pack(padx=10, pady=10)
            tk.Button(popup, text="Only PC1 & PC2", command=lambda: [popup.destroy(), on_pca_selection("PC1-2")]).pack(
                padx=10, pady=5)
            tk.Button(popup, text="All Components", command=lambda: [popup.destroy(), on_pca_selection("ALL")]).pack(
                padx=10, pady=5)
            return

        # === Estendi immagine a 3 canali se serve ===
        if image_array.shape[2] == 1:
            image_array = np.repeat(image_array, 3, axis=2)

        # === Caso ocSVM ===
        if self.parent.method.get() == "ocSVM":
            kernel = self.parent.ocsvm_kernel_combobox.get()
            # Sanity‑check: if the widget returned an invalid value (e.g. the C slider),
            # force a safe default so the ocSVM backend won’t raise.
            if kernel not in ("linear", "poly", "rbf", "sigmoid"):
                print(f"[DEBUG] Valore kernel non valido ({kernel}), uso 'rbf' come default")
                kernel = "rbf"
            C = self.parent.ocsvm_c_slider.get()
            anomalies = find_anomalies(image_array, threshold, self.parent.anomaly_type.get(),
                                       self.parent.method.get(), kernel, C)
        elif self.parent.method.get() == "DBSCAN":
            anomalies = find_anomalies(image_array, threshold, self.parent.anomaly_type.get(),
                                       self.parent.method.get(), eps, min_samples)
        else:
            anomalies = find_anomalies(image_array, threshold, self.parent.anomaly_type.get(),
                                       self.parent.method.get(), eps, min_samples)

        # === Applica maschera se selezione attiva ===
        img_width, img_height = self.parent.original_image_pil.size
        mask_size = (img_width, img_height)

        if self.parent.image_selection.polygon_points:
            normalized_polygon_points = [(x / img_width, y / img_height)
                                         for x, y in self.parent.image_selection.polygon_points]
            mask = create_mask(mask_size, normalized_polygon_points, normalize=True)
            combined_mask = anomalies & (mask.astype(bool))
            self.parent.analyzed_clean_image = np.copy(image_array)
            composite = composite_anomaly_overlay(image_array, combined_mask, color, opacity=200)
            self.parent.image = composite
            self.display_image_on_canvas(self.parent.image)
            self.parent.zoomed_selection_coords = None

        elif self.parent.image_selection.rect_coords:
            left, top, right, bottom = self.parent.image_selection.rect_coords
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            mask[top:bottom, left:right] = 1
            combined_mask = anomalies & (mask.astype(bool))
            self.parent.analyzed_clean_image = np.copy(image_array)
            composite = composite_anomaly_overlay(image_array, combined_mask, color, opacity=200)
            self.parent.image = composite
            self.display_image_on_canvas(self.parent.image)
            self.parent.zoomed_selection_coords = (left, top, right, bottom)

        else:
            self.parent.analyzed_clean_image = np.copy(image_array)
            composite = composite_anomaly_overlay(image_array, anomalies, color, opacity=200)
            self.parent.image = composite
            self.display_image_on_canvas(self.parent.image)

        # === Aggiorna GUI ===
        self.parent.threshold_slider.config(state=tk.NORMAL, command=self.update_anomalies, fg="white",
                                            troughcolor="light gray")
        self.parent.save_button.config(state=tk.NORMAL)
        self.parent.undo_button.config(state=tk.NORMAL)
        self.parent.redo_button.config(state=tk.DISABLED)
        self.parent.reset_button.config(state=tk.NORMAL)
        self.parent.history.append(np.copy(self.parent.image))
        self.parent.history_index += 1
        self.parent.update_status("Image analyzed")
        self.parent.analyzed_image = np.copy(self.parent.image)
        if self.parent.analyzed_image is not None:
            self.parent.anomaly_editor_button.config(state=tk.NORMAL)
        self.parent.move_logo_to_top_right()

    def update_threshold_slider(self, event=None):
        method = self.parent.method.get()
        if method in ["Isolation Forest", "K-means", "LOF", "RX Detector"]:
            self.parent.threshold_slider.config(from_=0, to=100)
            self.parent.threshold.set(50)
        else:
            self.parent.threshold_slider.config(from_=0, to=255)
            self.parent.threshold.set(100)

    def save_image(self):
        save_path = filedialog.asksaveasfilename(filetypes=[
            ("GeoTIFF files", "*.tif;*.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg;*.jpeg"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ])
        if not save_path:
            return

        _, ext = os.path.splitext(save_path)
        ext = ext.lower()
        if not ext:
            if self.parent.crs is not None and self.parent.geotransform is not None:
                save_path += ".tif"
            else:
                save_path += ".png"

        if self.parent.image is None:
            messagebox.showerror("Error", "No image to save.")
            return

        if self.parent.crs is not None and self.parent.geotransform is not None and save_path.lower().endswith(
                (".tif", ".tiff")):
            import rasterio
            arr = self.parent.image
            if arr.ndim == 3:
                arr = np.transpose(arr, (2, 0, 1))
            else:
                arr = arr[np.newaxis, ...]
            profile = {
                'driver': 'GTiff',
                'height': arr.shape[1],
                'width': arr.shape[2],
                'count': arr.shape[0],
                'dtype': arr.dtype,
                'crs': self.parent.crs,
                'transform': self.parent.geotransform
            }
            with rasterio.open(save_path, 'w', **profile) as dst:
                dst.write(arr)
            messagebox.showinfo("Image Saved", f"GeoTIFF saved to {save_path}")
            self.parent.update_status(f"GeoTIFF saved to {save_path}")
        else:
            if hasattr(self.parent, "pca_component_selector") and self.parent.pca_component_selector.winfo_exists():
                try:
                    selected_idx = int(self.parent.pca_component_selector.get()[2:]) - 1  # es. "PC1" → 0
                    h, w = self.parent.pca_shape
                    component_data = self.parent.pca_result[:, selected_idx].reshape(h, w)
                    norm = (component_data - component_data.min()) / (component_data.max() - component_data.min()) * 255
                    arr = norm.astype('uint8')
                    pil_img = Image.fromarray(arr, mode='L')
                    pil_img.save(save_path)
                    messagebox.showinfo("Image Saved", f"PCA Component saved to {save_path}")
                    self.parent.update_status(f"PCA Component saved to {save_path}")
                    return
                except Exception as e:
                    print("[DEBUG] Errore salvataggio componente PCA selezionata:", e)

            # fallback al salvataggio immagine standard
            arr = self.parent.image.astype('uint8')
            mode = 'RGB' if arr.shape[2] == 3 else 'L'
            pil_img = Image.fromarray(arr[..., 0] if mode == 'L' else arr, mode)
            pil_img.save(save_path)
            messagebox.showinfo("Image Saved", f"Image saved to {save_path}")
            self.parent.update_status(f"Image saved to {save_path}")

    def undo(self):
        if self.parent.history_index > 0:
            self.parent.history_index -= 1
            self.parent.image = self.parent.history[self.parent.history_index]
            self.display_image_on_canvas(self.parent.image)
            self.parent.redo_button.config(state=tk.NORMAL)
            if self.parent.history_index == 0:
                self.parent.undo_button.config(state=tk.DISABLED)
            self.parent.update_status("Undo")

    def redo(self):
        if self.parent.history_index < len(self.parent.history) - 1:
            self.parent.history_index += 1
            self.parent.image = self.parent.history[self.parent.history_index]
            self.display_image_on_canvas(self.parent.image)
            self.parent.undo_button.config(state=tk.NORMAL)
            if self.parent.history_index == len(self.parent.history) - 1:
                self.parent.redo_button.config(state=tk.DISABLED)
            self.parent.update_status("Redo")

    def go_home(self, reset_image=True):
        self.parent.hide_center_logo()

        if reset_image:
            if self.parent.original_image_np.ndim == 3 and self.parent.original_image_np.shape[2] >= 5:
                try:
                    import spectral
                    pseudo_rgb = self.parent.original_image_np[..., [2, 1, 0]].copy()
                except ImportError:
                    pseudo_rgb = self.parent.original_image_np[..., :3].copy()
                pseudo_rgb[pseudo_rgb < -9999] = np.nan
                low = np.nanmin(pseudo_rgb)
                high = np.nanmax(pseudo_rgb)
                if high - low < 1e-6:
                    arr_f = np.zeros_like(pseudo_rgb)
                else:
                    arr_f = (pseudo_rgb - low) / (high - low) * 255.0
                arr_f = np.nan_to_num(arr_f, nan=0)
                stretched = np.clip(arr_f, 0, 255).astype('uint8')
                self.parent.image = stretched
                self.parent.original_image_pil = Image.fromarray(stretched, 'RGB')
                self.parent.band.set("RGB")
            else:
                self.parent.image = np.copy(self.parent.original_image_np)
                self.parent.original_image_pil = Image.fromarray(self.parent.original_image_np.astype('uint8'))

        else:
            self.parent.image = np.copy(self.parent.image)

        # Reset displayed_image so next display_image_on_canvas is not skipped
        self.parent.displayed_image = None

        self.parent.analyzed_image = None
        self.parent.selection = None
        self.parent.image_selection.polygon_points = []
        self.parent.image_selection.rect_coords = None
        self.parent.zoomed_selection_coords = None

        if self.parent.image_selection.rect_id:
            self.parent.canvas.delete(self.parent.image_selection.rect_id)
            self.parent.image_selection.rect_id = None

        self.parent.canvas.delete("all")
        self.parent.scale_factor = 1.0
        self.scale_factor = 1.0
        self.base_display_image = None
        self.parent.x_offset = 0
        self.parent.y_offset = 0

        self.display_image_on_canvas(self.parent.image)

        self.parent.canvas.config(cursor="")
        self.parent.save_button.config(state=tk.DISABLED)
        self.parent.undo_button.config(state=tk.DISABLED)
        self.parent.redo_button.config(state=tk.DISABLED)
        self.parent.method.set("Standard")
        self.parent.threshold.set(100)
        self.parent.anomaly_combobox.set("Darker Pixels")
        self.parent.color_combobox.set("Red")
        self.parent.brightness_slider.set(0)
        self.parent.contrast_slider.set(0)
        self.parent.update_status("Image reset to original")
        self.parent.move_logo_to_top_right()
        self.parent.update_anomaly_type_state()
        self.reset_selection()
        self.parent.select_area_button.config(state=tk.NORMAL)
        self.parent.standard_selection_button.config(state=tk.NORMAL)
        self.parent.pca_selector_label.pack_forget()
        self.parent.pca_component_selector.pack_forget()
        self.parent.pca_component_selector.config(state=tk.DISABLED)

        self.parent.current_image = self.parent.original_image_pil.copy()  # 🔥 QUESTO È FONDAMENTALE
        self.reset_pan()

    def zoom_in(self):
        # Serve una baseline (immagine già ridimensionata per stare nel canvas)
        if self.base_display_image is None:
            if hasattr(self.parent, "current_image") and self.parent.current_image is not None:
                # fallback (meglio avere base_display_image settata da display_image_on_canvas)
                self.base_display_image = self.parent.current_image.copy()
            else:
                print("Errore: Nessuna immagine trovata per zoom.")
                return

        # Incrementa il fattore di scala (relativo alla baseline)
        self.scale_factor *= 1.2

        base = self.base_display_image
        base_w, base_h = base.size

        new_w = max(1, int(base_w * self.scale_factor))
        new_h = max(1, int(base_h * self.scale_factor))
        zoomed_image = base.resize((new_w, new_h), Image.LANCZOS)

        # Ritaglia al centro per mantenere la stessa finestra di visualizzazione (display_w/display_h)
        display_w, display_h = self.parent.display_w, self.parent.display_h
        left = max((new_w - display_w) // 2, 0)
        upper = max((new_h - display_h) // 2, 0)
        right = min(left + display_w, new_w)
        lower = min(upper + display_h, new_h)
        cropped_image = zoomed_image.crop((left, upper, right, lower))

        # Aggiorna lo stato (offset per pan/ticks in coordinate della zoomed_image)
        self.parent.current_image = zoomed_image
        self.parent.x_offset = left
        self.parent.y_offset = upper

        # Aggiorna canvas
        self.parent.tk_image = ImageTk.PhotoImage(cropped_image)
        self.parent.canvas.itemconfig(self.canvas_image, image=self.parent.tk_image)

        self.parent.update_status(f"Zoomed in: {self.scale_factor:.2f}x")
        self.update_ticks_and_scalebar(self.parent.x_offset, self.parent.y_offset, self.scale_factor)
        self.parent.zoom_out_button.config(state=tk.NORMAL)

    def enforce_canvas_bounds(self):
        """
        Mantiene l'immagine all'interno del canvas con il bordo nero e i tick.
        """
        self.canvas.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        bbox = self.canvas.bbox("all")

        if bbox:
            x_min, y_min, x_max, y_max = bbox
            offset_x = 0
            offset_y = 0

            # Verifica se l'immagine esce dai bordi e calcola l'offset
            if x_min > 0:
                offset_x = -x_min
            elif x_max < canvas_width:
                offset_x = canvas_width - x_max

            if y_min > 0:
                offset_y = -y_min
            elif y_max < canvas_height:
                offset_y = canvas_height - y_max

            # Sposta l'immagine per mantenerla nei limiti
            self.canvas.move("all", offset_x, offset_y)

    def zoom_out(self):
        if self.base_display_image is None:
            if hasattr(self.parent, "current_image") and self.parent.current_image is not None:
                self.base_display_image = self.parent.current_image.copy()
            else:
                print("Errore: Nessuna immagine trovata per zoom.")
                return

        new_scale = self.scale_factor / 1.2

        # Se scendiamo a 1.0 (o sotto), reset completo: torna ESATTAMENTE alla baseline
        if new_scale <= 1.0:
            self.scale_factor = 1.0
            base = self.base_display_image

            self.parent.current_image = base.copy()
            self.parent.x_offset = 0
            self.parent.y_offset = 0

            self.parent.tk_image = ImageTk.PhotoImage(base)
            self.parent.canvas.itemconfig(self.canvas_image, image=self.parent.tk_image)

            self.parent.update_status("Zoomed out: 1.00x")
            self.update_ticks_and_scalebar(self.parent.x_offset, self.parent.y_offset, self.scale_factor)
            self.parent.zoom_out_button.config(state=tk.DISABLED)
            return

        # Aggiorna lo scale_factor e ridisegna partendo SEMPRE dalla baseline
        self.scale_factor = new_scale

        base = self.base_display_image
        base_w, base_h = base.size

        new_w = max(1, int(base_w * self.scale_factor))
        new_h = max(1, int(base_h * self.scale_factor))
        zoomed_image = base.resize((new_w, new_h), Image.LANCZOS)

        display_w, display_h = self.parent.display_w, self.parent.display_h
        left = max((new_w - display_w) // 2, 0)
        upper = max((new_h - display_h) // 2, 0)
        right = min(left + display_w, new_w)
        lower = min(upper + display_h, new_h)
        cropped_image = zoomed_image.crop((left, upper, right, lower))

        self.parent.current_image = zoomed_image
        self.parent.x_offset = left
        self.parent.y_offset = upper

        self.parent.tk_image = ImageTk.PhotoImage(cropped_image)
        self.parent.canvas.itemconfig(self.canvas_image, image=self.parent.tk_image)

        self.parent.update_status(f"Zoomed out: {self.scale_factor:.2f}x")
        self.update_ticks_and_scalebar(self.parent.x_offset, self.parent.y_offset, self.scale_factor)

        if self.scale_factor <= 1.000001:
            self.parent.zoom_out_button.config(state=tk.DISABLED)

    def zoom_to_selection(self):
        if self.parent.image is None:
            self.parent.update_status("No valid selection to zoom")
            return

        arr = self.parent.image
        mode = 'RGB' if arr.shape[2] == 3 else 'L'
        pil_img = Image.fromarray(arr[...,0] if mode=='L' else arr.astype('uint8'), mode)

        img_w, img_h = self.parent.original_image_pil.size

        if self.parent.image_selection.polygon_points:
            x_coords, y_coords = zip(*(self.parent.image_selection.polygon_points))
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)


            min_x = max(int(min_x), 0)
            max_x = min(int(max_x), img_w)
            min_y = max(int(min_y), 0)
            max_y = min(int(max_y), img_h)


            roi = pil_img.crop((min_x, min_y, max_x, max_y))
            roi_width, roi_height = roi.size
            # Applica un fattore di zoom fisso (ad es. 2.0) per ingrandire la ROI
            zoom_factor = 2.0
            new_size = (int(roi_width * zoom_factor), int(roi_height * zoom_factor))
            zoomed_image = roi.resize(new_size, Image.LANCZOS)

            new_arr = np.array(zoomed_image)
            if new_arr.ndim == 2:
                new_arr = new_arr[..., np.newaxis]
            self.parent.image = new_arr
            self.parent.current_image = zoomed_image
            self.display_image_on_canvas(self.parent.image)
            self.parent.update_status("Zoomed to selection")
            self.parent.zoomed_selection_coords = (min_x, min_y, max_x, max_y)

        elif self.parent.zoomed_selection_coords:
            left, top, right, bottom = self.parent.zoomed_selection_coords

            # Poi clamp e un nuovo print
            left = max(int(left), 0)
            right = min(int(right), img_w)
            top = max(int(top), 0)
            bottom = min(int(bottom), img_h)

            roi = pil_img.crop((left, top, right, bottom))
            roi_width, roi_height = roi.size
            # Applica un fattore di zoom fisso (ad es. 2.0) per ingrandire la ROI
            zoom_factor = 2.0
            new_size = (int(roi_width * zoom_factor), int(roi_height * zoom_factor))
            zoomed_image = roi.resize(new_size, Image.LANCZOS)

            new_arr = np.array(zoomed_image)
            if new_arr.ndim == 2:
                new_arr = new_arr[..., np.newaxis]
            self.parent.image = new_arr
            self.parent.current_image = zoomed_image
            self.display_image_on_canvas(self.parent.image)
            self.parent.update_status("Zoomed to selection")
        else:
            self.parent.update_status("No valid selection to zoom")

        self.update_ticks_and_scalebar(self.parent.x_offset, self.parent.y_offset, self.scale_factor)

    def update_ticks_and_scalebar(self, x_offset, y_offset, scale_factor):
        if not self.parent.crs or not self.parent.geotransform:
            # Niente CRS => salta TUTTA la parte cartografica
            print("[DEBUG] No valid CRS, skipping tick/scalebar update.")
            return
        """
        Aggiorna le etichette dei ticks e la barra di scala in base a:
          - porzione effettiva dell'immagine visualizzata (dopo zoom/pan)
          - bounding box geografico di quella porzione
          - calcolo della scala in metri/pixel e aggiornamento scalebar
        """

        print("[DEBUG] Inizio update_ticks_and_scalebar...")

        from rasterio.transform import xy
        transformer_to_geographic = Transformer.from_crs(self.parent.crs, "EPSG:4326", always_xy=True)

        # ----------------------------------------------------------------------------
        # (1) Trova la porzione di immagine (in pixel originali) attualmente visualizzata
        # ----------------------------------------------------------------------------
        #  - self.parent.current_image: immagine ingrandita/ridotta
        #  - self.parent.original_image: immagine originale
        #  - self.parent.x_offset, self.parent.y_offset: offset di "pan" nella immagine zoommata
        #  - canvas.bbox(self.canvas_image) -> bounding box sul canvas (x1_canvas, y1_canvas, x2_canvas, y2_canvas)
        #    che ci serve per posizionare i tick in coordinate canvas.

        # Otteniamo la bounding box (in coordinate canvas) dell'immagine visualizzata
        x1_canvas, y1_canvas, x2_canvas, y2_canvas = self.parent.canvas.bbox(self.canvas_image)
        disp_w = x2_canvas - x1_canvas
        disp_h = y2_canvas - y1_canvas

        # Calcola fattore di scala "effettivo":
        # (es. se current_image.width = 2000 e original_image.width = 1000, scale_factor_actual=2)
        scale_factor_actual = self.parent.current_image.width / self.parent.original_image_pil.width

        # Ora calcoliamo, in pixel ORIGINALI, i 4 corner della porzione visualizzata
        #  px_left  = x_offset / scale_factor_actual
        #  px_right = (x_offset + disp_w) / scale_factor_actual
        #  etc.
        #  NOTA: row = coordinata verticale (y), col = coordinata orizzontale (x)

        px_left = self.parent.x_offset / scale_factor_actual
        px_top = self.parent.y_offset / scale_factor_actual
        px_right = (self.parent.x_offset + disp_w) / scale_factor_actual
        px_bottom = (self.parent.y_offset + disp_h) / scale_factor_actual

        # Clamp per evitare fuoriuscite:
        px_left = max(0, min(px_left, self.parent.original_image_pil.width))
        px_right = max(0, min(px_right, self.parent.original_image_pil.width))
        px_top = max(0, min(px_top, self.parent.original_image_pil.height))
        px_bottom = max(0, min(px_bottom, self.parent.original_image_pil.height))

        # Corner in (row, col) = (y, x)
        # top-left
        tl_x, tl_y = px_left, px_top
        # top-right
        tr_x, tr_y = px_right, px_top
        # bottom-left
        bl_x, bl_y = px_left, px_bottom
        # bottom-right
        br_x, br_y = px_right, px_bottom

        # ----------------------------------------------------------------------------
        # (2) Converte questi corner in lat/lon e determina lat_min,lat_max,lon_min,lon_max
        # ----------------------------------------------------------------------------
        # xy(transform, righe, colonne) usa la convenzione: row -> lat, col -> lon
        # Quindi passiamo (row=y, col=x) corrispondenti

        def pixel_to_latlon(row, col):
            """Converte (row, col) pixel originali in lat/lon."""
            x_geo, y_geo = xy(self.parent.geotransform, row, col)  # row->y, col->x
            lon, lat = transformer_to_geographic.transform(x_geo, y_geo)
            return lat, lon

        lat_tl, lon_tl = pixel_to_latlon(tl_y, tl_x)
        lat_tr, lon_tr = pixel_to_latlon(tr_y, tr_x)
        lat_bl, lon_bl = pixel_to_latlon(bl_y, bl_x)
        lat_br, lon_br = pixel_to_latlon(br_y, br_x)

        lat_vals = [lat_tl, lat_tr, lat_bl, lat_br]
        lon_vals = [lon_tl, lon_tr, lon_bl, lon_br]
        lat_min, lat_max = min(lat_vals), max(lat_vals)
        lon_min, lon_max = min(lon_vals), max(lon_vals)

        print(f"[DEBUG] lat_min={lat_min}, lat_max={lat_max}, lon_min={lon_min}, lon_max={lon_max}")

        # ----------------------------------------------------------------------------
        # (3) Calcolo pixel_width_m e aggiorna SCALABAR
        # ----------------------------------------------------------------------------
        # Se CRS geografico, calcoliamo la larghezza di 1 pixel in metri via to_utm.
        if CRS(self.parent.crs).is_geographic:
            # Convertiamo un delta lon in un delta metri
            utm_crs = CRS(self.parent.crs).to_utm()
            transformer_to_utm = Transformer.from_crs(self.parent.crs, utm_crs, always_xy=True)
            # Larghezza di un pixel in coordinate originali
            pixel_width_degrees = abs(self.parent.geotransform.a)
            lon1, lat1 = self.parent.geotransform.c, self.parent.geotransform.f
            lon2, lat2 = lon1 + pixel_width_degrees, lat1
            x1_m, y1_m = transformer_to_utm.transform(lon1, lat1)
            x2_m, y2_m = transformer_to_utm.transform(lon2, lat2)
            pixel_width_m = abs(x2_m - x1_m)
        else:
            pixel_width_m = abs(self.parent.geotransform.a)

        # Ora, in base al fattore di zoom:
        # pixel_width_m_displayed = pixel_width_m / scale_factor_actual
        pixel_width_m_displayed = pixel_width_m / scale_factor_actual
        print(f"[DEBUG] pixel_width_m={pixel_width_m}, pixel_width_m_displayed={pixel_width_m_displayed}")

        # ---- Aggiorna la scalebar esattamente come facevi prima ----
        if len(self.parent.scalebar_ids) >= 1:
            gui_width = self.parent.winfo_width()
            scale_length_px_gui = int((gui_width / 20) * 1.5)
            proposed_length_m = scale_length_px_gui * pixel_width_m_displayed

            base_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            chosen_scale_m = min(base_values, key=lambda x: abs(x - proposed_length_m))

            scalebar_length_px = int(chosen_scale_m / pixel_width_m_displayed)
            if chosen_scale_m < 1000:
                scale_text = f"{chosen_scale_m} m"
            else:
                km_value = chosen_scale_m / 1000.0
                scale_text = f"{km_value:.1f} km"

            # L'ultimo ID in scalebar_ids è la label testuale
            self.parent.canvas.itemconfig(self.parent.scalebar_ids[-1], text=scale_text)

            self.parent.canvas.coords(
                self.parent.scalebar_ids[-1],  # ID della label finale
                self.parent.scalebar_x + scalebar_length_px,
                self.parent.scalebar_y + 20
            )

            # I primi 5 ID: segmenti della scalebar
            for i, seg in enumerate(range(5)):
                seg_x1 = self.parent.scalebar_x + seg * (scalebar_length_px // 5)
                seg_x2 = seg_x1 + (scalebar_length_px // 5)
                fill_color = "black" if seg % 2 == 0 else "white"
                self.parent.canvas.coords(
                    self.parent.scalebar_ids[i],
                    seg_x1, self.parent.scalebar_y,
                    seg_x2, self.parent.scalebar_y + 10
                )
                self.parent.canvas.itemconfig(self.parent.scalebar_ids[i], fill=fill_color)

        # ----------------------------------------------------------------------------
        # (4) Cancella tick vecchi e disegna i nuovi sul bordo nero
        # ----------------------------------------------------------------------------
        for tick_id in self.parent.tick_ids:
            self.parent.canvas.delete(tick_id)
        self.parent.tick_ids.clear()

        # I "bordi fisici" dell'immagine visualizzata sul canvas
        # sono (x1_canvas, y1_canvas) in alto-sinistra, (x2_canvas, y2_canvas) in basso-destra

        disp_w = x2_canvas - x1_canvas
        disp_h = y2_canvas - y1_canvas
        tick_len = 10

        def to_dms(value, is_lat=False):
            degrees = int(value)
            minutes = int((abs(value) - abs(degrees)) * 60)
            seconds = (abs(value) - abs(degrees)) * 3600 - minutes * 60
            hemi = ''
            if is_lat:
                hemi = 'N' if value >= 0 else 'S'
            else:
                hemi = 'E' if value >= 0 else 'W'
            return f"{abs(degrees)}°{minutes}'{seconds:.1f}\"{hemi}"

        # Tick lat (alto/basso) -> lat_max / lat_min
        lat_positions = [y1_canvas, y2_canvas]
        lat_values = [lat_max, lat_min]

        for pos_canvas, lat_val in zip(lat_positions, lat_values):
            label_text = to_dms(lat_val, is_lat=True)

            # Lato sinistro
            left_tick = self.parent.canvas.create_line(
                x1_canvas, pos_canvas,
                x1_canvas - tick_len, pos_canvas,
                fill="black"
            )
            self.parent.tick_ids.append(left_tick)
            left_label = self.parent.canvas.create_text(
                x1_canvas - tick_len - 40, pos_canvas,
                text=label_text, fill="black", font=("Arial", 15, "bold")
            )
            self.parent.tick_ids.append(left_label)

            # Lato destro
            right_tick = self.parent.canvas.create_line(
                x2_canvas, pos_canvas,
                x2_canvas + tick_len, pos_canvas,
                fill="black"
            )
            self.parent.tick_ids.append(right_tick)
            right_label = self.parent.canvas.create_text(
                x2_canvas + tick_len + 40, pos_canvas,
                text=label_text, fill="black", font=("Arial", 15, "bold")
            )
            self.parent.tick_ids.append(right_label)

        # Tick lon (sinistra / centro / destra)
        lon_positions = [
            x1_canvas,
            x1_canvas + disp_w / 2,
            x2_canvas
        ]
        lon_values = [
            lon_min,
            (lon_min + lon_max) / 2,
            lon_max
        ]

        for pos_canvas, lon_val in zip(lon_positions, lon_values):
            label_text = to_dms(lon_val, is_lat=False)

            # Lato superiore
            top_tick = self.parent.canvas.create_line(
                pos_canvas, y1_canvas,
                pos_canvas, y1_canvas - tick_len,
                fill="black"
            )
            self.parent.tick_ids.append(top_tick)
            top_label = self.parent.canvas.create_text(
                pos_canvas, y1_canvas - tick_len - 10,
                text=label_text, fill="black", font=("Arial", 15, "bold")
            )
            self.parent.tick_ids.append(top_label)

            # Lato inferiore
            bottom_tick = self.parent.canvas.create_line(
                pos_canvas, y2_canvas,
                pos_canvas, y2_canvas + tick_len,
                fill="black"
            )
            self.parent.tick_ids.append(bottom_tick)
            bottom_label = self.parent.canvas.create_text(
                pos_canvas, y2_canvas + tick_len + 10,
                text=label_text, fill="black", font=("Arial", 15, "bold")
            )
            self.parent.tick_ids.append(bottom_label)

        print("[DEBUG] Fine aggiornamento ticks e scalebar.")



    def start_pan(self):
        self.parent.canvas.config(cursor="fleur")
        self.parent.canvas.bind("<ButtonPress-1>", self.start_pan_drag)
        self.parent.canvas.bind("<B1-Motion>", self.pan_image)
        self.parent.canvas.bind("<ButtonRelease-1>", self.end_pan)

    def start_pan_drag(self, event):
        self.panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        print(f"[DEBUG] Inizio pan: pan_start_x={self.pan_start_x}, pan_start_y={self.pan_start_y}")

    def pan_image(self, event):
        if not self.panning:
            return

        # Calcolo del delta del movimento del mouse
        dx = self.pan_start_x - event.x
        dy = self.pan_start_y - event.y

        # Aggiorno la posizione iniziale del pan
        self.pan_start_x = event.x
        self.pan_start_y = event.y

        # Calcolo nuovi offset
        new_x_offset = self.parent.x_offset + dx
        new_y_offset = self.parent.y_offset + dy

        # Dimensioni immagine zoomata e bordi neri
        img_width, img_height = self.parent.current_image.size
        canvas_width = self.parent.display_w  # Dimensioni dell'area con i bordi neri
        canvas_height = self.parent.display_h

        # Calcolo limiti offset per i bordi neri
        min_x_offset = 0
        min_y_offset = 0
        max_x_offset = img_width - canvas_width
        max_y_offset = img_height - canvas_height

        # Clamp degli offset ai limiti dell'immagine zoomata
        clamped_x_offset = max(min_x_offset, min(new_x_offset, max_x_offset))
        clamped_y_offset = max(min_y_offset, min(new_y_offset, max_y_offset))

        # Aggiorno gli offset globali
        self.parent.x_offset = clamped_x_offset
        self.parent.y_offset = clamped_y_offset

        # Ritaglio e aggiornamento immagine
        self.update_view()

    def update_view(self):
        """
        Aggiorna la porzione visibile dell'immagine basata sugli offset correnti.
        """
        # Dimensioni dell'immagine zoomata
        img_width, img_height = self.parent.current_image.size

        # Dimensioni dell'area visibile con i bordi neri
        canvas_width = self.parent.display_w
        canvas_height = self.parent.display_h

        # Calcola le coordinate del ritaglio
        left = self.parent.x_offset
        top = self.parent.y_offset
        right = left + canvas_width
        bottom = top + canvas_height

        # Ritaglia l'immagine
        cropped_image = self.parent.current_image.crop((left, top, right, bottom))

        # Aggiorna la porzione visibile sul canvas
        self.parent.tk_image = ImageTk.PhotoImage(cropped_image)
        self.parent.canvas.itemconfig(self.canvas_image, image=self.parent.tk_image)
    def end_pan(self, event):
        self.panning = False
        self.parent.canvas.config(cursor="")
        print(f"[DEBUG] Fine pan: x_offset={self.parent.x_offset}, y_offset={self.parent.y_offset}")

        # Richiama l'update dei ticks e della scalebar,
        # usando l'offset e il fattore di scala corrente
        self.update_ticks_and_scalebar(self.parent.x_offset, self.parent.y_offset, self.scale_factor)

    def reset_pan(self):
        self.panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.parent.x_offset = 0
        self.parent.y_offset = 0
        self.parent.canvas.unbind("<ButtonPress-1>")
        self.parent.canvas.unbind("<B1-Motion>")
        self.parent.canvas.unbind("<ButtonRelease-1>")

    def reset_selection(self):
        self.parent.canvas.bind("<ButtonPress-1>", self.parent.image_selection.on_button_press)
        self.parent.canvas.bind("<B1-Motion>", self.parent.image_selection.on_move_press)
        self.parent.canvas.bind("<ButtonRelease-1>", self.parent.image_selection.on_button_release)
        self.parent.canvas.bind("<Double-Button-1>", self.parent.image_selection.on_button_double_click)
        self.parent.select_area_button.config(state=tk.NORMAL)
        self.parent.image_selection.polygon_points = []
        self.parent.image_selection.rect_coords = None
        if self.parent.image_selection.rect_id:
            self.parent.canvas.delete(self.parent.image_selection.rect_id)
            self.parent.image_selection.rect_id = None

    def open_pca_viewer(self):
        if not hasattr(self.parent, "pca_result") or self.parent.pca_result is None:
            tk.messagebox.showwarning("PCA Viewer", "No PCA result available.")
            return

        # Recupera maschera e bbox corretti
        selection_mask = getattr(self.parent, "pca_mask", None)
        selection_bbox = None

        if self.parent.image_selection.rect_coords:
            selection_bbox = self.parent.image_selection.rect_coords
        elif self.parent.image_selection.polygon_points:
            # Calcola bounding box anche per selezione poligonale
            xs = [x for x, y in self.parent.image_selection.polygon_points]
            ys = [y for x, y in self.parent.image_selection.polygon_points]
            selection_bbox = (min(xs), min(ys), max(xs), max(ys))

        # Esegui crop della PCA se selezione esiste
        if selection_bbox is not None:
            left, top, right, bottom = selection_bbox
            h_crop = bottom - top
            w_crop = right - left
            n_components = self.parent.pca_result.shape[1]

            pca_result_cropped = np.empty((h_crop * w_crop, n_components), dtype=np.float32)
            for i in range(n_components):
                full_component = self.parent.pca_result[:, i].reshape(self.parent.pca_shape)
                cropped_component = full_component[top:bottom, left:right].flatten()
                pca_result_cropped[:, i] = cropped_component

            pca_shape_cropped = (h_crop, w_crop)
        else:
            pca_result_cropped = self.parent.pca_result
            pca_shape_cropped = self.parent.pca_shape

        # Apri il viewer con i parametri corretti
        PCAEditor(
            self.parent,
            self,
            pca_result_cropped,
            pca_shape_cropped,
            self.parent.pca_mode_used,
            self.parent.pca_colormap.get(),
            selection_mask=selection_mask,
            selection_bbox=selection_bbox,
            explained_variance_ratio=self.parent.explained_variance_ratio
        )

    def on_main_resize(self, event=None):
        if hasattr(self.parent, "image") and self.parent.image is not None:
            self.display_image_on_canvas(self.parent.image)

    # Ensure displayed_image is reset when loading a new image
    def load_image(self, *args, **kwargs):
        # Placeholder: actual implementation should reset displayed_image
        self.parent.displayed_image = None
