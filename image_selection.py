import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from PIL import Image
from tkinter import filedialog, messagebox
import tkinter as tk

class ImageSelection:
    def __init__(self, parent):
        self.parent = parent
        self.polygon_points = []
        self.poly_line = None
        self.selection_finished = False
        self.rect_id = None
        self.rect_coords = None
        self.start_x = 0
        self.start_y = 0
        self.image = None  # memorizza l'immagine come array NumPy

    # --- helper: crea preview RGB 8 bit da stack multispettrale (B,G,R,[NIR]) ---
    def _make_rgb_preview(self, ms_arr):
        """
        ms_arr: (H,W,C) con C>=3, uint16/float/uint8
        Ritorna (H,W,3) uint8 mappata R=band3, G=band2, B=band1 con stretch percentili 2-98.
        """
        import numpy as np

        H, W, C = ms_arr.shape
        # mappa True Color tipica (es. PlanetScope): R=3, G=2, B=1
        r = ms_arr[:, :, 2].astype(np.float32)
        g = ms_arr[:, :, 1].astype(np.float32)
        b = ms_arr[:, :, 0].astype(np.float32)

        def _stretch_2_98(x):
            p2, p98 = np.percentile(x, (2, 98))
            if p98 <= p2:
                p2, p98 = float(x.min()), float(x.max() if x.max() > x.min() else x.min() + 1.0)
            x = (x - p2) / (p98 - p2)
            x = np.clip(x, 0.0, 1.0)
            return (x * 255.0).astype(np.uint8)

        r8 = _stretch_2_98(r)
        g8 = _stretch_2_98(g)
        b8 = _stretch_2_98(b)
        return np.dstack([r8, g8, b8])

    def load_multispectral_image(self, file_path):
        with rasterio.open(file_path) as src:
            bands = []
            for i in range(1, src.count + 1):
                band = src.read(i)
                bands.append(band)
            multispectral_image = np.stack(bands, axis=-1)  # (H, W, n_bands)
            self.parent.geotransform = src.transform
            self.parent.crs = src.crs
        return multispectral_image

    def load_image(self):
        self.parent.image_path = filedialog.askopenfilename(filetypes=[
            ("Multispectral files", "*.tif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ])
        if not self.parent.image_path:
            return

        try:
            # Verifica se l'immagine è multispettrale
            if self.parent.image_path.endswith(('.tif', '.tiff')):
                with rasterio.open(self.parent.image_path) as src:
                    # Definisci il CRS target proiettato (es. EPSG:3857)
                    target_crs = 'EPSG:3857'

                    # Calcola transform, width, height per il nuovo CRS
                    transform, width, height = calculate_default_transform(
                        src.crs, target_crs, src.width, src.height, *src.bounds
                    )

                    profile = src.profile
                    profile.update({
                        'crs': target_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })

                    # Crea un array numpy per contenere la nuova immagine riproiettata
                    reprojected_array = np.zeros((height, width, src.count), dtype=src.meta['dtype'])

                    # Riproietta ogni banda
                    for i in range(1, src.count + 1):
                        reproject(
                            source=src.read(i),
                            destination=reprojected_array[:, :, i - 1],
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest
                        )

                    # Aggiorna geoinfo
                    self.parent.geotransform = transform
                    self.parent.crs = target_crs

                    if reprojected_array.ndim == 2:
                        reprojected_array = reprojected_array[..., np.newaxis]

                    # Stack completo per le analisi
                    self.image = reprojected_array
                    self.parent.multispectral_stack = self.image

                    # Numero bande e flag multispettrale
                    num_bands = self.image.shape[2]
                    self.parent.is_multispectral = (num_bands > 3)

                    # --- Anteprima per il canvas: sempre 3 canali ---
                    if num_bands >= 3:
                        rgb_preview = self._make_rgb_preview(self.image)
                    else:
                        ch = self.image[:, :, 0]
                        ch8 = (255 * (ch - ch.min()) / (ch.max() - ch.min() + 1e-9)).astype(np.uint8)
                        rgb_preview = np.dstack([ch8, ch8, ch8])

                    # Aggiorna riferimenti usati dal renderer
                    self.parent.displayed_image = None
                    self.parent.original_image = rgb_preview.copy()
                    self.parent.original_image_np = rgb_preview.copy()
            else:
                pil_img = Image.open(self.parent.image_path)
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                image_array = np.array(pil_img)
                if image_array.ndim == 2:
                    image_array = image_array[..., np.newaxis]

                self.image = image_array
                self.parent.displayed_image = None  # forza ridisegno in display_image_on_canvas
                self.parent.original_image = self.image.copy()
                self.parent.original_image_np = self.image.copy()
                num_bands = self.image.shape[2]
                self.parent.is_multispectral = (num_bands > 3)
                self.parent.geotransform = None
                self.parent.crs = None

            # Aggiorna la combobox delle bande in base al numero di bande
            if num_bands == 1:
                # Monobanda
                self.parent.band_combobox.config(values=["Band 1"], state="disabled")
                self.parent.band.set("Band 1")
            elif num_bands == 3:
                # RGB
                self.parent.band_combobox.config(values=["RGB"] + ["Band 1 (R)", "Band 2 (G)", "Band 3 (B)"], state="readonly")
                self.parent.band.set("RGB")
            else:
                # Multispettrale
                bands_list =["RGB"] + [f"Band {i+1}" for i in range(num_bands)]
                self.parent.band_combobox.config(values=bands_list, state="readonly")
                self.parent.band.set("RGB")

            self.parent.image_processing_gui.display_image_on_canvas(self.parent.original_image_np)
            self.parent.select_area_button.config(state="normal")
            self.parent.analyze_button.config(state="normal")
            if hasattr(self.parent, "back_button"):
                self.parent.back_button.config(state="normal")
            self.parent.zoom_in_button.config(state="normal")
            self.parent.zoom_to_selection_button.config(state="normal")
            self.parent.pan_button.config(state="normal")
            self.parent.standard_selection_button.config(state="normal")
            self.parent.update_status("Image loaded")

            self.parent.hide_center_logo()
            self.parent.move_logo_to_top_right()

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def enable_selection(self):
        # Disabilita i binding di pan:
        self.parent.canvas.unbind("<B1-Motion>")
        self.parent.canvas.unbind("<ButtonPress-1>")
        self.parent.canvas.unbind("<ButtonRelease-1>")

        self.polygon_points = []

        # Pulisci canvas ma senza cancellare tutto (evita "all")
        # Elimina solo elementi della selezione precedente, se servisse (opzionale):
        # self.parent.canvas.delete("polygon")  # Se usi tag
        # oppure lascia solo questo:
        # Mostra l’immagine attualmente in uso (eventuali modifiche contrasto/istogramma incluse)
        import numpy as np
        self.parent.image_processing_gui.display_image_on_canvas(np.copy(self.parent.original_image_np))

        # Ora bind solo i tasti per poligono
        self.parent.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.parent.canvas.bind("<Double-Button-1>", self.on_button_double_click)

        self.parent.canvas.config(cursor="cross")
        self.parent.update_status("Select area by clicking points, double-click to complete")

        self.parent.zoom_to_selection_button.config(state="normal")
        self.parent.pan_button.config(state="disabled")

    def enable_standard_selection(self):
        if self.selection_finished:
            # Cancella la selezione poligonale completata
            self.parent.canvas.delete("all")
            # Ripristina l'immagine (senza i disegni di selezione)
            self.parent.image_processing_gui.display_image_on_canvas(self.parent.original_image_np)
            self.polygon_points = []
            self.poly_line = None
            self.selection_finished = False

        self.parent.canvas.bind("<ButtonPress-1>", self.start_standard_selection)
        self.parent.canvas.bind("<B1-Motion>", self.update_standard_selection)
        self.parent.canvas.bind("<ButtonRelease-1>", self.end_standard_selection)
        self.parent.update_status("Standard selection enabled")

    def on_button_press(self, event):
        if self.selection_finished:
            # Cancella la selezione poligonale completata
            self.parent.canvas.delete("all")
            # Ripristina l'immagine (senza i disegni di selezione)
            self.parent.image_processing_gui.display_image_on_canvas(self.parent.original_image_np)
            self.polygon_points = []
            self.poly_line = None
            self.selection_finished = False
            self.parent.update_status("Original image")


        if not self.parent.canvas.config()['cursor'][-1] == "cross":
            return

        self.polygon_points.append((event.x, event.y))

        radius = 3
        self.parent.canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius,
                                       fill='red')

        if self.poly_line:
            self.parent.canvas.delete(self.poly_line)

        if len(self.polygon_points) > 1:
            self.poly_line = self.parent.canvas.create_line(
                *sum(self.polygon_points, ()),
                fill='red'
            )

    def on_move_press(self, event):
        # Se l'attributo display_w non è stato ancora definito, esci subito
        if not hasattr(self.parent, "display_w") or self.parent.display_w is None:
            return

        # Calcola il movimento del mouse rispetto alla posizione iniziale
        dx = event.x - self.start_x
        dy = event.y - self.start_y

        # Calcola i nuovi offset
        new_x_offset = self.parent.x_offset + dx
        new_y_offset = self.parent.y_offset + dy

        # Dimensioni del canvas
        canvas_width = self.parent.canvas.winfo_width()
        canvas_height = self.parent.canvas.winfo_height()

        # Dimensioni dell'immagine zoomata
        img_width = self.parent.display_w
        img_height = self.parent.display_h

        # Calcola i limiti degli offset per mantenere l'immagine nel canvas
        min_x_offset = min(0, canvas_width - img_width)
        max_x_offset = max(0, canvas_width - img_width)
        min_y_offset = min(0, canvas_height - img_height)
        max_y_offset = max(0, canvas_height - img_height)

        # Restringi i nuovi offset ai limiti calcolati
        clamped_x_offset = max(min_x_offset, min(new_x_offset, max_x_offset))
        clamped_y_offset = max(min_y_offset, min(new_y_offset, max_y_offset))

        # Aggiorna gli offset della posizione iniziale
        self.parent.x_offset = clamped_x_offset
        self.parent.y_offset = clamped_y_offset

        # Aggiorna le coordinate di partenza del mouse
        self.start_x = event.x
        self.start_y = event.y

        # Aggiorna l'immagine sul canvas
        self.parent.image_processing_gui.display_image_on_canvas(self.parent.original_image_np)

        # Aggiorna lo stato
        self.update_status(f"Pan in corso: x_offset={self.parent.x_offset}, y_offset={self.parent.y_offset}")

    def on_button_release(self, event):
        # Verifica se il pan è attivo (ad esempio se il flag panning del parent è True)
        if hasattr(self.parent, "panning") and self.parent.panning:
            # Usa gli offset definiti nel parent
            self.parent.update_status(
                f"Pan completato: x_offset={self.parent.x_offset}, y_offset={self.parent.y_offset}")
        # Altrimenti non fare nulla (o eventualmente ripristina la modalità di selezione)

    def on_button_double_click(self, event):
        if len(self.polygon_points) > 2:
            # Disegna il poligono sul canvas come prima
            self.parent.canvas.create_polygon(self.polygon_points, outline='red', fill='', width=2)
            self.parent.canvas.config(cursor="")
            self.parent.update_status("Polygon area selected")


            # Ora convertiamo i punti dal sistema di coordinate del canvas a quello dell'immagine originale
            image_array = np.array(self.parent.original_image_np)
            img_height, img_width = image_array.shape[:2]

            actual_scale = self.parent.effective_scale_factor
            canvas_width = self.parent.canvas.winfo_width()
            canvas_height = self.parent.canvas.winfo_height()
            center_x = (canvas_width - self.parent.display_w) // 2
            center_y = (canvas_height - self.parent.display_h) // 2

            disp_x_offset = center_x + self.parent.x_offset
            disp_y_offset = center_y + self.parent.y_offset

            converted_points = []
            for (x_canvas, y_canvas) in self.polygon_points:
                x_in_disp = x_canvas - disp_x_offset
                y_in_disp = y_canvas - disp_y_offset
                x_orig = x_in_disp / actual_scale
                y_orig = y_in_disp / actual_scale
                converted_points.append((x_orig, y_orig))

            self.polygon_points = converted_points
            self.selection_finished = True

    def start_standard_selection(self, event):
        if self.rect_id:
            self.parent.canvas.delete(self.rect_id)
        self.start_x = event.x
        self.start_y = event.y
        self.rect_id = self.parent.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y,
                                                           outline="red")

    def update_standard_selection(self, event):
        self.parent.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def end_standard_selection(self, event):
        canvas_width = self.parent.canvas.winfo_width()
        canvas_height = self.parent.canvas.winfo_height()
        img_height, img_width = self.parent.original_image_np.shape[:2]


        self.end_x = event.x
        self.end_y = event.y
        self.parent.update_status("Area selected")
        self.parent.canvas.config(cursor="")
        self.parent.zoom_to_selection_button.config(state="normal")

        # Usa la scala effettiva salvata dopo display_image_on_canvas:
        actual_scale = self.parent.effective_scale_factor

        # Calcola l'offset dovuto al centramento dell'immagine sul canvas:
        canvas_width = self.parent.canvas.winfo_width()
        canvas_height = self.parent.canvas.winfo_height()
        center_x = (canvas_width - self.parent.display_w) // 2
        center_y = (canvas_height - self.parent.display_h) // 2

        # Gli offset effettivi sono la somma dell'offset di pan e di quello dovuto al centramento:
        disp_x_offset = center_x + self.parent.x_offset
        disp_y_offset = center_y + self.parent.y_offset

        # Calcola le coordinate in "display space"
        left_in_disp = self.start_x - disp_x_offset
        top_in_disp = self.start_y - disp_y_offset
        right_in_disp = self.end_x - disp_x_offset
        bottom_in_disp = self.end_y - disp_y_offset

        # Converti in coordinate originali
        left = int(left_in_disp / actual_scale)
        top = int(top_in_disp / actual_scale)
        right = int(right_in_disp / actual_scale)
        bottom = int(bottom_in_disp / actual_scale)

        # Ora clamp
        left = max(left, 0)
        top = max(top, 0)
        right = min(right, img_width)
        bottom = min(bottom, img_height)

        self.rect_coords = (left, top, right, bottom)
        self.parent.image_selection.rect_coords = self.rect_coords
        self.parent.zoomed_selection_coords = self.rect_coords


    def update_status(self, message):
        self.parent.update_status(message)
    # ------------------------------------------------------------------
    #  Utility methods to completely reset any active selection
    # ------------------------------------------------------------------
    def clear_canvas_selection(self):
        """Remove selection graphics (polygon/rectangle) from the canvas."""
        if self.poly_line:
            self.parent.canvas.delete(self.poly_line)
            self.poly_line = None
        if self.rect_id:
            self.parent.canvas.delete(self.rect_id)
            self.rect_id = None

    def reset_selection(self):
        """
        Restore the initial selection state, clearing graphics & data and
        disabling related GUI buttons.
        """
        # Remove graphics elements
        self.clear_canvas_selection()

        # Reset logical data
        self.polygon_points = []
        self.rect_coords = None
        self.selection_finished = False

        # Restore parent GUI state
        self.parent.canvas.config(cursor="")
        if hasattr(self.parent, "zoom_to_selection_button"):
            self.parent.zoom_to_selection_button.config(state=tk.DISABLED)
        if hasattr(self.parent, "pan_button"):
            self.parent.pan_button.config(state=tk.DISABLED)

        # Reset zoom/pan vars in parent if present
        if hasattr(self.parent, "zoomed_selection_coords"):
            self.parent.zoomed_selection_coords = None
        if hasattr(self.parent, "zoomed_in"):
            self.parent.zoomed_in = False

    def get_selection_mask(self, shape):
        mask = np.zeros(shape, dtype=np.uint8)

        if self.polygon_points:
            from image_processing_gui import create_mask
            h, w = shape
            normalized = [(x / w, y / h) for x, y in self.polygon_points]
            mask = create_mask(shape, normalized, normalize=True)
        elif self.rect_coords is not None and len(self.rect_coords) == 4:
            try:
                left, top, right, bottom = self.rect_coords
                mask[top:bottom, left:right] = 1
            except Exception as e:
                print(f"[DEBUG] Errore generazione mask da rect_coords: {e}")

        return mask