import tkinter as tk
import io
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from matplotlib import cm
import rasterio
from config import LOGO_CHANGES_TRASP_PATH, LOGO_ISPC_TRASP_PATH
from tooltip import Tooltip, get_tooltip

class PCAEditor(tk.Toplevel):
    def __init__(self, root, main_gui, pca_result, shape, mode, colormap,
                 selection_mask=None, selection_bbox=None, explained_variance_ratio=None):
        super().__init__(root)  # root è il widget Tk principale (ImageAnalyzer)
        screen_width = int(self.winfo_screenwidth() * 0.9)
        screen_height = int(self.winfo_screenheight() * 0.9)
        self.geometry(f"{screen_width}x{screen_height}")
        self.main_gui = main_gui
        self.parent = root
        self.pca_result = pca_result
        self.explained_variance_ratio = explained_variance_ratio
        self.h, self.w = shape
        self.mode = mode
        self.colormap = colormap
        self.crs = getattr(root, "crs", None)
        self.geotransform = getattr(root, "geotransform", None)

        # ✅ Selezione passata direttamente come parametro
        self.selection_mask = selection_mask
        self.selection_bbox = selection_bbox

        # self.explained_variance_ratio is already set above with correct default

        self.title("PCA Viewer")

        # === Frame dei loghi in alto ===
        self.logo_frame = tk.Frame(self)
        self.logo_frame.pack(fill=tk.X)

        self.logo_label_left = tk.Label(self.logo_frame)
        self.logo_label_left.pack(side=tk.LEFT, padx=10)

        self.logo_label_right = tk.Label(self.logo_frame)
        self.logo_label_right.pack(side=tk.RIGHT, padx=10)

        self.load_logos()

        # Ottieni dimensioni canvas principale
        root.update_idletasks()
        canvas_width = root.canvas.winfo_width()
        canvas_height = root.canvas.winfo_height()

        # Margine extra per ticks e coordinate
        extra_margin = 120
        adjusted_height = canvas_height + extra_margin

        if canvas_width > 1 and canvas_height > 1:
            self.geometry(f"{canvas_width}x{adjusted_height}")

        # Usa l'altezza ridotta per scalare l'immagine tenendo conto del margine
        self.display_width = canvas_width
        self.display_height = canvas_height - extra_margin

        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        scroll_frame = tk.Frame(self)
        scroll_frame.pack(fill=tk.X, padx=10)

        x_scroll = tk.Scrollbar(scroll_frame, orient=tk.HORIZONTAL)
        x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.control_canvas = tk.Canvas(scroll_frame, height=60, xscrollcommand=x_scroll.set)
        self.control_canvas.pack(side=tk.TOP, fill=tk.X, expand=True)

        x_scroll.config(command=self.control_canvas.xview)

        control_frame = tk.Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=control_frame, anchor='nw')
        control_frame.bind("<Configure>",
                           lambda e: self.control_canvas.config(scrollregion=self.control_canvas.bbox("all")))

        self.update_idletasks()
        self.control_canvas.config(scrollregion=self.control_canvas.bbox("all"))

        self.label = tk.Label(control_frame, text="")
        self.label.pack(side=tk.LEFT, padx=10)

        tk.Button(control_frame, text="Previous", command=self.show_prev).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Next", command=self.show_next).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Save current PCA", command=self.save_current_pca).pack(side=tk.RIGHT, padx=5)
        tk.Button(control_frame, text="Save all PCA", command=self.save_all_pca).pack(side=tk.RIGHT, padx=5)

        self.save_plot_btn = tk.Button(control_frame, text="Save Plot", command=self.save_plot)
        self.save_plot_btn.pack(side=tk.RIGHT, padx=5)
        if not (self.crs and self.geotransform):
            self.save_plot_btn.config(state=tk.DISABLED)
        Tooltip(self.save_plot_btn, get_tooltip("save_plot_button"))

        self.load_sada_btn = tk.Button(control_frame, text="Load in S.A.D.A.", command=self.load_in_sada)
        self.load_sada_btn.pack(side=tk.RIGHT, padx=5)
        Tooltip(self.load_sada_btn, "Load the current PCA component into the main S.A.D.A. canvas as a new image band.")


        self.colormap_var = tk.StringVar(value=self.colormap)
        colormap_menu = ttk.Combobox(control_frame, textvariable=self.colormap_var,
                                     values=["Grayscale", "Viridis", "Plasma", "Turbo", "Cividis"],
                                     state="readonly")
        colormap_menu.set("Grayscale")
        colormap_menu.pack(side=tk.RIGHT, padx=10)
        colormap_menu.bind("<<ComboboxSelected>>", self.on_colormap_change)

        self.pca_images = []
        self.current_index = 0

        self.update_idletasks()  # assicura che le dimensioni siano reali
        self.prepare_images()
        self.after(150, self.display_current_image)
        self.status_label = tk.Label(self, text="", bd=1, relief=tk.SUNKEN, anchor="w")
        self.status_label.pack(fill=tk.X, side=tk.BOTTOM)

        # Bind resize event to update image display
        self.bind("<Configure>", self.on_main_resize)


    def prepare_images(self):
        self.pca_images.clear()
        n_components = self.pca_result.shape[1]

        for i in range(n_components):
            try:
                component = self.pca_result[:, i].reshape((self.h, self.w))
            except Exception as e:
                print(f"[ERROR] reshape fallita per componente {i + 1}: {e}")
                continue

            c_min = component.min()
            c_max = component.max()
            if c_max - c_min > 1e-8:
                component = (component - c_min) / (c_max - c_min)
            else:
                component = np.zeros_like(component)

            component = np.clip(component * 255, 0, 255).astype(np.uint8)
            print(
                f"[DEBUG] PCA Component {i + 1}: min={component.min()}, max={component.max()}, mean={component.mean()}")

            selected_colormap = self.colormap_var.get()
            if selected_colormap == "Grayscale":
                rgb = np.stack([component] * 3, axis=-1)
            else:
                cmap = cm.get_cmap(selected_colormap.lower())  # es. 'viridis', 'plasma', etc.
                rgb = (cmap(component / 255.0)[:, :, :3] * 255).astype(np.uint8)

            pil_img = Image.fromarray(rgb)
            canvas_w = self.display_width
            canvas_h = self.display_height
            img_w, img_h = pil_img.size
            scale_factor = min(canvas_w / img_w, canvas_h / img_h, 1.0)
            self.current_scale_factor = scale_factor
            new_size = (int(img_w * scale_factor), int(img_h * scale_factor))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)
            self.pca_images.append(pil_img)

    def display_current_image(self):
        self.canvas.delete("all")
        image = self.pca_images[self.current_index]
        self.tk_image = ImageTk.PhotoImage(image)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_w, img_h = image.size

        x = (canvas_w - img_w) // 2
        y = (canvas_h - img_h) // 2
        self.canvas_image = self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)

        try:
            var = self.explained_variance_ratio[self.current_index] * 100
            self.label.config(
                text=f"PCA Component {self.current_index + 1} / {len(self.pca_images)} – Variance: {var:.2f}%"
            )
        except Exception as e:
            print(f"[DEBUG] Could not access explained variance ratio: {e}")
            self.label.config(
                text=f"PCA Component {self.current_index + 1} / {len(self.pca_images)}"
            )

        if self.crs is not None and self.geotransform is not None:
            from rasterio.transform import xy
            from pyproj import Transformer, CRS

            current_crs = CRS(self.crs)
            transformer_to_geographic = Transformer.from_crs(current_crs, "EPSG:4326", always_xy=True)

            if hasattr(self, 'selection_bbox') and self.selection_bbox is not None:
                top, left, bottom, right = self.selection_bbox[1], self.selection_bbox[0], self.selection_bbox[3], \
                self.selection_bbox[2]
                corners = [(top, left), (top, right), (bottom, left), (bottom, right)]
                w_sel = right - left
                h_sel = bottom - top
            else:
                corners = [(0, 0), (0, self.w), (self.h, 0), (self.h, self.w)]
                w_sel = self.w
                h_sel = self.h

            lats = []
            lons = []
            for row_c, col_c in corners:
                x_c, y_c = xy(self.geotransform, row_c, col_c)
                lon_c, lat_c = transformer_to_geographic.transform(x_c, y_c)
                lats.append(lat_c)
                lons.append(lon_c)

            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)

            def latlon_to_canvas(lat, lon):
                x_g, y_g = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True).transform(lon, lat)
                col = (x_g - self.geotransform.c) / self.geotransform.a
                row = (y_g - self.geotransform.f) / self.geotransform.e
                col -= left if hasattr(self, 'selection_bbox') and self.selection_bbox is not None else 0
                row -= top if hasattr(self, 'selection_bbox') and self.selection_bbox is not None else 0
                x_canvas = x + (col * (img_w / w_sel))
                y_canvas = y + (row * (img_h / h_sel))
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
                self.canvas.create_line(x, y_c, x - tick_len, y_c, fill="black")
                self.canvas.create_text(x - tick_len - 40, y_c, text=lat_dms, fill="black", font=("Arial", 12, "bold"))
                self.canvas.create_line(x + img_w, y_c, x + img_w + tick_len, y_c, fill="black")
                self.canvas.create_text(x + img_w + tick_len + 40, y_c, text=lat_dms, fill="black",
                                        font=("Arial", 12, "bold"))

            for lon_val in lon_ticks:
                x_c, y_c = latlon_to_canvas((lat_min + lat_max) / 2, lon_val)
                lon_dms = to_dms(lon_val, is_lat=False)
                self.canvas.create_line(x_c, y, x_c, y - tick_len, fill="black")
                self.canvas.create_text(x_c, y - tick_len - 10, text=lon_dms, fill="black", font=("Arial", 12, "bold"))
                self.canvas.create_line(x_c, y + img_h, x_c, y + img_h + tick_len, fill="black")
                self.canvas.create_text(x_c, y + img_h + tick_len + 10, text=lon_dms, fill="black",
                                        font=("Arial", 12, "bold"))

            north_x = x + img_w + 40
            north_y = y + 60
            self.canvas.create_polygon(
                [(north_x, north_y), (north_x - 10, north_y + 30), (north_x + 10, north_y + 30)],
                fill="black", outline="black"
            )
            self.canvas.create_text(north_x, north_y - 15, text="N", font=("Arial", 15, "bold"), fill="black")

            pixel_width = abs(self.geotransform.a)
            scale_factor = getattr(self, "current_scale_factor", 1.0)
            pixel_width_m_displayed = pixel_width / scale_factor

            gui_width = self.winfo_width()
            scale_length_px_gui = int((gui_width / 20) * 1.5)
            proposed_length_m = scale_length_px_gui * pixel_width_m_displayed
            base_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
            chosen_scale_m = min(base_values, key=lambda x: abs(x - proposed_length_m))
            scalebar_length_px = int(chosen_scale_m / pixel_width_m_displayed)

            segment_count = 5
            segment_px = scalebar_length_px // segment_count

            scalebar_x = x + img_w + 20
            scalebar_y = y + img_h - 50

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

            self.canvas.create_rectangle(
                x - 2, y - 2, x + img_w + 2, y + img_h + 2,
                outline="black", width=2
            )

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()

    def show_next(self):
        if self.current_index < len(self.pca_images) - 1:
            self.current_index += 1
            self.display_current_image()

    def on_colormap_change(self, event=None):
        self.colormap = self.colormap_var.get()
        self.prepare_images()
        self.display_current_image()

    def save_current_pca(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".tif",
                                                 filetypes=[("GeoTIFF", "*.tif"), ("PNG", "*.png")])
        if not file_path:
            return

        i = self.current_index
        component = self.pca_result[:, i].reshape((self.h, self.w))
        colormap = self.colormap_var.get()

        if self.selection_mask is not None and self.selection_bbox is not None:
            left, top, right, bottom = self.selection_bbox
            mask_cropped = self.selection_mask[top:bottom, left:right]
            valid_vals = component[mask_cropped == 1]
            vmin, vmax = valid_vals.min(), valid_vals.max()
        else:
            vmin, vmax = component.min(), component.max()

        if vmax - vmin > 1e-8:
            component = (component - vmin) / (vmax - vmin)
        else:
            component = np.zeros_like(component)

        component = np.clip(component, 0, 1)

        if colormap == "Grayscale":
            gray_array = (component * 255).astype(np.uint8)
            if file_path.endswith(".tif") and self.crs and self.geotransform:
                with rasterio.open(file_path, 'w', driver='GTiff', height=gray_array.shape[0],
                                   width=gray_array.shape[1],
                                   count=1, dtype=gray_array.dtype, crs=self.crs, transform=self.geotransform) as dst:
                    dst.write(gray_array, 1)
            else:
                Image.fromarray(gray_array).save(file_path.replace(".tif", ".png"))
        else:
            cmap = cm.get_cmap(colormap.lower())
            rgb_array = (cmap(component)[:, :, :3] * 255).astype(np.uint8)
            if file_path.endswith(".tif") and self.crs and self.geotransform:
                with rasterio.open(file_path, 'w', driver='GTiff', height=rgb_array.shape[0], width=rgb_array.shape[1],
                                   count=3, dtype=rgb_array.dtype, crs=self.crs, transform=self.geotransform) as dst:
                    for b in range(3):
                        dst.write(rgb_array[:, :, b], b + 1)
            else:
                Image.fromarray(rgb_array).save(file_path.replace(".tif", ".png"))

    def save_all_pca(self):
        dir_path = filedialog.askdirectory()
        if not dir_path:
            return

        for i in range(self.pca_result.shape[1]):
            component = self.pca_result[:, i].reshape((self.h, self.w))
            colormap = self.colormap_var.get()

            if self.selection_mask is not None and self.selection_bbox is not None:
                left, top, right, bottom = self.selection_bbox
                mask_cropped = self.selection_mask[top:bottom, left:right]
                valid_vals = component[mask_cropped == 1]
                vmin, vmax = valid_vals.min(), valid_vals.max()
            else:
                vmin, vmax = component.min(), component.max()

            if vmax - vmin > 1e-8:
                component = (component - vmin) / (vmax - vmin)
            else:
                component = np.zeros_like(component)

            component = np.clip(component, 0, 1)

            path = f"{dir_path}/pca_component_{i + 1}.tif"

            if colormap == "Grayscale":
                gray_array = (component * 255).astype(np.uint8)
                if self.crs and self.geotransform:
                    with rasterio.open(path, 'w', driver='GTiff', height=gray_array.shape[0], width=gray_array.shape[1],
                                       count=1, dtype=gray_array.dtype, crs=self.crs,
                                       transform=self.geotransform) as dst:
                        dst.write(gray_array, 1)
                else:
                    Image.fromarray(gray_array).save(path.replace(".tif", ".png"))
            else:
                cmap = cm.get_cmap(colormap.lower())
                rgb_array = (cmap(component)[:, :, :3] * 255).astype(np.uint8)
                if self.crs and self.geotransform:
                    with rasterio.open(path, 'w', driver='GTiff', height=rgb_array.shape[0], width=rgb_array.shape[1],
                                       count=3, dtype=rgb_array.dtype, crs=self.crs,
                                       transform=self.geotransform) as dst:
                        for b in range(3):
                            dst.write(rgb_array[:, :, b], b + 1)
                else:
                    Image.fromarray(rgb_array).save(path.replace(".tif", ".png"))


    def save_plot(self):
        self.canvas.update()

        if not hasattr(self, "canvas_image"):
            self.status_label.config(text="Error: no canvas_image found.")
            return

        bbox = self.canvas.bbox(self.canvas_image)
        if bbox is None:
            self.status_label.config(text="Error: no image to save.")
            return

        x1, y1, x2, y2 = bbox
        top_margin = 120
        side_margin = 120
        bottom_margin = 60

        x1 = max(0, x1 - side_margin)
        y1 = max(0, y1 - top_margin)

        scalebar_extra = 200
        if hasattr(self, "display_w") and hasattr(self, "zoom_factor") and \
                hasattr(self.parent, "geotransform") and self.parent.geotransform is not None:
            try:
                pixel_size = abs(self.parent.geotransform.a)
                target_length_m = 100
                scale_px = int(target_length_m / pixel_size * self.zoom_factor)
                scalebar_extra = scale_px // 2 + 100
            except Exception:
                pass

        x2 += side_margin + scalebar_extra
        y2 += bottom_margin

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
            self.status_label.config(text=f"Plot saved to {file_path}")

    def load_in_sada(self):
        analyzer = self.parent  # ImageAnalyzer (root)
        i = self.current_index
        component = self.pca_result[:, i].reshape((self.h, self.w))

        if self.selection_mask is not None and self.selection_bbox is not None:
            left, top, right, bottom = self.selection_bbox
            mask_cropped = self.selection_mask[top:bottom, left:right]
            masked = component[mask_cropped == 1]
            vmin, vmax = masked.min(), masked.max()
        else:
            vmin, vmax = component.min(), component.max()

        if vmax - vmin > 1e-8:
            norm = (component - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(component)

        norm = np.clip(norm, 0, 1)
        array = (norm * 255).astype(np.uint8)
        new_band = array[..., np.newaxis]

        chosen_name = f"PCA{self.current_index + 1}"

        # ✅ Usa la stessa funzione già utilizzata dal Raster Calculator
        analyzer.update_image_from_raster_calc(new_band, band_name=chosen_name)

        self.destroy()

    def load_logos(self):
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
    def on_main_resize(self, event):
        self.display_current_image()