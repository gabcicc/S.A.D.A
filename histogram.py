import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageOps
import cv2
from tooltip import Tooltip, get_tooltip



def open_histogram_window(parent, image, original_image_pil, update_image_callback, pixel_mode=False, selection_mask=None):
    """
    Apre una finestra Toplevel che visualizza l'istogramma di 'image'.
    Se pixel_mode=True, i pulsanti di trasformazione e reset verranno disabilitati.
    """
    if image is None:
        return

    hist_win = tk.Toplevel(parent)
    if pixel_mode:
        hist_win.title("Pixel Histogram")
    else:
        hist_win.title("Histogram")

    # Crea la figura matplotlib e disegna l'istogramma
    fig, ax = plt.subplots(figsize=(6,4))
    draw_histogram(ax, image)

    canvas = FigureCanvasTkAgg(fig, master=hist_win)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Frame per i pulsanti
    btn_frame = tk.Frame(hist_win)
    btn_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # Pulsanti di trasformazione
    stretch_btn = tk.Button(
        btn_frame,
        text="Stretch",
        command=lambda: apply_histogram_stretch(parent, image, ax, canvas, update_image_callback, selection_mask=selection_mask),
    )
    stretch_btn.pack(side=tk.LEFT, padx=5, pady=5)

    equalize_btn = tk.Button(
        btn_frame,
        text="Equalize",
        command=lambda: apply_histogram_equalization(parent, image, ax, canvas, update_image_callback, selection_mask=selection_mask),
    )
    equalize_btn.pack(side=tk.LEFT, padx=5, pady=5)

    gamma_btn = tk.Button(
        btn_frame,
        text="Gamma Correction",
        command=lambda: apply_gamma_correction(parent, image, ax, canvas, update_image_callback, selection_mask=selection_mask),
    )
    gamma_btn.pack(side=tk.LEFT, padx=5, pady=5)

    log_btn = tk.Button(
        btn_frame,
        text="Log Transform",
        command=lambda: apply_log_transformation(parent, image, ax, canvas, update_image_callback, selection_mask=selection_mask),
    )
    log_btn.pack(side=tk.LEFT, padx=5, pady=5)

    clahe_btn = tk.Button(
        btn_frame,
        text="Adaptive Equalization",
        command=lambda: apply_adaptive_equalization(parent, image, ax, canvas, update_image_callback, selection_mask=selection_mask),
    )
    clahe_btn.pack(side=tk.LEFT, padx=5, pady=5)

    save_histo_btn = tk.Button(
        btn_frame,
        text="Save Histo",
        command=lambda: save_histogram_plot(hist_win, fig),
    )
    save_histo_btn.pack(side=tk.LEFT, padx=5, pady=5)

    reset_btn = tk.Button(
        btn_frame,
        text="Reset",
        command=lambda: apply_histogram_reset(parent, ax, canvas, update_image_callback, selection_mask=selection_mask),
    )
    reset_btn.pack(side=tk.LEFT, padx=5, pady=5)

    # Se pixel_mode=True, disabilita i pulsanti di trasformazione e reset
    if pixel_mode:
        stretch_btn.config(state=tk.DISABLED)
        equalize_btn.config(state=tk.DISABLED)
        gamma_btn.config(state=tk.DISABLED)
        log_btn.config(state=tk.DISABLED)
        clahe_btn.config(state=tk.DISABLED)
        reset_btn.config(state=tk.DISABLED)

    return hist_win


def draw_histogram(ax, image):
    ax.clear()
    if len(image.shape) == 2 or image.shape[2] == 1:
        # Grayscale
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
        ax.plot(bins[:-1], hist, color='black')
        ax.set_title("Grayscale Histogram")
    else:
        colors = ['red', 'green', 'blue']
        for i, col in enumerate(colors):
            hist, bins = np.histogram(image[:, :, i].flatten(), bins=256, range=[0, 256])
            ax.plot(bins[:-1], hist, color=col, label=f"{col} channel")
        ax.set_title("RGB Histogram")
        ax.legend()
    ax.set_xlim([0, 256])


def apply_histogram_stretch(parent, image, ax, canvas, update_image_callback, selection_mask=None):
    # Ricalcola sempre la maschera di selezione corrente
    selection_mask = parent.image_selection.get_selection_mask(image.shape[:2])
    if len(image.shape) == 2 or image.shape[2] == 1:
        img = image.astype(np.float32)
        min_val = img.min()
        max_val = img.max()
        if max_val - min_val > 0:
            stretched = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            stretched = image.copy()
    else:
        img = image.astype(np.float32)
        stretched = np.empty_like(img, dtype=np.uint8)
        for i in range(3):
            channel = img[:, :, i]
            min_val = channel.min()
            max_val = channel.max()
            if max_val - min_val > 0:
                stretched[:, :, i] = ((channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                stretched[:, :, i] = image[:, :, i]

    result = stretched
    # Apply selection mask if provided
    if selection_mask is not None and np.any(selection_mask) and selection_mask.shape[:2] == result.shape[:2]:
        base_image = parent.image.copy()
        if base_image.ndim == 2:
            base_image = base_image[..., np.newaxis]
        if result.ndim == 2:
            result = result[..., np.newaxis]
        base_image[selection_mask == 1] = result[selection_mask == 1]
        result = base_image

    update_image_callback(result)
    draw_histogram(ax, result)
    canvas.draw()
    parent.save_button.config(state=tk.NORMAL)


def apply_histogram_equalization(parent, image, ax, canvas, update_image_callback, selection_mask=None):
    # Ricalcola sempre la maschera di selezione corrente
    selection_mask = parent.image_selection.get_selection_mask(image.shape[:2])
    if len(image.shape) == 2 or image.shape[2] == 1:
        pil_img = Image.fromarray(image.squeeze(), mode='L')
        eq_img = ImageOps.equalize(pil_img)
        eq_arr = np.array(eq_img)
        if eq_arr.ndim == 2:
            eq_arr = eq_arr[..., np.newaxis]
    else:
        channels = []
        for i in range(3):
            pil_channel = Image.fromarray(image[:, :, i], mode='L')
            eq_channel = ImageOps.equalize(pil_channel)
            channels.append(np.array(eq_channel))
        eq_arr = np.stack(channels, axis=2)

    result = eq_arr
    # Apply selection mask if provided
    if selection_mask is not None and np.any(selection_mask) and selection_mask.shape[:2] == result.shape[:2]:
        base_image = parent.image.copy()
        if base_image.ndim == 2:
            base_image = base_image[..., np.newaxis]
        if result.ndim == 2:
            result = result[..., np.newaxis]
        base_image[selection_mask == 1] = result[selection_mask == 1]
        result = base_image

    update_image_callback(result)
    draw_histogram(ax, result)
    canvas.draw()
    parent.save_button.config(state=tk.NORMAL)


def apply_gamma_correction(parent, image, ax, canvas, update_image_callback, selection_mask=None):
    # Ricalcola sempre la maschera di selezione corrente
    selection_mask = parent.image_selection.get_selection_mask(image.shape[:2])
    # Valore di gamma fisso; puoi in futuro aggiungere uno slider dedicato
    gamma = 0.5  # Per gamma < 1 l'immagine diventa più luminosa; 1.0 = nessuna correzione
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    if len(image.shape) == 2 or image.shape[2] == 1:
        corrected = cv2.LUT(image.squeeze(), table)
        corrected = corrected[..., np.newaxis]
    else:
        corrected = cv2.LUT(image, table)

    result = corrected
    # Apply selection mask if provided
    if selection_mask is not None and np.any(selection_mask) and selection_mask.shape[:2] == result.shape[:2]:
        base_image = parent.image.copy()
        if base_image.ndim == 2:
            base_image = base_image[..., np.newaxis]
        if result.ndim == 2:
            result = result[..., np.newaxis]
        base_image[selection_mask == 1] = result[selection_mask == 1]
        result = base_image

    update_image_callback(result)
    draw_histogram(ax, result)
    canvas.draw()
    parent.save_button.config(state=tk.NORMAL)


def apply_log_transformation(parent, image, ax, canvas, update_image_callback, selection_mask=None):
    # Ricalcola sempre la maschera di selezione corrente
    selection_mask = parent.image_selection.get_selection_mask(image.shape[:2])
    img = image.astype(np.float32)
    max_val = img.max()
    c = 255 / np.log(1 + max_val)
    log_transformed = c * np.log(1 + img)
    log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)

    result = log_transformed
    # Apply selection mask if provided
    if selection_mask is not None and np.any(selection_mask) and selection_mask.shape[:2] == result.shape[:2]:
        base_image = parent.image.copy()
        if base_image.ndim == 2:
            base_image = base_image[..., np.newaxis]
        if result.ndim == 2:
            result = result[..., np.newaxis]
        base_image[selection_mask == 1] = result[selection_mask == 1]
        result = base_image

    update_image_callback(result)
    draw_histogram(ax, result)
    canvas.draw()
    parent.save_button.config(state=tk.NORMAL)


def apply_adaptive_equalization(parent, image, ax, canvas, update_image_callback, selection_mask=None):
    # Ricalcola sempre la maschera di selezione corrente
    selection_mask = parent.image_selection.get_selection_mask(image.shape[:2])
    # Utilizza CLAHE: per immagini grayscale e per RGB
    if len(image.shape) == 2 or image.shape[2] == 1:
        if image.ndim == 3:
            gray = image.squeeze()
        else:
            gray = image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        eq = eq[..., np.newaxis]
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        eq = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    result = eq
    # Apply selection mask if provided
    if selection_mask is not None and np.any(selection_mask) and selection_mask.shape[:2] == result.shape[:2]:
        base_image = parent.image.copy()
        if base_image.ndim == 2:
            base_image = base_image[..., np.newaxis]
        if result.ndim == 2:
            result = result[..., np.newaxis]
        base_image[selection_mask == 1] = result[selection_mask == 1]
        result = base_image

    update_image_callback(result)
    draw_histogram(ax, result)
    canvas.draw()
    parent.save_button.config(state=tk.NORMAL)


def apply_histogram_reset(parent, ax, canvas, update_image_callback, selection_mask=None):
    # Ricalcola sempre la maschera di selezione corrente
    selection_mask = parent.image_selection.get_selection_mask(parent.image.shape[:2])
    """
    Riporta l'immagine al suo stato originale, rispettando la banda selezionata e
    applicando lo stretching corretto (percentile o lineare).
    """
    import numpy as np
    import re

    original = np.copy(parent.original_image_np)

    selected_band = parent.band.get()

    # Multispettrale
    if original.ndim == 3 and original.shape[2] >= 5:
        if selected_band == "RGB":
            pseudo_rgb = original[..., [2, 1, 0]].astype(np.float32)
            pseudo_rgb[pseudo_rgb < -9999] = np.nan
            low = np.nanpercentile(pseudo_rgb, 2)
            high = np.nanpercentile(pseudo_rgb, 98)
            if high - low < 1e-6:
                arr_f = np.zeros_like(pseudo_rgb)
            else:
                arr_f = (pseudo_rgb - low) / (high - low) * 255.0
            arr_f = np.nan_to_num(arr_f, nan=0)
            result = np.clip(arr_f, 0, 255).astype('uint8')
        else:
            # Estrai indice banda da nome (es. "Band 3" → index 2)
            match = re.search(r"Band\s+(\d+)", selected_band)
            band_index = int(match.group(1)) - 1 if match else 0
            band = original[..., band_index].astype(np.float32)
            band[band < -9999] = np.nan
            low = np.nanpercentile(band, 2)
            high = np.nanpercentile(band, 98)
            if high - low < 1e-6:
                arr_f = np.zeros_like(band)
            else:
                arr_f = (band - low) / (high - low) * 255.0
            arr_f = np.nan_to_num(arr_f, nan=0)
            band_stretched = np.clip(arr_f, 0, 255).astype('uint8')
            result = band_stretched[..., np.newaxis]
    else:
        # Immagine RGB classica o grayscale
        if selected_band == "RGB":
            band = original[..., :3] if original.ndim == 3 else original
        else:
            match = re.search(r"Band\s+(\d+)", selected_band)
            band_index = int(match.group(1)) - 1 if match else 0
            band = original[..., band_index] if original.ndim == 3 else original
        band = band.astype(np.float32)
        band_min, band_max = band.min(), band.max()
        if band_max - band_min > 0:
            arr_f = (band - band_min) / (band_max - band_min) * 255
        else:
            arr_f = np.zeros_like(band)
        arr_f = np.nan_to_num(arr_f, nan=0)
        result = np.clip(arr_f, 0, 255).astype('uint8')
        if result.ndim == 2:
            result = result[..., np.newaxis]

    # Apply selection mask if provided
    if selection_mask is not None and np.any(selection_mask) and selection_mask.shape[:2] == result.shape[:2]:
        base_image = parent.image.copy()
        if base_image.ndim == 2:
            base_image = base_image[..., np.newaxis]
        if result.ndim == 2:
            result = result[..., np.newaxis]
        base_image[selection_mask == 1] = result[selection_mask == 1]
        result = base_image

    update_image_callback(result)
    draw_histogram(ax, result)
    canvas.draw()

def save_histogram_plot(hist_win, fig):
    from tkinter import filedialog
    file_path = filedialog.asksaveasfilename(parent=hist_win, defaultextension=".png",
                                               filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if file_path:
        fig.savefig(file_path)