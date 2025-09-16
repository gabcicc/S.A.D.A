# Dizionario dei testi tooltip
import tkinter as tk

TOOLTIPS = {
    "method_combobox": (
        "Select the method to analyze anomalies. Options include:\n"
        "- Standard: Basic anomaly detection using thresholds.\n"
        "- PCA: Principal Component Analysis to detect anomalies based on important features.\n"
        "- K-means: Clustering algorithm that identifies outliers as anomalies.\n"
        "- Isolation Forest: Detects anomalies by isolating data points through random partitioning.\n"
        "- DBSCAN: Density-Based Spatial Clustering with outliers as unclustered points.\n"
        "- ocSVM: one-class Support Vector Machine separates anomalies from normal data of background.\n"
        "- LOF: Local Outlier Factor identifies anomalies based on the local density deviation of a given data point with respect to its neighbors.\n"
        "- RX Detector: Calculates anomalies by measuring the Mahalanobis distance of each pixel from the global mean and covariance in the spectral space.\n"
        "- LISA: Local Indicators of Spatial Association provides Moran's, Getis-Ord and INDICATE's index"
    ),
    "pca_slider": "Set the number of PCA components for dimensionality reduction.",
    "eps_slider": (
        "Set the maximum distance between samples for them to be considered in the same neighborhood "
        "(DBSCAN parameter)."
    ),
    "min_samples_slider": (
        "Set the number of samples in a neighborhood for a point to be considered a core point "
        "(DBSCAN parameter)."
    ),
    "ocSVM_kernel": (
        "Select the kernel type for ocSVM. Options include:\n"
        "- linear\n"
        "- polynomial\n"
        "- radial basis function (rbf)\n"
        "- sigmoid"
    ),
    "ocSVM_c_slider": (
        "Set the regularization parameter for ocSVM. Smaller values create smoother decision boundaries, "
        "while larger values focus on classifying all training points correctly."
    ),
    "threshold_slider": (
        "Set the threshold value to control the sensitivity of anomaly detection. "
        "Higher values may detect fewer anomalies."
    ),
    "anomaly_combobox": (
        "Select the type of anomaly to detect in the Standard method:\n"
        "- Darker Pixels: Areas darker than the threshold.\n"
        "- Bright Pixels: Areas brighter than the threshold."
    ),
    "color_combobox": "Select the color to highlight detected anomalies in the image.",
    "brightness_slider": (
        "Adjust the brightness of the image before anomaly detection. Higher values increase brightness."
    ),
    "contrast_slider": (
        "Adjust the contrast of the image before anomaly detection. Higher values increase contrast."
    ),
    "histogram_stretch": "Stretches the contrast by mapping the minimum and maximum pixel values to 0 and 255.",
    "histogram_equalize": "Equalizes the histogram so that the intensity values are more evenly distributed.",
    "histogram_gamma": "Applies a gamma correction (e.g., gamma < 1 brightens the image, gamma > 1 darkens it).",
    "histogram_log": "Applies a logarithmic transformation to enhance details in darker areas.",
    "histogram_adaptive": (
        "Uses CLAHE (Adaptive Histogram Equalization) to improve local contrast "
        "without over-saturating the brighter areas."
    ),

    "auto_rgb": "Automatic assignement of B1=R,B2=G and B3=B.",
    "raster_ndvi": "Computes NDVI: (NIR - R) / (NIR + R). This index measures healthy green vegetation using chlorophyll absorption and reflectance. It’s robust but may saturate in dense vegetation.",
    "raster_gndvi": "Computes GNDVI: (NIR - G) / (NIR + G). Similar to NDVI but uses the green spectrum; more sensitive to chlorophyll concentration.",
    "raster_bndvi": "Computes BNDVI: (NIR - B) / (NIR + B). Requires near-infrared (NIR) and green (B).",
    "raster_sr": "Computes SR: NIR / R. A simple ratio of peak vegetation reflectance to deepest chlorophyll absorption; effective but may saturate in dense vegetation.",
    "raster_evi": "Computes EVI: 2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1). Developed for MODIS to improve NDVI in high-LAI areas; uses blue reflectance to reduce soil and atmospheric effects.",
    "raster_gemi": "Computes GEMI: (2 * (NIR^2 - R^2) + 1.5 * NIR + 0.5 * R) / (NIR + R + 0.5). Non-linear NDVI variant less sensitive to atmosphere; not recommended for sparse or moderately vegetated areas due to soil influence.",
    "raster_osavi": "Computes OSAVI: (NIR - R) / (NIR + R + 0.16).Modified SAVI using a fixed 0.16 soil adjustment; ideal for sparse vegetation with visible soil, offering better sensitivity than SAVI.",
    "raster_vari": "Computes VARI: (G - R) / (G + R - B). VARI is designed to mitigating illumination differences and atmospheric effects. It is ideal for RGB images .",
    "raster_tgi": "Computes TGI: -0.5 * (190 * (R - G) - 120 * (R - B)). TGI indicates chlorophyll content: positive for green vegetation, negative for features like red soils.",
    "raster_add_band": "Add the computed index as a new image band, available for selection and analysis (e.g., PCA).",
    "lisa_index_combobox": (
            "Select which LISA index to visualize:\n"
         " Moran’s I: detects clusters of spatial autocorrelation (local similarity or dissimilarity).\n"
        "- Getis-Ord Gᵢ: identifies hot and cold spots based on local intensity.\n"
        "- INDIC (Moran ∩ Getis): highlights only areas flagged as spatial outliers by both Moran's I and Getis-Ord G*..\n"
    ),
    "save_plot_button": "Save the current view as a PNG plot, including scale bar, north arrow and geographic coordinates.",
}

# Funzione per ottenere il tooltip dal dizionario

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, _cx, _cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", foreground="#000000",
                         relief=tk.SOLID, borderwidth=1,
                         font=("tahoma", "8", "normal"))
        label.pack(ipadx=5, ipady=5)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()
def get_tooltip(key):
    """
    Ottieni il testo del tooltip per una determinata chiave.

    :param key: Chiave associata al tooltip (es: "method_combobox").
    :return: Testo del tooltip (stringa).
    """
    return TOOLTIPS.get(key, "No tooltip available for this item.")