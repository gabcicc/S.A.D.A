import os

# Impostazioni del logo
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_CHANGES_PATH = os.path.join(SCRIPT_DIR, "images", "logo_changes.png")
LOGO_CHANGES_TRASP_PATH = os.path.join(SCRIPT_DIR, "images", "logo_changes_trasp.png")
LOGO_ISPC_PATH = os.path.join(SCRIPT_DIR, "images", "ispc_logo.png")
LOGO_ISPC_TRASP_PATH = os.path.join(SCRIPT_DIR, "images", "ispc_logo_white.png")

# Impostazioni dell'immagine
IMAGE_MAX_SIZE = (800, 600)

# Mappa dei colori
COLOR_MAP = {
    "Red": [255, 0, 0],
    "White": [255, 255, 255],
    "Blue": [0, 0, 255],
    "Green": [0, 255, 0],
    "Black": [0, 0, 0]
}

# Opzioni per il tipo di anomalia
ANOMALY_TYPES = ["Dark Pixels", "Bright Pixels"]

# Configurazione dei metodi di rilevamento anomalie
METHODS = ["Standard", "PCA", "K-means", "Isolation Forest", "DBSCAN", "ocSVM", "LOF", "RX Detector", "LISA" ]

# Parametri di default per SVM
ocSVM_KERNELS = ["linear", "poly", "rbf", "sigmoid"]  # I kernel supportati da SVM
ocSVM_DEFAULT_KERNEL = "rbf"  # Il kernel predefinito
ocSVM_DEFAULT_C = 1.0  # Il valore predefinito del parametro C per SVM

PCA_COMPONENTS = 1  # Default a 1, puoi cambiarlo in base alle tue esigenze

# Bande supportate per multispettrale
MULTISPECTRAL_BANDS = ["Band 1", "Band 2", "Band 3", "Band 4", "Band 5", "Band 6", "Band 7", "Band 8"]