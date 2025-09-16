import numpy as np
import numpy.linalg as la
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM

def create_mask(size, points, normalize=False):
    if normalize:
        points = [(int(x * size[0]), int(y * size[1])) for x, y in points]
    mask = Image.new('L', size, 0)
    ImageDraw.Draw(mask).polygon(points, outline=1, fill=1)
    return np.array(mask)

def find_anomalies(image, threshold, anomaly_type, method="Standard", *args):
    # image: np.ndarray (H, W, C)
    if method == "PCA":
        n_components = args[0]  # primo arg per PCA
        return find_anomalies_pca(image, threshold, n_components)
    elif method == "K-means":
        max_clusters = args[0] if len(args) > 0 else 10
        return find_anomalies_kmeans(image, threshold, max_clusters)
    elif method == "Isolation Forest":
        return find_anomalies_isolation_forest(image, threshold)
    elif method == "DBSCAN":
        return find_anomalies_dbscan(image, *args)
    elif method == "ocSVM":
        kernel = args[0] if len(args) > 0 else "rbf"
        C = args[1] if len(args) > 1 else 1.0
        return find_anomalies_ocsvm(image, kernel, C)
    elif method == "LOF":
        return find_anomalies_lof(image, threshold, *args)
    elif method == "RX Detector":
        return find_anomalies_rx(image, threshold, *args)
    elif method == "LISA":
        return find_anomalies_lisa(image, threshold, *args)
    else:
        # Metodo Standard
        if anomaly_type == "Darker Pixels":
            return image[:, :, 0] < threshold
        else:
            return image[:, :, 0] > threshold

def find_anomalies_pca(image, threshold, n_components):
    # image: np.ndarray (H,W,C)
    reshaped = image.reshape(-1, image.shape[2])

    max_components = min(reshaped.shape[0], reshaped.shape[1])
    n_components = min(n_components, max_components)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(reshaped)
    transformed_mean = np.mean(transformed, axis=1)

    anomalies = np.abs(transformed_mean - np.mean(transformed_mean)) > threshold
    anomaly_mask = anomalies.reshape(image.shape[:2])

    return anomaly_mask, pca.explained_variance_ratio_

def find_anomalies_kmeans(image, threshold, max_clusters=10):
    reshaped = image.reshape(-1, image.shape[2])

    distortions = []
    K = range(2, int(max_clusters) + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=0)
        kmeans.fit(reshaped)
        distortions.append(kmeans.inertia_)

    if len(distortions) > 1:
        diff = np.diff(distortions)
        diff_ratio = diff[:-1] / diff[1:]
        if len(diff_ratio) > 0:
            optimal_k = np.argmin(diff_ratio) + 2
        else:
            optimal_k = 2
    else:
        optimal_k = 2

    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=0)
    labels = kmeans.fit_predict(reshaped)
    cluster_centers = kmeans.cluster_centers_

    distances = np.linalg.norm(reshaped - cluster_centers[labels], axis=1)
    threshold_distance = np.percentile(distances, threshold)
    anomalies = distances > threshold_distance

    return anomalies.reshape(image.shape[:2])

def find_anomalies_isolation_forest(image, threshold):
    reshaped = image.reshape(-1, image.shape[2])
    isolation_forest = IsolationForest(contamination=0.1, random_state=0)
    isolation_forest.fit(reshaped)
    scores = isolation_forest.decision_function(reshaped)
    anomalies = scores < np.percentile(scores, 100 - threshold)
    return anomalies.reshape(image.shape[:2])

def find_anomalies_dbscan(image, eps=0.5, min_samples=5, apply_pca=True):
    reshaped = image.reshape(-1, image.shape[2])

    if apply_pca:
        max_components = min(reshaped.shape[0], reshaped.shape[1])
        n_components = min(10, max_components)
        if n_components > 1:
            pca = PCA(n_components=n_components)
            reshaped = pca.fit_transform(reshaped)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(reshaped)
    anomalies = labels == -1
    return anomalies.reshape(image.shape[:2])

def find_anomalies_ocsvm(image, kernel="rbf", C=1.0, sampling_rate=0.1):
    if kernel not in ["linear", "poly", "rbf", "sigmoid"]:
        raise ValueError(f"Invalid kernel: {kernel}. Must be one of ['linear', 'poly', 'rbf', 'sigmoid'].")

    reshaped = image.reshape(-1, image.shape[2])
    sample_size = int(reshaped.shape[0] * sampling_rate)
    indices = np.random.choice(reshaped.shape[0], sample_size, replace=False)
    sampled_reshaped = reshaped[indices]

    svm = OneClassSVM(kernel=kernel, nu=0.1, gamma='auto')
    svm.fit(sampled_reshaped)
    predictions = svm.predict(sampled_reshaped)

    anomalies = np.full(reshaped.shape[0], False)
    anomalies[indices] = (predictions == -1)

    return anomalies.reshape(image.shape[:2])


def find_anomalies_lof(image, threshold, *args):
    """
    Calcola le anomalie usando Local Outlier Factor (LOF).

    :param image: np.ndarray (H,W,C)
    :param threshold: percentile su LOF scores per decidere outlier (default=80)
    :param *args: argomenti extra (es. n_neighbors) se passati
    :return: maschera booleana (H, W) con True per i pixel anomali
    """
    import numpy as np
    from sklearn.neighbors import LocalOutlierFactor

    # Imposta un default per n_neighbors, es. 50, e proteggi da valori non validi
    n_neighbors = 50
    if len(args) > 0:
        try:
            n = int(args[0])
            if n > 0:
                n_neighbors = n
            else:
                print(f"[WARN] n_neighbors passato = {n} non valido. Uso default {n_neighbors}")
        except Exception as e:
            print(f"[WARN] Errore nel parsing di n_neighbors: {e}. Uso default {n_neighbors}")

    H, W, C = image.shape
    reshaped = image.reshape(-1, C).astype(float)

    # Aggiungi un piccolo jitter per "rompere" i duplicati
    epsilon = 1e-3
    reshaped += np.random.normal(0, epsilon, reshaped.shape)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
    labels = lof.fit_predict(reshaped)
    negative_factor = -lof.negative_outlier_factor_

    cutoff = np.percentile(negative_factor, threshold)
    anomalies = negative_factor > cutoff
    return anomalies.reshape(H, W)


def find_anomalies_rx(image, threshold, *args):
    """
    RX Detector: calcola la Mahalanobis distance di ogni pixel dalla media/cov globale,
    e prende come outlier chi supera percentile 'threshold'.
    :param image: np.ndarray (H,W,C)
    :param threshold: percentile
    :param *args: argomenti extra ignorati
    """
    H, W, C = image.shape
    reshaped = image.reshape(-1, C).astype(float)

    mean_vec = np.mean(reshaped, axis=0)
    cov_mat = np.cov(reshaped, rowvar=False)
    try:
        inv_cov = la.inv(cov_mat)
    except la.LinAlgError:
        inv_cov = la.pinv(cov_mat)

    diff = reshaped - mean_vec
    md_squared = np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
    md = np.sqrt(md_squared)


    cutoff = np.percentile(md, threshold)
    anomalies = md > cutoff
    return anomalies.reshape(H, W)


def find_anomalies_lisa(image, threshold, *args, selection_mask=None):
    """
    Calcola gli indici LISA reali: "moran", "getis", "indic".
    - Riduce l'immagine se troppo grande.
    - Applica una maschera di selezione se presente.
    - Adatta automaticamente la soglia se non trova anomalie.
    - Riporta le maschere alla dimensione originale.
    """

    import numpy as np
    import cv2
    from libpysal.weights import lat2W
    from esda.moran import Moran_Local
    from esda.getisord import G_Local

    MAX_SIZE = 600
    MIN_ANOMALIES = 10
    mode = args[0] if len(args) > 0 else "Single"

    H_orig, W_orig, C = image.shape
    resize_applied = False

    # Applica maschera di selezione (se presente)
    if selection_mask is not None:
        print("[DEBUG] Applicazione della maschera di selezione a LISA")
        image = image.copy()
        image[~selection_mask] = 0

    if H_orig > MAX_SIZE or W_orig > MAX_SIZE:
        scale = min(MAX_SIZE / H_orig, MAX_SIZE / W_orig)
        new_H, new_W = int(H_orig * scale), int(W_orig * scale)
        print(f"[DEBUG] Downscaling LISA image from ({H_orig},{W_orig}) to ({new_H},{new_W})")
        image = cv2.resize(image, (new_W, new_H), interpolation=cv2.INTER_AREA)
        resize_applied = True
    else:
        new_H, new_W = H_orig, W_orig

    if mode == "RGB":
        gray = image.mean(axis=2)
    else:
        gray = image[..., 0] if C == 1 else image.mean(axis=2)

    flat = gray.flatten()
    w = lat2W(new_H, new_W, rook=True)
    w.transform = 'r'

    moran = Moran_Local(flat, w)
    getis = G_Local(flat, w)

    moran_z = moran.z_sim
    getis_z = getis.Zs

    def adaptive_mask(z_scores, initial_threshold):
        for t in range(initial_threshold, 49, -5):
            cutoff = np.percentile(z_scores, t)
            mask = z_scores >= cutoff
            count = np.sum(mask)
            if count >= MIN_ANOMALIES:
                print(f"[DEBUG] Threshold adattato: {t}° percentile → {count} anomalie")
                return mask
        print("[DEBUG] Nessuna soglia ha raggiunto il minimo. Nessuna anomalia trovata.")
        return np.zeros_like(z_scores, dtype=bool)

    moran_mask = adaptive_mask(moran_z, threshold)
    getis_mask = adaptive_mask(getis_z, threshold)
    indic_mask = moran_mask & getis_mask

    print("[DEBUG] moran_z range:", np.min(moran_z), np.max(moran_z))
    print("[DEBUG] getis_z range:", np.min(getis_z), np.max(getis_z))
    print(f"[DEBUG] Indic (intersezione) → {np.sum(indic_mask)} anomalie")
    print("[DEBUG] Num anomalies:", np.sum(moran_mask), np.sum(getis_mask), np.sum(indic_mask))

    def restore_mask(mask):
        reshaped = mask.reshape(new_H, new_W).astype(np.uint8) * 255
        restored = cv2.resize(reshaped, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST) > 0
        return restored if mask is not None else restored

    return {
        "moran": restore_mask(moran_mask),
        "getis": restore_mask(getis_mask),
        "indic": restore_mask(indic_mask)
    }

def highlight_anomalies(image, anomalies, color):
    """
    Evidenzia le anomalie nell'immagine con il colore selezionato.
    image: np.ndarray (H,W,C)
    anomalies: boolean mask (H,W)
    """
    # Aggiunta: se immagine ha 1 canale, la espandiamo a 3
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    color_map = {
        "Red": [255, 0, 0],
        "White": [255, 255, 255],
        "Blue": [0, 0, 255],
        "Green": [0, 255, 0],
        "Black": [0, 0, 0]
    }
    rgb_color = np.array(color_map[color], dtype=np.uint8)

    highlighted_image = image.copy()
    anomalies = anomalies.astype(bool)
    highlighted_image[anomalies] = rgb_color
    return highlighted_image


def create_anomaly_layer(anomalies, color, shape, opacity=200):
    """
    Crea un layer RGBA delle stesse dimensioni dell'immagine originale in cui:
      - I pixel per cui la maschera (anomalies) è True vengono colorati con il colore scelto e l'opacità indicata.
      - I pixel non anomali sono completamente trasparenti (alpha=0).

    :param anomalies: array booleano (H, W) con True per pixel anomali.
    :param color: stringa (es. "Red", "Blue", "Green", ecc.).
    :param shape: tuple (H, W) della dimensione dell'immagine.
    :param opacity: intero (0-255) per l'opacità del layer (default=200).
    :return: array NumPy (H, W, 4) del layer RGBA.
    """
    import numpy as np
    layer = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
    color_map = {
        "Red": [255, 0, 0],
        "White": [255, 255, 255],
        "Blue": [0, 0, 255],
        "Green": [0, 255, 0],
        "Black": [0, 0, 0]
    }
    rgb = np.array(color_map[color], dtype=np.uint8)
    # Imposta nei pixel anomali il colore e l'opacità; gli altri rimangono trasparenti
    layer[anomalies] = np.concatenate((rgb, [opacity]))
    return layer


def composite_anomaly_overlay(original_image, anomaly_mask, color, opacity=200):
    from PIL import Image
    import numpy as np

    # Se l'immagine originale ha 5 o più canali e non è in formato uint8,
    # applichiamo la stessa logica di pseudoRGB usata in display_image_on_canvas
    if original_image.ndim == 3 and original_image.shape[2] >= 5:
        print("[DEBUG] composite_anomaly_overlay: Detected multi-channel image (>=5 channels). Applying pseudoRGB stretch.")
        # Seleziona bande: ad esempio banda 3, banda 2, banda 1 (modifica se necessario)
        pseudo_rgb = original_image[..., [2, 1, 0]].copy()
        # Sostituisci eventuali valori di nodata (es. -10000) con NaN
        if not np.issubdtype(pseudo_rgb.dtype, np.floating):
            pseudo_rgb = pseudo_rgb.astype(np.float32)
        pseudo_rgb[pseudo_rgb < -9999] = np.nan
        # Calcola low/high ignorando i NaN
        low = np.nanmin(pseudo_rgb)
        high = np.nanmax(pseudo_rgb)
        print(f"[DEBUG] composite_anomaly_overlay: stretch range: {low} .. {high}")
        # Evita divisioni per zero
        if high - low < 1e-6:
            arr_f = np.zeros_like(pseudo_rgb)
        else:
            arr_f = (pseudo_rgb - low) / (high - low) * 255.0
        # Converte eventuali NaN in 0 e assicura il range [0,255]
        arr_f = np.nan_to_num(arr_f, nan=0)
        arr_f = np.clip(arr_f, 0, 255).astype('uint8')
        # Sostituisci l'immagine originale con quella convertita in pseudoRGB
        original_image = arr_f
    else:
        # Se l'immagine non è uint8, assicurati di convertirla
        if original_image.dtype != np.uint8:
            original_image = np.clip(original_image, 0, 255).astype('uint8')

    # Ora converte l'immagine (ora in formato RGB a 3 canali) in un'immagine PIL in modalità RGBA
    original_img = Image.fromarray(original_image).convert("RGBA")

    # Crea il layer delle anomalie usando la funzione create_anomaly_layer (già definita)
    layer_array = create_anomaly_layer(anomaly_mask, color, original_image.shape[:2], opacity)
    anomaly_layer = Image.fromarray(layer_array)

    # Componi il layer sopra l'immagine originale
    composite_img = Image.alpha_composite(original_img, anomaly_layer)

    # Ritorna l'immagine composita convertita in RGB (come array NumPy)
    return np.array(composite_img.convert("RGB"))


