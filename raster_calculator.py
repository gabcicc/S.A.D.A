import tkinter as tk
from tkinter import messagebox
import numpy as np
from index_viewer import IndexViewer
from tooltip import get_tooltip, Tooltip

def open_raster_calculator_via_assignment(parent, update_callback):
    """
    Prima apre la finestra di Band Assignment.
    Se parent.band_map esiste e corrisponde al numero di canali, salta l'assegnazione e
    apre direttamente il Raster Calculator. Altrimenti, apre l'assegnazione e poi,
    quando l'utente clicca OK, apre il Raster Calculator con la mappa di bande assegnate.
    """
    # Sorgente dati per il Raster Calculator: preferisci lo stack multispettrale completo se disponibile
    arr = getattr(parent, 'multispectral_stack', None)
    if not isinstance(arr, np.ndarray) or arr.ndim < 3:
        # fallback alla preview / immagine corrente
        arr = parent.original_image_np

    # Conta canali in modo robusto
    n_channels = 1 if (hasattr(arr, 'ndim') and arr.ndim == 2) else (arr.shape[2] if hasattr(arr, 'shape') and arr.ndim == 3 else 1)

    # Se esiste già parent.band_map con lo stesso numero di bande,
    # andiamo direttamente al Raster Calculator
    if hasattr(parent, 'band_map') and isinstance(parent.band_map, dict):
        band_map = parent.band_map
        ignored = getattr(parent, 'ignored_bands', set()) or set()
        # Completa se ogni banda è assegnata o ignorata
        complete = all((f"B{i + 1}" in band_map) or (i in ignored) for i in range(n_channels))
        if complete:
            open_raster_calculator(parent, arr, band_map, update_callback)
            return

    # Altrimenti, apriamo la finestra di assegnamento
    def on_band_map_ready(bm):
        # Salviamo bm in parent
        parent.band_map = bm
        open_raster_calculator(parent, arr, bm, update_callback)

    open_band_assignment(parent, arr, on_band_map_ready)


def open_band_assignment(parent, original_image_np, band_assignment_callback):
    """
    Apre una finestra Toplevel per assegnare ad ogni banda reale un nome logico
    (R, G, B, NIR, ecc.) **oppure** eliminarla se non serve.
    Tutte le bande devono risultare *assegnate* oppure *eliminate* affinché
    il pulsante OK si attivi.
    """
    assign_win = tk.Toplevel(parent)
    assign_win.title("Band Assignment")

    # === Dati iniziali ===
    if original_image_np.ndim == 2:
        n_channels = 1
    else:
        n_channels = original_image_np.shape[2]
    real_band_list = [f"B{i+1}" for i in range(n_channels)]

    logic_choices = ["R", "G", "B", "NIR", "RED-EDGE", "SWIR", "X", "+"]

    # Mappa bande già salvata (se esiste) e set di bande rimosse
    band_map: dict[str, str] = getattr(parent, "band_map", {}) or {}
    removed_bands: set[str] = set()

    # Precarica bande ignorate da sessioni precedenti (indici 0-based -> etichette 'B{i+1}')
    prev_ignored = getattr(parent, 'ignored_bands', set()) or set()
    ignored_labels = {f"B{i + 1}" for i in prev_ignored}

    # === GUI layout principale ===
    frame_main = tk.Frame(assign_win)
    frame_main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

    # -- Listbox sinistra: bande reali --
    tk.Label(frame_main, text="Real Bands:").grid(row=0, column=0)
    left_listbox = tk.Listbox(frame_main, height=6, exportselection=False)
    left_listbox.grid(row=1, column=0, padx=5, pady=5)
    for b in real_band_list:
        if b not in band_map and b not in ignored_labels:
            left_listbox.insert(tk.END, b)
        elif b in ignored_labels:
            removed_bands.add(b)  # ripristina lo stato "ignored"

    # -- Listbox destra: nomi logici --
    tk.Label(frame_main, text="Logical Names:").grid(row=0, column=2)
    right_listbox = tk.Listbox(frame_main, height=6, exportselection=False)
    right_listbox.grid(row=1, column=2, padx=5, pady=5)
    for c in logic_choices:
        right_listbox.insert(tk.END, c)

    # Pulsanti azione nella linea centrale
    assign_btn = tk.Button(frame_main, text="Assign", state=tk.DISABLED)
    assign_btn.grid(row=1, column=1, padx=5)
    delete_btn = tk.Button(frame_main, text="Ignore", state=tk.DISABLED)
    delete_btn.grid(row=2, column=1, padx=5)

    # === Etichetta stato ===
    status_label = tk.Label(
        assign_win,
        text="Select real band first, then select corresponding logical name.",
        anchor="w",
    )
    status_label.pack(side=tk.BOTTOM, fill=tk.X)

    # === CALLBACK E UTILITÀ ===
    def refresh_status():
        """Aggiorna testo di stato e abilita/disabilita OK."""
        parts = [f"{k}->{v}" for k, v in band_map.items()]
        parts.extend(f"{b}->(ignored)" for b in removed_bands)
        status_label.config(
            text=", ".join(parts)
            if parts
            else "Select real band first, then select corresponding logical name."
        )

        # Tutte le bande devono essere in band_map oppure removed_bands
        done = len(band_map) + len(removed_bands) == len(real_band_list)
        ok_btn.config(state=tk.NORMAL if done else tk.DISABLED)

    def update_buttons(*_):
        """Gestisce l'abilitazione dei pulsanti Assign/Delete."""
        # Assign richiede entrambe le selezioni
        if left_listbox.curselection() and right_listbox.curselection():
            assign_btn.config(state=tk.NORMAL)
        else:
            assign_btn.config(state=tk.DISABLED)

        # Delete richiede solo la selezione a sinistra
        if left_listbox.curselection():
            delete_btn.config(state=tk.NORMAL)
        else:
            delete_btn.config(state=tk.DISABLED)

    def on_right_select(_event):
        """Permette di aggiungere nomi logici personalizzati tramite '+'. """
        if right_listbox.curselection():
            sel = right_listbox.get(right_listbox.curselection())
            if sel == "+":
                new_name = tk.simpledialog.askstring(
                    "Custom Logical Name", "Enter new logical name:"
                )
                if new_name:
                    right_listbox.insert(tk.END, new_name)
                    right_listbox.selection_clear(0, tk.END)
                    right_listbox.selection_set(tk.END)
                else:
                    right_listbox.selection_clear(0, tk.END)
        update_buttons()

    def on_assign():
        """Assegna il nome logico selezionato alla banda reale selezionata."""
        if not (left_listbox.curselection() and right_listbox.curselection()):
            return
        li = left_listbox.curselection()[0]
        ri = right_listbox.curselection()[0]
        real_name = left_listbox.get(li)
        logic_name = right_listbox.get(ri)

        band_map[real_name] = logic_name  # salva
        left_listbox.delete(li)
        right_listbox.delete(ri)
        left_listbox.selection_clear(0, tk.END)
        right_listbox.selection_clear(0, tk.END)
        assign_btn.config(state=tk.DISABLED)
        delete_btn.config(state=tk.DISABLED)
        refresh_status()

    def on_delete():
        """Rimuove definitivamente la banda selezionata."""
        if not left_listbox.curselection():
            return
        li = left_listbox.curselection()[0]
        real_name = left_listbox.get(li)

        # Se era già stata assegnata in precedenza la deassegniamo
        band_map.pop(real_name, None)
        removed_bands.add(real_name)

        left_listbox.delete(li)
        left_listbox.selection_clear(0, tk.END)
        assign_btn.config(state=tk.DISABLED)
        delete_btn.config(state=tk.DISABLED)
        refresh_status()

    def on_auto_rgb():
        """Assegna automaticamente R,G,B se disponibili tra le bande residue."""
        template = {"B1": "R", "B2": "G", "B3": "B"}
        for real_name, logic_name in template.items():
            if real_name not in band_map and real_name not in removed_bands:
                try:
                    idx_left = left_listbox.get(0, tk.END).index(real_name)
                except ValueError:
                    continue  # già tolta
                band_map[real_name] = logic_name
                left_listbox.delete(idx_left)

                # rimuove anche nome logico dalla lista destra se presente
                try:
                    idx_right = right_listbox.get(0, tk.END).index(logic_name)
                    right_listbox.delete(idx_right)
                except ValueError:
                    pass
        refresh_status()

    def on_reset_assign():
        """Ripristina la finestra allo stato iniziale."""
        left_listbox.delete(0, tk.END)
        for b in real_band_list:
            left_listbox.insert(tk.END, b)
        right_listbox.delete(0, tk.END)
        for c in logic_choices:
            right_listbox.insert(tk.END, c)
        band_map.clear()
        removed_bands.clear()
        assign_btn.config(state=tk.DISABLED)
        delete_btn.config(state=tk.DISABLED)
        refresh_status()

    # === Binds e pulsanti ===
    left_listbox.bind("<<ListboxSelect>>", update_buttons)
    right_listbox.bind("<<ListboxSelect>>", update_buttons)
    right_listbox.bind("<<ListboxSelect>>", on_right_select)

    assign_btn.config(command=on_assign)
    delete_btn.config(command=on_delete)

    auto_rgb_btn = tk.Button(assign_win, text="Auto RGB assignment", command=on_auto_rgb)
    auto_rgb_btn.pack(side=tk.BOTTOM, pady=5)
    Tooltip(auto_rgb_btn, get_tooltip("auto_rgb"))

    reset_assign_btn = tk.Button(assign_win, text="Reset Assign", command=on_reset_assign)
    reset_assign_btn.pack(side=tk.BOTTOM, pady=5)

    def on_ok():
        # Persisti la mappatura su parent prima di chiudere
        try:
            parent.band_map = dict(band_map)  # copie difensive
        except Exception:
            parent.band_map = band_map
        # Costruisci logic_to_index: {nome_logico: indice_reale}
        logic_to_index = {}
        for real_name, logic_name in band_map.items():
            # real_name atteso come 'B{i}'
            try:
                idx = int(real_name[1:]) - 1
                if idx >= 0:
                    logic_to_index[logic_name] = idx
            except Exception:
                continue
        setattr(parent, 'logic_to_index', logic_to_index)
        # Aggiorna la combobox con i nuovi nomi (senza aggiungere bande)
        try:
            parent.update_band_combobox(None)
        except Exception:
            pass
        # Persiste le bande ignorate come indici 0-based
        try:
            parent.ignored_bands = {int(lbl[1:]) - 1 for lbl in removed_bands}
        except Exception:
            parent.ignored_bands = set()
        assign_win.destroy()
        band_assignment_callback(band_map)

    ok_btn = tk.Button(assign_win, text="OK", state=tk.DISABLED, command=on_ok)
    ok_btn.pack(side=tk.BOTTOM, pady=5)

    # Stato iniziale
    refresh_status()
    update_buttons()

    return assign_win


def open_raster_calculator(parent, original_image_np, band_map, update_callback):
    """
    Apre la finestra Toplevel per il Raster Calculator con i nomi di banda logici (es. R, G, B, NIR).
    """
    calc_win = tk.Toplevel(parent)
    calc_win.title("Raster Calculator")

    status_label = tk.Label(calc_win, text="", fg="white", anchor="w")
    status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
    parent._raster_calc_status_label = status_label

    top_frame = tk.Frame(calc_win)
    top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    assigned_bands_str = ", ".join([f"{k}->{v}" for k,v in band_map.items()])
    band_label = tk.Label(top_frame, text="Assigned Bands:")
    band_label.pack(side=tk.LEFT, padx=5)
    band_map_show = tk.Label(top_frame, text=assigned_bands_str)
    band_map_show.pack(side=tk.LEFT, padx=5)
    # Determina le bande logiche assegnate
    assigned_logic_names = set(band_map.values())

    expr_frame = tk.Frame(calc_win)
    expr_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    expr_label = tk.Label(expr_frame, text="Expression:")
    expr_label.pack(side=tk.LEFT, padx=5)
    expr_entry = tk.Entry(expr_frame, width=30)
    expr_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    quick_frame = tk.Frame(calc_win)
    quick_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

    reset_bands_btn = tk.Button(
        calc_win, text="Reset Band Assignment",
        command=lambda: on_reset_bands(calc_win, parent, original_image_np, update_callback)
    )
    reset_bands_btn.pack(side=tk.BOTTOM, padx=5, pady=5)

    ndvi_btn = tk.Button(
        quick_frame, text="NDVI",
        command=lambda: expr_entry.insert(tk.END, "(NIR - R) / (NIR + R)")
    )
    ndvi_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(ndvi_btn, get_tooltip("raster_ndvi"))

    gndvi_btn = tk.Button(
        quick_frame, text="GNDVI",
        command=lambda: expr_entry.insert(tk.END, "(NIR - G) / (NIR + G)")
    )
    gndvi_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(gndvi_btn, get_tooltip("raster_gndvi"))

    bndvi_btn = tk.Button(
        quick_frame, text="BNDVI",
        command=lambda: expr_entry.insert(tk.END, "(NIR - B) / (NIR + B)")
    )
    bndvi_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(bndvi_btn, get_tooltip("raster_bndvi"))

    sr_btn = tk.Button(
        quick_frame, text="SR",
        command=lambda: expr_entry.insert(tk.END, "NIR / R")
    )
    sr_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(sr_btn, get_tooltip("raster_sr"))

    evi_btn = tk.Button(
        quick_frame, text="EVI",
        command=lambda: expr_entry.insert(tk.END, "2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1)")
    )
    evi_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(evi_btn, get_tooltip("raster_evi"))

    gemi_btn = tk.Button(
        quick_frame, text="GEMI",
        command=lambda: expr_entry.insert(tk.END, "(2*(NIR**2-R**2)+1.5*NIR+0.5*R)/(NIR+R+0.5)")
    )
    gemi_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(gemi_btn, get_tooltip("raster_gemi"))

    osavi_btn = tk.Button(
        quick_frame, text="OSAVI",
        command=lambda: expr_entry.insert(tk.END, "(NIR - R) / (NIR + R + 0.16)")
    )
    osavi_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(osavi_btn, get_tooltip("raster_osavi"))

    vari_btn = tk.Button(
        quick_frame, text="VARI",
        command=lambda: expr_entry.insert(tk.END, "(G - R) / (G + R - B)")
    )
    vari_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(vari_btn, get_tooltip("raster_vari"))

    tgi_btn = tk.Button(
        quick_frame, text="TGI",
        command=lambda: expr_entry.insert(tk.END, "-0.5 * (190 * (R - G) - 120 * (R - B))")
    )
    tgi_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(tgi_btn, get_tooltip("raster_tgi"))

    # --- Compute All Indices button ---
    compute_all_btn = tk.Button(
        quick_frame, text="Compute All Indices",
        command=lambda: compute_all_indices(parent, original_image_np, band_map)
    )
    compute_all_btn.pack(side=tk.LEFT, padx=5)
    Tooltip(compute_all_btn, "Compute and visualize all predefined indices.")

    # Abilita o disabilita i pulsanti in base alle bande assegnate
    def has(*required):
        return all(b in assigned_logic_names for b in required)

    ndvi_btn.config(state=tk.NORMAL if has("NIR", "R") else tk.DISABLED)
    gndvi_btn.config(state=tk.NORMAL if has("NIR", "G") else tk.DISABLED)
    bndvi_btn.config(state=tk.NORMAL if has("NIR", "B") else tk.DISABLED)
    sr_btn.config(state=tk.NORMAL if has("NIR", "R") else tk.DISABLED)
    evi_btn.config(state=tk.NORMAL if has("NIR", "R", "G", "B") else tk.DISABLED)
    gemi_btn.config(state=tk.NORMAL if has("NIR", "R") else tk.DISABLED)
    osavi_btn.config(state=tk.NORMAL if has("NIR", "R") else tk.DISABLED)
    vari_btn.config(state=tk.NORMAL if has("G", "R", "B") else tk.DISABLED)
    tgi_btn.config(state=tk.NORMAL if has("R", "G", "B") else tk.DISABLED)



    compute_btn = tk.Button(
        calc_win, text="Compute",
        command=lambda: compute_raster_expression_with_map(parent, original_image_np, band_map, expr_entry.get(), update_callback)
    )
    compute_btn.pack(side=tk.BOTTOM, padx=5, pady=5)

    # === Pulsante per aggiungere come nuova banda ===
    add_band_btn = tk.Button(
        calc_win, text="Add as image band", state=tk.DISABLED,
        command=lambda: add_result_as_band(parent, update_callback)
    )
    add_band_btn.pack(side=tk.BOTTOM, padx=5, pady=5)
    Tooltip(add_band_btn, get_tooltip("raster_add_band"))

    def on_reset_calc():
        # 1) Resetta l'immagine allo stato originale
        apply_raster_calc_reset(parent, update_callback)
        # 2) Svuota la casella Expression
        expr_entry.delete(0, tk.END)
        # Disabilita il pulsante "Add as image band" e pulisci lo status
        add_band_btn.config(state=tk.DISABLED)
        if hasattr(parent, "_raster_calc_status_label"):
            parent._raster_calc_status_label.config(text="", fg="white")

    reset_calc_btn = tk.Button(
        calc_win, text="Clear Expression",
        command=on_reset_calc
    )
    reset_calc_btn.pack(side=tk.BOTTOM, padx=5, pady=5)

    # Rimuove l'anteprima se si chiude la finestra
    def _on_close():
        apply_raster_calc_reset(parent, update_callback)
        calc_win.destroy()

    calc_win.protocol("WM_DELETE_WINDOW", _on_close)

    return calc_win

def on_reset_bands(calc_win, parent, original_image_np, update_callback):
    """
    Chiude la finestra del Raster Calculator, azzera parent.band_map
    e riapre la finestra di band assignment da zero.
    """
    # Chiudi la finestra calcolatore
    calc_win.destroy()
    # Svuota le mappe
    parent.band_map = {}
    try:
        parent.logic_to_index = {}
    except Exception:
        pass
    try:
        parent.ignored_bands = set()
    except Exception:
        pass
    # Riapri la finestra di band assignment
    def on_band_map_ready(new_map):
        # Salviamo e riapriamo raster calculator
        parent.band_map = new_map
        open_raster_calculator(parent, original_image_np, new_map, update_callback)

    open_band_assignment(parent, original_image_np, on_band_map_ready)


def compute_raster_expression_with_map(parent, original_image_np, band_map, expression, update_callback):
    """
    Esegue il calcolo dell'espressione raster su original_image_np,
    sostituendo i nomi logici (R,G,B,NIR,...) in base a band_map (es. {'B4':'NIR','B3':'R'}).
    """
    if original_image_np.ndim < 3:
        messagebox.showerror("Raster Calculator", "Image is not multi-band.")
        return

    arr = original_image_np
    n_channels = arr.shape[2]
    real_band_list = [f"B{i+1}" for i in range(n_channels)]
    real_index_map = {real_band_list[i]: i for i in range(n_channels)}

    # Se band_map è vuoto, errore
    if not band_map:
        messagebox.showerror("Raster Calculator", "No band map assigned. Please assign bands first.")
        return

    # Costruiamo env: key=logico (R, G, B, NIR), value=arr[..., idx]
    env = {}
    for real_band, logic_name in band_map.items():
        if real_band in real_index_map:
            idx = real_index_map[real_band]
            env[logic_name] = arr[..., idx].astype(float)

    # Protezione contro div/0 e valori estremi nei canali chiave
    for key in ["R", "G", "B", "NIR"]:
        if key in env:
            band = env[key]
            band[band == 0] = 1e-6
            env[key] = band

    # Check se l'espressione contiene nomi logici non presenti in env
    import re
    tokens = re.findall(r"[A-Za-z_]\w*", expression)
    for t in tokens:
        if t not in env and t.upper() not in ["SIN","COS","TAN","LOG","EXP"]: # or fun speciali
            messagebox.showerror("Raster Calculator", f"Missing band logic name '{t}' in band_map.")
            return

    import numpy as np

    parent.last_expression = expression  # salva l'espressione usata
    # Esegui eval
    try:
        result = eval(expression, {"__builtins__": None, "np": np}, env)
    except Exception as e:
        messagebox.showerror("Raster Calculator", f"Invalid expression:\n{e}")
        return

    # Normalizziamo in [0..255] per visualizzare
    import numpy as np
    result = np.array(result)
    if np.all(np.isnan(result)):
        messagebox.showwarning("Raster Calculator", "Result is all NaN. Check your expression.")
    # Applica uno stretch percentile per evitare outlier
    finite_vals = result[np.isfinite(result)]
    if finite_vals.size == 0:
        messagebox.showwarning("Raster Calculator", "All result values are NaN or Inf.")
        return

    low = np.percentile(finite_vals, 2)
    high = np.percentile(finite_vals, 98)

    if high - low > 1e-6:
        scaled = (result - low) / (high - low) * 255.0
    else:
        scaled = np.zeros_like(result)

    # Rimuove NaN e inf prima del cast
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)

    if scaled.ndim == 2:
        scaled = scaled[..., np.newaxis]

    # Memorizza il risultato per uso futuro ma NON lo aggiunge
    parent.last_raster_result = scaled

    # Mostra anteprima nel canvas ma non altera original_image_np
    if update_callback:
        update_callback(scaled, "__preview__")  # "__preview__" o None per evitare aggiunte

    # Abilita il pulsante "Add as image band"
    if hasattr(parent, "_raster_calc_status_label"):
        parent._raster_calc_status_label.config(text="Ready to add result as band", fg="white")

    for child in parent.winfo_children():
        if isinstance(child, tk.Toplevel) and child.title() == "Raster Calculator":
            for widget in child.winfo_children():
                if isinstance(widget, tk.Button) and widget.cget("text") == "Add as image band":
                    widget.config(state=tk.NORMAL)


def apply_raster_calc_reset(parent, update_callback):
    """
    Elimina la banda di anteprima (__preview__) e ripristina lo stato originale SENZA
    aggiungere nuovamente le bande di origine (evita duplicati nel band selector).
    """
    # Pulisce eventuali variabili di stato legate all'anteprima
    for attr in ("last_raster_result", "last_expression"):
        if hasattr(parent, attr):
            delattr(parent, attr)

    # Chiede al callback di rimuovere la preview.
    # Convenzione: img=None + band_name -> rimozione di quella banda.
    if update_callback:
        try:
            update_callback(None, "__preview__")
        except TypeError:
            # Se il callback richiede un terzo argomento opzionale 'action'
            update_callback(None, "__preview__", "remove")

def add_result_as_band(parent, update_callback):
    if not hasattr(parent, 'last_raster_result'):
        messagebox.showerror("Raster Calculator", "No computed result found.")
        return

    new_band = parent.last_raster_result
    if new_band.ndim == 2:
        new_band = new_band[..., np.newaxis]

    expr = parent.last_expression.lower() if hasattr(parent, "last_expression") else ""
    print(f"[DEBUG] Expression for logical name: '{expr}'")

    known_formulas = {
        "(nir-r)/(nir+r)": "NDVI",
        "(nir-g)/(nir+g)": "GNDVI",
        "(nir-b)/(nir+b)": "BNDVI",
        "nir/r": "SR",
        "2.5*(nir-r)/(nir+6*r-7.5*b+1)": "EVI",
        "(2*(nir**2-r**2)+1.5*nir+0.5*r)/(nir+r+0.5)": "GEMI",
        "(nir-r)/(nir+r+0.16)": "OSAVI",
        "(g-r)/(g+r-b)": "VARI",
        "-0.5*(190*(r-g)-120*(r-b))": "TGI"
    }

    def _normalize(s: str) -> str:
        """Remove spaces, make lowercase and turn '**' into '^' for easy matching."""
        return s.replace(" ", "").lower().replace("**", "^")

    expr_normalized = _normalize(expr)
    known_formulas_norm = {_normalize(k): v for k, v in known_formulas.items()}
    chosen_name = known_formulas_norm.get(expr_normalized)

    if not chosen_name:
        existing_custom = [v for v in parent.band_map.values() if v.startswith("CustomIndex")]
        next_id = len(existing_custom) + 1
        chosen_name = f"CustomIndex{next_id}"

    print(f"[DEBUG] Assigned band name: {chosen_name}")

    # === Niente aggiunta diretta! ===
    # L'aggiunta sarà gestita da update_callback
    if update_callback:
        update_callback(new_band, chosen_name)

    if hasattr(parent, "_raster_calc_status_label"):
        parent._raster_calc_status_label.config(text=f"Index '{chosen_name}' added as image band.", fg="white")


# --- Compute all indices function ---
def compute_all_indices(parent, original_image_np, band_map):
    from tkinter import messagebox
    import numpy as np
    # rimosso PCAEditor

    if original_image_np.ndim < 3:
        messagebox.showerror("Raster Calculator", "Image is not multi-band.")
        return

    arr = original_image_np
    n_channels = arr.shape[2]
    real_band_list = [f"B{i+1}" for i in range(n_channels)]
    real_index_map = {real_band_list[i]: i for i in range(n_channels)}

    if not band_map:
        messagebox.showerror("Raster Calculator", "No band map assigned. Please assign bands first.")
        return

    env = {}
    for real_band, logic_name in band_map.items():
        if real_band in real_index_map:
            idx = real_index_map[real_band]
            env[logic_name] = arr[..., idx].astype(float)

    for key in ["R", "G", "B", "NIR"]:
        if key in env:
            band = env[key]
            band[band == 0] = 1e-6
            env[key] = band

    index_expressions = {
        "NDVI": "(NIR - R) / (NIR + R)",
        "GNDVI": "(NIR - G) / (NIR + G)",
        "BNDVI": "(NIR - B) / (NIR + B)",
        "SR": "NIR / R",
        "EVI": "2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1)",
        "GEMI": "(2*(NIR**2 - R**2) + 1.5*NIR + 0.5*R)/(NIR + R + 0.5)",
        "OSAVI": "(NIR - R) / (NIR + R + 0.16)",
        "VARI": "(G - R) / (G + R - B)",
        "TGI": "-0.5 * (190 * (R - G) - 120 * (R - B))"
    }

    available = []
    for name, expr in index_expressions.items():
        try:
            result = eval(expr, {"__builtins__": None, "np": np}, env)
            result = np.array(result)
            finite_vals = result[np.isfinite(result)]
            if finite_vals.size == 0:
                continue
            low, high = np.percentile(finite_vals, 2), np.percentile(finite_vals, 98)
            if high - low > 1e-6:
                scaled = (result - low) / (high - low) * 255.0
            else:
                scaled = np.zeros_like(result)
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
            scaled = np.clip(scaled, 0, 255).astype(np.uint8)
            available.append((name, scaled[..., np.newaxis]))
        except Exception as e:
            print(f"[DEBUG] Skipped {name} due to error: {e}")

    if not available:
        messagebox.showinfo("Raster Calculator", "No index could be computed.")
        return

    h, w = arr.shape[:2]
    stack = np.concatenate([b for _, b in available], axis=2)
    result = stack.reshape((-1, stack.shape[2]))

    IndexViewer(
        root=parent,
        main_gui=parent,
        index_result=result,
        shape=(h, w),
        colormap="Grayscale",
        index_names=[name for name, _ in available],
        selection_mask=None,
        selection_bbox=None
    )