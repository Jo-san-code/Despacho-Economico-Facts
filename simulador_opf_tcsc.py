# -*- coding: utf-8 -*-
"""
Simulador Universal de Flujo de Potencia Optimo (OPF)
con evaluacion del dispositivo FACTS TCSC
(Thyristor Controlled Series Capacitor) linea por linea.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Intentamos importar FPDF para el reporte PDF
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

# =============================================================================
# 1. MOTOR MATEMATICO (Backend OPF + TCSC)
# =============================================================================
S_base = 100.0

V_MIN_OP = 0.95
V_MAX_OP = 1.05

PCT_COMPENSACION = 0.40      
K_TCSC_MAX = 0.70            


def construir_ybus(n_nodos, lineas):
    Ybus = np.zeros((n_nodos, n_nodos), dtype=complex)
    for envio, recepcion, z, y_medio in lineas:
        y = 1.0 / z
        i, j = envio - 1, recepcion - 1

        Ybus[i, j] -= y
        Ybus[j, i] -= y

        Ybus[i, i] += y + y_medio
        Ybus[j, j] += y + y_medio

    return Ybus


def gauss_seidel(nodos, Ybus, max_iter=700, tol=1e-8, alpha=1.4):
    n = len(nodos)
    V = np.array(
        [nodos[i + 1][2] * np.exp(1j * 0.0) for i in range(n)],
        dtype=complex
    )

    for it in range(max_iter):
        V_prev = V.copy()

        for i in range(n):
            id_nodo, tipo, V_init, Pgen, Qgen, Pcarga, Qcarga = nodos[i + 1]

            if tipo == 0:
                continue

            Pneto = Pgen - Pcarga
            suma = np.sum([Ybus[i, j] * V[j] for j in range(n) if j != i])

            if tipo == 1:
                Q_calc = -np.imag(np.conj(V[i]) * (Ybus[i, i] * V[i] + suma))
                S = Pneto + 1j * (Q_calc - Qcarga)
                V_calc = (1.0 / Ybus[i, i]) * ((S.conjugate() / V[i].conjugate()) - suma)
                V[i] = V_init * np.exp(1j * np.angle(V_calc))

            elif tipo == 2:
                S = Pneto + 1j * (Qgen - Qcarga)
                V_calc = (1.0 / Ybus[i, i]) * ((S.conjugate() / V[i].conjugate()) - suma)
                V[i] = V_prev[i] + alpha * (V_calc - V_prev[i])

        if np.max(np.abs(V - V_prev)) < tol:
            return V, True, it + 1

    return V, False, max_iter


def calcular_despacho_economico(demanda_mw, datos_gen):
    G = len(datos_gen)
    P_despachado = np.zeros(G)

    lam_min = 0.0
    lam_max = 8000.0
    tol = 0.0001

    pmax_total = sum(datos_gen[i][5] for i in range(G))
    pmin_total = sum(datos_gen[i][4] for i in range(G))

    if demanda_mw > pmax_total:
        P_despachado = [datos_gen[i][5] for i in range(G)]
        lam_mid = lam_max

    elif demanda_mw < pmin_total:
        P_despachado = [datos_gen[i][4] for i in range(G)]
        lam_mid = lam_min

    else:
        for _ in range(200):
            lam_mid = (lam_min + lam_max) / 2.0
            sum_p = 0.0

            for i in range(G):
                b = datos_gen[i][2]
                c = datos_gen[i][3]
                pmin = datos_gen[i][4]
                pmax = datos_gen[i][5]

                pi = (lam_mid - b) / (2 * c)

                if pi < pmin:
                    pi = pmin
                elif pi > pmax:
                    pi = pmax

                P_despachado[i] = pi
                sum_p += pi

            if abs(sum_p - demanda_mw) < tol:
                break
            elif sum_p < demanda_mw:
                lam_min = lam_mid
            else:
                lam_max = lam_mid

    costo_despacho = 0

    for i in range(G):
        a = datos_gen[i][1]
        b = datos_gen[i][2]
        c = datos_gen[i][3]
        pi = P_despachado[i]

        costo_despacho += a + (b * pi) + (c * (pi ** 2))

    return list(P_despachado), lam_mid, costo_despacho


def preparar_datos(df_nodos, df_lineas, df_gens):
    nodos_base = {}
    for _, row in df_nodos.iterrows():
        nodos_base[int(row['ID'])] = [
            int(row['ID']),
            int(row['Tipo']),
            row['V_init'],
            row['P_gen_MW'] / S_base,
            row['Q_gen_MVAR'] / S_base,
            row['P_carga_MW'] / S_base,
            row['Q_carga_MVAR'] / S_base
        ]

    lineas_base = []
    limites_termicos = []
    for _, row in df_lineas.iterrows():
        r_val = float(row['R'])
        lineas_base.append(
            (int(row['De']), int(row['A']), r_val,
             float(row['X']), float(row['B']))
        )
        # Si R es practicamente 0, es un transformador. Se descarta limite termico.
        if abs(r_val) < 1e-9:
            limites_termicos.append(float('inf'))
        else:
            limites_termicos.append(row['Limite_MVA'] / S_base)

    datos_gen = {}
    for i, row in df_gens.iterrows():
        precio = row['Precio_MWh']
        if precio <= 0:
            precio = 0.01

        a_calc = (1500.0 / precio) + 20.0
        b_calc = precio
        c_calc = 0.0004 * precio

        datos_gen[i] = [
            int(row['Nodo']),
            a_calc,
            b_calc,
            c_calc,
            row['Pmin_MW'],
            row['Pmax_MW']
        ]

    return nodos_base, lineas_base, limites_termicos, datos_gen


def construir_lineas_con_tcsc(lineas_base, k_vector):
    lineas = []
    for idx, (e, r, R, X, B) in enumerate(lineas_base):
        k = float(k_vector[idx]) if k_vector is not None else 0.0
        k = max(0.0, min(k, 0.95))

        Xc = k * X
        X_eff = X - Xc

        if abs(X_eff) < 1e-6:
            X_eff = 1e-6

        z = complex(R, X_eff)
        y_medio = complex(0.0, B / 2.0)
        lineas.append((e, r, z, y_medio))
    return lineas


def _calcular_costo_real(P_gen_mw, datos_gen):
    costo = 0.0
    for i in range(len(datos_gen)):
        a = datos_gen[i][1]
        b = datos_gen[i][2]
        c = datos_gen[i][3]
        P = P_gen_mw[i]
        costo += a + b * P + c * P ** 2
    return costo


def _calcular_lambda_real(P_gen_mw, datos_gen, nodos=None):
    # Costo marginal: lambda = b + c * P_real  (P_real en MW)
    slack_idx = None
    if nodos is not None:
        for i in range(len(datos_gen)):
            nodo_gen = datos_gen[i][0]
            if nodos.get(nodo_gen, [None])[1] == 0:
                slack_idx = i
                break
    if slack_idx is not None:
        b = datos_gen[slack_idx][2]
        c = datos_gen[slack_idx][3]
        P = P_gen_mw[slack_idx]
        return b + c * P
    marginales = []
    for i in range(len(datos_gen)):
        b = datos_gen[i][2]
        c = datos_gen[i][3]
        P = P_gen_mw[i]
        marginales.append(b + c * P)
    return max(marginales)


def resolver_sistema(nodos_base, lineas_base, limites_termicos,
                     datos_gen, k_vector=None):
    import copy
    nodos = copy.deepcopy(nodos_base)
    lineas = construir_lineas_con_tcsc(lineas_base, k_vector)

    Ybus = construir_ybus(len(nodos), lineas)

    # Calculamos la demanda neta restando la generacion fija en nodos PQ (Tipo 2)
    carga_total_mw = sum(n[5] for k, n in nodos.items()) * S_base
    gen_fija_mw = sum(n[3] for k, n in nodos.items() if n[1] == 2) * S_base
    demanda_base_mw = carga_total_mw - gen_fija_mw

    perdidas_estimadas = 0.0
    error = 1.0
    iteracion = 1
    P_gen_mw, lam_despacho, costo_despacho = [], 0.0, 0.0
    conv_flujo = True

    while error > 1e-5 and iteracion <= 50:
        demanda_total = demanda_base_mw + perdidas_estimadas

        P_gen_mw, lam_despacho, costo_despacho = calcular_despacho_economico(
            demanda_total, datos_gen
        )

        for i in range(len(datos_gen)):
            nodo_con = datos_gen[i][0]
            if nodos[nodo_con][1] == 1:
                nodos[nodo_con][3] = P_gen_mw[i] / S_base

        voltajes, conv, _ = gauss_seidel(nodos, Ybus)
        conv_flujo = conv

        I_inyectada = Ybus.dot(voltajes)
        S_nodo_neta = voltajes * np.conj(I_inyectada)

        slack_id = next(k for k, v in nodos.items() if v[1] == 0)
        slack_idx = list(nodos.keys()).index(slack_id)

        p_slack_gen_pu = S_nodo_neta[slack_idx].real + nodos[slack_id][5]
        q_slack_gen_pu = S_nodo_neta[slack_idx].imag + nodos[slack_id][6]

        p_carga_total = sum(n[5] for k, n in nodos.items())
        q_carga_total = sum(n[6] for k, n in nodos.items())

        p_gen_total = sum(n[3] for k, n in nodos.items() if n[1] in [1, 2]) + p_slack_gen_pu
        q_gen_total = sum(n[4] for k, n in nodos.items() if n[1] in [1, 2]) + q_slack_gen_pu

        perdidas_P_pu = p_gen_total - p_carga_total
        perdidas_Q_pu = q_gen_total - q_carga_total

        perdidas_reales_mw = perdidas_P_pu * S_base

        error = abs(perdidas_reales_mw - perdidas_estimadas)
        perdidas_estimadas = perdidas_reales_mw
        iteracion += 1

    for i in range(len(datos_gen)):
        if nodos[datos_gen[i][0]][1] == 0:
            P_gen_mw[i] = p_slack_gen_pu * S_base

    costo_real = _calcular_costo_real(P_gen_mw, datos_gen)
    lambda_real = _calcular_lambda_real(P_gen_mw, datos_gen, nodos=nodos)

    lam_despacho_orig = lam_despacho
    costo_despacho_orig = costo_despacho

    flujos_raw = []
    flujos_detalle = []

    for idx, (e, r, z, ym) in enumerate(lineas):
        i, j = e - 1, r - 1
        y = 1.0 / z

        Iij = (voltajes[i] - voltajes[j]) * y + voltajes[i] * ym
        Sij = voltajes[i] * np.conj(Iij)

        Iji = (voltajes[j] - voltajes[i]) * y + voltajes[j] * ym
        Sji = voltajes[j] * np.conj(Iji)

        S_perdida = Sij + Sji
        es_trafo = abs(z.real) < 1e-9

        if limites_termicos[idx] == float('inf'):
            carga_pct = 0.0
        else:
            carga_pct = (abs(Sij) / limites_termicos[idx]) * 100.0

        flujos_raw.append((e, r, Sij))
        flujos_detalle.append({
            "linea": f"{e} -> {r}",
            "Sij": Sij,
            "Sji": Sji,
            "S_perdida": S_perdida,
            "carga_pct": carga_pct,
            "limite_mva": limites_termicos[idx] * S_base if not es_trafo else float('inf'),
            "I_mag_pu": abs(Iij),
            "es_trafo": es_trafo
        })

    V_mag = np.abs(voltajes)

    return {
        "voltajes": voltajes,
        "V_min": float(V_mag.min()),
        "V_max": float(V_mag.max()),
        "desviacion_V": float(np.sum((V_mag - 1.0) ** 2)),
        "P_gen_mw": P_gen_mw,
        "lambda": lambda_real,
        "lambda_despacho": lam_despacho_orig,
        "costo": costo_real,
        "costo_despacho": costo_despacho_orig,
        "perdidas_P_pu": perdidas_P_pu,
        "perdidas_Q_pu": perdidas_Q_pu,
        "perdidas_P_mw": perdidas_P_pu * S_base,
        "perdidas_Q_mvar": perdidas_Q_pu * S_base,
        "demanda_base_mw": demanda_base_mw,
        "iteraciones": iteracion - 1,
        "convergio": conv_flujo,
        "flujos_raw": flujos_raw,
        "flujos": flujos_detalle,
        "lineas": lineas,
        "limites": limites_termicos,
        "datos_gen": datos_gen
    }


def evaluar_tcsc(nodos_base, lineas_base, limites_termicos, datos_gen,
                 caso_base, k_lista=None, k_nominal=None):
    if k_lista is None:
        k_lista = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    if k_nominal is None:
        k_nominal = PCT_COMPENSACION

    n_lineas = len(lineas_base)

    costo_base = caso_base["costo"]
    perd_base = caso_base["perdidas_P_mw"]

    sobrecarga_base = max(f["carga_pct"] for f in caso_base["flujos"])
    limite_carga = max(100.0, sobrecarga_base)
    v_min_ref = min(V_MIN_OP, caso_base["V_min"])
    v_max_ref = max(V_MAX_OP, caso_base["V_max"])

    filas_resumen = []
    filas_barrido = []
    mejor = None

    for idx in range(n_lineas):
        e, r, R, X, B = lineas_base[idx]
        nombre = f"{e} -> {r}"

        carga_base_linea = caso_base["flujos"][idx]["carga_pct"]
        es_trafo = abs(R) < 1e-9

        for k in k_lista:
            k_vector = [0.0] * n_lineas
            k_vector[idx] = k

            try:
                res_k = resolver_sistema(
                    nodos_base, lineas_base, limites_termicos,
                    datos_gen, k_vector
                )
            except Exception:
                continue

            if not np.all(np.isfinite(np.abs(res_k["voltajes"]))):
                continue

            Xc_k = k * X
            sob_k = max(f["carga_pct"] for f in res_k["flujos"])
            fact_k = (
                res_k["convergio"]
                and sob_k <= limite_carga + 0.5
                and res_k["V_min"] >= v_min_ref
                and res_k["V_max"] <= v_max_ref
                and k <= K_TCSC_MAX
                and not es_trafo
            )

            filas_barrido.append([
                nombre,
                round(k * 100, 1),
                round(Xc_k, 5),
                round(X - Xc_k, 5),
                round(res_k["flujos"][idx]["Sij"].real * S_base, 2),
                round(res_k["flujos"][idx]["carga_pct"], 2) if not es_trafo else 0.0,
                round(res_k["perdidas_P_mw"], 3),
                round(perd_base - res_k["perdidas_P_mw"], 3),
                round(res_k["costo"], 2),
                round(costo_base - res_k["costo"], 2),
                round(res_k["V_min"], 4),
                round(res_k["V_max"], 4),
                "SI" if fact_k else "NO",
                "<== nominal" if abs(k - k_nominal) < 1e-9 else ""
            ])

        k_vector = [0.0] * n_lineas
        k_vector[idx] = k_nominal
        Xc = k_nominal * X

        try:
            res = resolver_sistema(
                nodos_base, lineas_base, limites_termicos, datos_gen, k_vector
            )
        except Exception:
            res = None

        if res is None or not np.all(np.isfinite(np.abs(res["voltajes"]))):
            filas_resumen.append([
                nombre, round(X, 5), round(k_nominal * 100, 1),
                round(Xc, 5), round(X - Xc, 5),
                round(carga_base_linea, 2) if not es_trafo else "N/A", "-", "-",
                round(perd_base, 3), "-", "-", "-", "-", "-",
                "NO CONVERGE"
            ])
            continue

        carga_k = res["flujos"][idx]["carga_pct"]
        perd_k = res["perdidas_P_mw"]
        costo_k = res["costo"]

        sobrecarga = max(f["carga_pct"] for f in res["flujos"])
        voltaje_ok = (res["V_min"] >= v_min_ref and res["V_max"] <= v_max_ref)
        factible = (
            res["convergio"]
            and sobrecarga <= limite_carga + 0.5
            and voltaje_ok
            and not es_trafo
        )

        d_carga = carga_k - carga_base_linea
        d_perd = perd_base - perd_k
        ahorro = costo_base - costo_k

        if es_trafo:
            estado = "TRAFO (Ignorar TCSC)"
        elif not factible:
            estado = "NO VIABLE"
        elif d_perd > 0.05 and ahorro >= 0:
            estado = "MUY RECOMENDABLE"
        elif d_perd > 0.0 or ahorro > 0:
            estado = "RECOMENDABLE"
        else:
            estado = "NO APORTA"

        filas_resumen.append([
            nombre,
            round(X, 5),
            round(k_nominal * 100, 1),
            round(Xc, 5),
            round(X - Xc, 5),
            round(carga_base_linea, 2) if not es_trafo else "N/A",
            round(carga_k, 2) if not es_trafo else "N/A",
            round(d_carga, 2) if not es_trafo else "N/A",
            round(perd_base, 3),
            round(perd_k, 3),
            round(d_perd, 3),
            round(ahorro, 2),
            round(res["V_min"], 4),
            round(res["V_max"], 4),
            estado
        ])

        merito = ahorro + d_perd * res["lambda"]
        if not factible:
            merito -= 1e6

        cand = {
            "k": k_nominal,
            "Xc": Xc,
            "merito": merito,
            "carga": carga_k,
            "perdidas": perd_k,
            "costo": costo_k,
            "V_min": res["V_min"],
            "V_max": res["V_max"],
            "factible": factible,
            "sobrecarga_max": sobrecarga,
            "linea": nombre,
            "idx": idx
        }

        if mejor is None or cand["merito"] > mejor["merito"]:
            mejor = cand

    pct_txt = f"{k_nominal*100:.0f}"

    df_tcsc = pd.DataFrame(
        filas_resumen,
        columns=[
            "Linea",
            "X linea (pu)",
            "Compensacion (%)",
            f"Xc = {pct_txt}% X (pu)",
            "X - Xc (pu)",
            "Carga sin TCSC (%)",
            "Carga con TCSC (%)",
            "Delta Carga (%)",
            "Perdidas sin TCSC (MW)",
            "Perdidas con TCSC (MW)",
            "Reduccion Perdidas (MW)",
            "Ahorro Costo ($/h)",
            "|V| min (pu)",
            "|V| max (pu)",
            "Evaluacion TCSC"
        ]
    )

    df_barrido = pd.DataFrame(
        filas_barrido,
        columns=[
            "Linea",
            "Compensacion (%)",
            "Xc (pu)",
            "X - Xc (pu)",
            "P linea (MW)",
            "Carga (%)",
            "Perdidas P (MW)",
            "Delta Perdidas (MW)",
            "Costo ($/h)",
            "Ahorro ($/h)",
            "|V| min",
            "|V| max",
            "Factible",
            "Nota"
        ]
    )

    return df_tcsc, df_barrido, mejor


def escalar_demanda(nodos_base, factor):
    import copy
    nodos = copy.deepcopy(nodos_base)
    for k in nodos:
        nodos[k][5] *= factor
        nodos[k][6] *= factor
    return nodos


def identificar_linea_critica(res, umbral=100.0):
    peor = max(res["flujos"], key=lambda f: f["carga_pct"])
    if peor["carga_pct"] < umbral:
        return "-", None
    return peor["linea"].replace(" -> ", "-"), peor["carga_pct"]


def comparar_casos(df_nodos, df_lineas, df_gens,
                   factor_demanda=1.0, k_compensacion=None):
    if k_compensacion is None:
        k_compensacion = PCT_COMPENSACION

    nodos_base, lineas_base, limites, datos_gen = preparar_datos(
        df_nodos, df_lineas, df_gens
    )

    filas = []
    detalle = []
    n_caso = 0

    def agregar(etiqueta_facts, res, k=None, escenario="", xc=None,
                X_orig=None):
        nonlocal n_caso
        n_caso += 1

        linea_crit, carga = identificar_linea_critica(res)
        carga_txt = f"{carga:.2f} %" if carga is not None else "-"

        X_txt = f"{X_orig:.5f}" if X_orig is not None else "-"
        Xc_txt = f"{xc:.5f}" if xc is not None else "-"
        Xc_corr_txt = f"{X_orig - xc:.5f}" if (X_orig is not None and xc is not None) else "-"

        filas.append([
            f"Caso {n_caso}",
            etiqueta_facts,
            f"{res['costo']:,.2f}",
            f"{res['lambda']:.2f}",
            linea_crit,
            carga_txt,
            X_txt,
            Xc_txt,
            Xc_corr_txt,
            f"{res['perdidas_P_mw']:.3f}"
        ])

        carga_max = max(f["carga_pct"] for f in res["flujos"])
        detalle.append([
            f"Caso {n_caso}",
            escenario,
            etiqueta_facts,
            "-" if k is None else f"{k*100:.0f}",
            "-" if xc is None else round(xc, 5),
            round(res["demanda_base_mw"], 2),
            round(res["perdidas_P_mw"], 3),
            round(res["perdidas_Q_mvar"], 3),
            round(res["V_min"], 4),
            round(res["V_max"], 4),
            round(carga_max, 2),
            "SI" if carga_max <= 100.0 else "NO",
            round(res["costo"], 2),
            round(res["lambda"], 4),
            "-" if X_orig is None else round(X_orig, 5),
            "-" if xc is None else round(xc, 5),
            "-" if (X_orig is None or xc is None) else round(X_orig - xc, 5)
        ])

    res1 = resolver_sistema(nodos_base, lineas_base, limites, datos_gen, None)
    agregar("No", res1, None, "Demanda nominal (100 %)")

    if abs(factor_demanda - 1.0) < 1e-9:
        nodos_est = nodos_base
        res2 = res1
    else:
        nodos_est = escalar_demanda(nodos_base, factor_demanda)
        res2 = resolver_sistema(nodos_est, lineas_base, limites, datos_gen, None)

    etiqueta_est = f"Demanda al {factor_demanda*100:.0f} %"
    agregar("No", res2, None, etiqueta_est)

    linea_crit_base, _ = identificar_linea_critica(res2)

    for idx, (e, r, R, X, B) in enumerate(lineas_base):
        k_vector = [0.0] * len(lineas_base)
        k_vector[idx] = k_compensacion
        Xc = k_compensacion * X

        try:
            res = resolver_sistema(
                nodos_est, lineas_base, limites, datos_gen, k_vector
            )
            if not np.all(np.isfinite(np.abs(res["voltajes"]))):
                res = None
        except Exception:
            res = None

        if res is None:
            n_caso += 1
            filas.append([
                f"Caso {n_caso}", f"Si, {e}-{r}", "-", "-",
                "No converge", "-", f"{X:.5f}", f"{Xc:.5f}",
                f"{X - Xc:.5f}", "-"
            ])
            detalle.append([
                f"Caso {n_caso}", etiqueta_est, f"Si, {e}-{r}",
                f"{k_compensacion*100:.0f}", round(Xc, 5),
                "-", "-", "-", "-", "-", "-", "NO", "-",
                "-", round(X, 5), round(Xc, 5), round(X - Xc, 5)
            ])
            continue

        agregar(f"Si, {e}-{r}", res, k_compensacion, etiqueta_est, Xc, X)

    df_casos = pd.DataFrame(
        filas,
        columns=[
            "Caso", "FACTS", "Costo $/h", "lambda $/MWh",
            "Linea critica", "Cargabilidad",
            "X original (pu)", "Xc (pu)", "X corregida (pu)",
            "Perdidas (MW)"
        ]
    )

    df_detalle = pd.DataFrame(
        detalle,
        columns=[
            "Caso", "Escenario de demanda", "FACTS", "Compensacion (%)",
            "Xc (pu)",
            "Demanda (MW)", "Perdidas P (MW)", "Perdidas Q (MVAr)",
            "|V| min (pu)", "|V| max (pu)", "Cargabilidad max (%)",
            "Red sin sobrecarga", "Costo ($/h)", "lambda ($/MWh)",
            "X original (pu)", "Xc (pu)", "X corregida (pu)"
        ]
    )

    return df_casos, df_detalle, linea_crit_base


def ejecutar_analisis_opf(df_nodos, df_lineas, df_gens, evaluar_facts=True):
    nodos_base, lineas_base, limites_termicos, datos_gen = preparar_datos(
        df_nodos, df_lineas, df_gens
    )

    base = resolver_sistema(
        nodos_base, lineas_base, limites_termicos, datos_gen, None
    )

    voltajes = base["voltajes"]
    P_gen_mw = base["P_gen_mw"]

    datos_gens_tabla = []
    for i in range(len(datos_gen)):
        a = datos_gen[i][1]
        b = datos_gen[i][2]
        c = datos_gen[i][3]
        P = P_gen_mw[i]
        costo_gen = a + b * P + c * P ** 2
        # Costo marginal de cada generador: CMg = b + c * P_real
        marg = b + c * P
        datos_gens_tabla.append([
            f"Gen {i+1} (Nodo {datos_gen[i][0]})",
            round(P, 2),
            round(costo_gen, 2),
            round(marg, 4),
            round(a, 2),
            round(b, 2),
            round(c, 4)
        ])

    df_res_gens = pd.DataFrame(
        datos_gens_tabla,
        columns=[
            "Generador", "P Real (MW)", "Costo Gen ($/h)",
            "CMg ($/MWh)",
            "Costo Fijo 'a' ($/h)", "Costo 'b' ($/MWh)",
            "Costo 'c' ($/MW^2h)"
        ]
    )

    df_res_voltajes = pd.DataFrame({
        "Nodo": [i + 1 for i in range(len(voltajes))],
        "Magnitud |V| (pu)": [round(abs(v), 4) for v in voltajes],
        "Angulo (deg)": [round(np.angle(v, deg=True), 4) for v in voltajes]
    })

    datos_flujos = []
    trafos_detectados = []
    for f in base["flujos"]:
        if f.get("es_trafo", False):
            estado = "TRANSFORMADOR"
            trafos_detectados.append(f["linea"])
            lim_val = "N/A"
            carga_val = "N/A"
        else:
            estado = "ALERTA" if f["carga_pct"] > 100 else "OK"
            lim_val = round(f["limite_mva"], 2)
            carga_val = round(f["carga_pct"], 2)
            
        datos_flujos.append([
            f["linea"],
            round(abs(f["Sij"]) * S_base, 2),
            lim_val,
            carga_val,
            estado
        ])

    df_res_flujos = pd.DataFrame(
        datos_flujos,
        columns=["Linea", "|S| (MVA)", "Limite (MVA)", "Carga (%)", "Estado"]
    )

    datos_flujos_perdidas = []
    for f in base["flujos"]:
        Sij, Sji, Sp = f["Sij"], f["Sji"], f["S_perdida"]
        datos_flujos_perdidas.append([
            f["linea"],
            round(Sij.real * S_base, 2),
            round(Sij.imag * S_base, 2),
            round(abs(Sij) * S_base, 2),
            round(Sji.real * S_base, 2),
            round(Sji.imag * S_base, 2),
            round(abs(Sji) * S_base, 2),
            round(Sp.real * S_base, 2),
            round(Sp.imag * S_base, 2)
        ])

    df_res_flujos_perdidas = pd.DataFrame(
        datos_flujos_perdidas,
        columns=[
            "Linea", "P envio (MW)", "Q envio (MVAr)", "|S| envio (MVA)",
            "P recepcion (MW)", "Q recepcion (MVAr)", "|S| recepcion (MVA)",
            "Perdidas P (MW)", "Perdidas Q (MVAr)"
        ]
    )

    if evaluar_facts:
        df_tcsc, df_barrido, mejor = evaluar_tcsc(
            nodos_base, lineas_base, limites_termicos, datos_gen, base
        )
    else:
        df_tcsc = pd.DataFrame()
        df_barrido = pd.DataFrame()
        mejor = None

    resumen_txt = (
        f"Analisis completado en {base['iteraciones']} iteraciones (OPF).\n"
        f"Costo Total Real de Operacion: $ {base['costo']:,.2f} / hr\n"
        f"Costo Marginal (Lambda): $ {base['lambda']:.2f} / MWh\n"
        f"Perdidas Totales: {base['perdidas_P_mw']:.3f} MW\n"
        f"Demanda Total + Perdidas: "
        f"{base['demanda_base_mw'] + base['perdidas_P_mw']:.2f} MW"
    )

    if base["lambda_despacho"] >= 5000.0:
        resumen_txt += (
            f"\n\nNOTA: Demanda ({base['demanda_base_mw']:.0f} MW) > "
            f"Pmax total ({sum(datos_gen[i][5] for i in range(len(datos_gen))):.0f} MW).\n"
            f"Lambda del despacho satura a 5000 (sin significado economico).\n"
            f"Lambda real = CMg del slack = {base['lambda']:.2f} $/MWh\n"
            f"(el slack absorbe perdidas y determina el precio marginal real).\n"
            f"Costo real = suma(a_i + b_i*P_i + c_i*P_i^2) con P_i reales."
        )

    if trafos_detectados:
        resumen_txt += (
            f"\n\nNOTA: Las ramas {', '.join(trafos_detectados)} se identificaron como TRANSFORMADORES (R=0).\n"
            f"Se ha descartado su limite termico para evitar alertas de sobrecarga erroneas."
        )

    if mejor is not None:
        resumen_txt += (
            f"\n\nTCSC recomendado: linea {mejor['linea']} con Xc = "
            f"{mejor['k']*100:.0f} % X = {mejor['Xc']:.5f} pu  |  "
            f"Perdidas: {mejor['perdidas']:.3f} MW  |  "
            f"Ahorro: $ {base['costo'] - mejor['costo']:,.2f} / hr"
        )

    return (
        resumen_txt,
        df_res_gens,
        df_res_voltajes,
        df_res_flujos,
        df_res_flujos_perdidas,
        base["flujos_raw"],
        base["perdidas_P_pu"],
        base["perdidas_Q_pu"],
        df_tcsc,
        df_barrido,
        mejor,
        base
    )


# =============================================================================
# 2. INTERFAZ GRAFICA
# =============================================================================
class SimuladorOPF(tk.Tk):

    def __init__(self):
        super().__init__()

        # Inicializar variable para guardar resultados
        self.ultima_ejecucion = None

        self.title("Simulador Universal de Flujo de Potencia Optimo + FACTS TCSC")
        self.geometry("1300x850")
        self.configure(padx=10, pady=10)

        frame_top = tk.Frame(self)
        frame_top.pack(fill="x", pady=(0, 10))

        tk.Label(
            frame_top,
            text="Configura tu sistema de potencia o utiliza los datos por defecto:",
            font=("Arial", 12, "bold")
        ).pack(side="left")

        tk.Button(
            frame_top,
            text="GENERAR REPORTE PDF",
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=10,
            command=self.generar_reporte_pdf
        ).pack(side="right", padx=(10, 0))

        tk.Button(
            frame_top,
            text="EJECUTAR ANALISIS OPF + TCSC",
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=20,
            command=self.ejecutar
        ).pack(side="right")

        self.var_facts = tk.BooleanVar(value=True)
        tk.Checkbutton(
            frame_top,
            text="Evaluar TCSC en cada linea",
            variable=self.var_facts,
            font=("Arial", 10)
        ).pack(side="right", padx=15)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both")

        cols_nodos = ["ID", "Tipo", "V_init", "P_gen_MW", "Q_gen_MVAR", "P_carga_MW", "Q_carga_MVAR", "P_neta_MW", "Q_neta_MVAR"]
        datos_nodos = [
            [1, 0, 1.0, 0.0 , 0.0 , 0.0, 0.0, 0.0, 0.0],
            [2, 1, 1.00, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [3, 1, 1.00, 85.0 , 0.0 , 0.0, 0.0, 0.0, 0.0],
            [4, 2, 1.00, 0.0 , 0.0, 0.0, 0.0, 0.0, 0.0],
            [5, 2, 1.00, 0.0 , 0.0 , 90.0, 30.0, 0.0, 0.0],
            [6, 2, 1.00, 0.0, 0.0, 0.0,0.0, 0.0,0.0],
            [7, 2, 1.00, 0.0, 0.0, 100.0,35.0, 0.0,0.0],
            [8, 2, 1.00, 0.0, 0.0, 0.0,0.0, 0.0,0.0],
            [9, 2, 1.00, 0.0, 0.0, 125.0,50.0, 0.0,0.0],
        ]
        cols_lineas = ["De", "A", "R", "X", "B", "Limite_MVA"]
        datos_lineas = [
            [1, 4, 0.00, 0.0576, 0.0,260],
            [4, 5, 0.0170, 0.0920, 0.0790, 150],
            [5, 6, 0.0390, 0.1700, 0.1790, 100],
            [3, 6, 0.000, 0.0586, 0.000, 150],
            [6, 7, 0.0119, 0.1008, 0.1045, 100],
            [7, 8, 0.0085, 0.0720, 0.0745,150],
             [8, 2, 0.00, 0.0625, 0.0,250],
            [8, 9, 0.0320, 0.1610, 0.1530,150],
            [9, 4, 0.0100, 0.0850, 0.0880,150],
        ]

        cols_gens = ["Nodo", "Precio_MWh", "Pmin_MW", "Pmax_MW"]
        datos_gens =  [
            [1, 5.0, 10.0, 250.0],
            [2, 6.0, 20.0, 300.0],
            [3, 7.50, 10.0, 150.0]
        ]

        info_nodos = (
            "Leyenda Tipo de Nodo:\n"
            "0 = Nodo Slack\n"
            "1 = Generador (PV)\n"
            "2 = Carga (PQ)\n"
            "Nota: Las Potencias Netas se auto-calculan."
        )

        self.tree_input_nodos = self.crear_pestana_interactiva(
            "Nodos", cols_nodos, datos_nodos, info_nodos
        )

        self.tree_input_lineas = self.crear_pestana_interactiva(
            "Lineas", cols_lineas, datos_lineas,
            "Ingresa las impedancias en p.u.\ny el Limite en MVA."
        )

        self.tree_input_gens = self.crear_pestana_interactiva(
            "Generadores", cols_gens, datos_gens,
            "Ingresa el Precio de Generacion ($/MWh).\n"
            "El sistema calculara automaticamente 'a' y 'c'\n"
            "usando correlaciones fisicas."
        )

        self.frame_resumen = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_resumen, text="Resumen y Despacho")

        self.lbl_resumen = tk.Label(
            self.frame_resumen,
            text="Ejecuta el analisis para ver los resultados.",
            font=("Arial", 12),
            justify="left"
        )
        self.lbl_resumen.pack(pady=10)

        self.tree_res_gens = self.crear_treeview(self.frame_resumen)

        self.frame_voltajes = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_voltajes, text="Voltajes")
        self.tree_res_voltajes = self.crear_treeview(self.frame_voltajes)

        self.frame_flujos = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_flujos, text="Flujos y Cargabilidad")
        self.tree_res_flujos = self.crear_treeview(self.frame_flujos)

        self.frame_flujos_perdidas = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_flujos_perdidas, text="Flujos y Perdidas")
        self.tree_res_flujos_perdidas = self.crear_treeview(self.frame_flujos_perdidas)

        self.frame_tcsc = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_tcsc, text="FACTS TCSC por Linea")

        self.lbl_tcsc = tk.Label(
            self.frame_tcsc,
            text=(
                "Ecuacion aplicada:  Z_ij = R + jX   ->   Z'_ij = R + j(X - Xc)  "
                "con  Xc = 40 % * X\n"
                "Costo y Lambda calculados con las potencias REALES "
                "despues del flujo de carga."
            ),
            font=("Arial", 11),
            justify="left"
        )
        self.lbl_tcsc.pack(pady=8)

        self.tree_res_tcsc = self.crear_treeview(self.frame_tcsc)

        self.frame_barrido = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_barrido, text="Barrido de Compensacion")
        self.tree_res_barrido = self.crear_treeview(self.frame_barrido)

        self.frame_casos = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_casos, text="Comparativa de Casos")

        panel_casos = tk.Frame(self.frame_casos)
        panel_casos.pack(fill="x", padx=10, pady=(10, 0))

        tk.Label(
            panel_casos,
            text="Factor de demanda para el escenario estresado (%):",
            font=("Arial", 10, "bold")
        ).pack(side="left")

        self.ent_factor = tk.Entry(panel_casos, width=8)
        self.ent_factor.insert(0, "120")
        self.ent_factor.pack(side="left", padx=8)

        tk.Label(
            panel_casos,
            text="Compensacion  Xc = % de X:",
            font=("Arial", 10, "bold")
        ).pack(side="left", padx=(15, 0))

        self.ent_pct_xc = tk.Entry(panel_casos, width=8)
        self.ent_pct_xc.insert(0, f"{PCT_COMPENSACION*100:.0f}")
        self.ent_pct_xc.pack(side="left", padx=8)

        tk.Button(
            panel_casos,
            text="Generar tabla comparativa",
            bg="#8e44ad", fg="white", font=("Arial", 10, "bold"),
            command=self.generar_casos
        ).pack(side="left", padx=10)

        tk.Button(
            panel_casos,
            text="Exportar a CSV",
            bg="#34495e", fg="white", font=("Arial", 10),
            command=self.exportar_casos
        ).pack(side="left")

        self.lbl_casos = tk.Label(
            self.frame_casos,
            text=(
                "Ecuacion: Z'_ij = R + j(X - Xc)  con  Xc = k % * X\n"
                "Caso 1: demanda nominal sin FACTS  |  Caso 2: demanda estresada "
                "sin FACTS  |  Casos siguientes: un TCSC instalado en cada linea."
            ),
            font=("Arial", 10), justify="left", fg="#555555"
        )
        self.lbl_casos.pack(pady=8)

        self.tree_res_casos = self.crear_treeview(self.frame_casos)

        tk.Label(
            self.frame_casos,
            text="Detalle tecnico de cada caso",
            font=("Arial", 10, "bold")
        ).pack()

        self.tree_res_casos_det = self.crear_treeview(self.frame_casos)

        self.df_casos = None
        self.df_casos_det = None

        self.frame_graficas = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_graficas, text="Graficas")

    def crear_pestana_interactiva(self, titulo, columnas, datos_default, info_texto):
        frame_main = ttk.Frame(self.notebook)
        self.notebook.add(frame_main, text=titulo)

        panel_izq_container = tk.LabelFrame(
            frame_main, text="Panel de Edicion",
            padx=5, pady=5, font=("Arial", 10, "bold")
        )
        panel_izq_container.pack(side="left", fill="y", padx=10, pady=10)

        canvas_form = tk.Canvas(panel_izq_container, width=240, highlightthickness=0)
        scroll_form = ttk.Scrollbar(panel_izq_container, orient="vertical", command=canvas_form.yview)
        
        frame_form = tk.Frame(canvas_form)

        frame_form.bind(
            "<Configure>",
            lambda e: canvas_form.configure(scrollregion=canvas_form.bbox("all"))
        )

        canvas_form.create_window((0, 0), window=frame_form, anchor="nw")
        canvas_form.configure(yscrollcommand=scroll_form.set)

        canvas_form.pack(side="left", fill="both", expand=True)
        scroll_form.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            canvas_form.yview_scroll(int(-1 * (event.delta / 120)), "units")
            
        def bind_scroll(event):
            canvas_form.bind_all("<MouseWheel>", _on_mousewheel)
            
        def unbind_scroll(event):
            canvas_form.unbind_all("<MouseWheel>")
            
        panel_izq_container.bind("<Enter>", bind_scroll)
        panel_izq_container.bind("<Leave>", unbind_scroll)

        tk.Label(
            frame_form, text=info_texto, fg="#555555", justify="left"
        ).pack(anchor="w", pady=(0, 10))

        entries = {}
        for col in columnas:
            tk.Label(
                frame_form, text=f"{col}:", font=("Arial", 9, "bold")
            ).pack(anchor="w", pady=(2, 0))

            ent = tk.Entry(frame_form, width=25)
            ent.pack(fill="x", padx=5) 
            entries[col] = ent
            
            if col in ["P_neta_MW", "Q_neta_MVAR"]:
                ent.insert(0, "Auto-calculado")
                ent.config(state="readonly")

        def procesar_valores(vals):
            vals_limpios = []
            for i, v in enumerate(vals):
                val_str = str(v).strip().replace(',', '.') # Soporte para decimales con coma
                col_actual = columnas[i]
                
                if val_str == "" or val_str == "Auto-calculado" or val_str.lower() == "nan":
                    vals_limpios.append("0")
                else:
                    # Formateo estricto dependiendo del tipo de dato requerido
                    try:
                        # Columnas que estrictamente deben ser numeros enteros (sin decimales)
                        if col_actual in ["ID", "Tipo", "De", "A", "Nodo"]:
                            val_num = int(float(val_str))
                            vals_limpios.append(str(val_num))
                        # Columnas que son parametros fisicos de ingenieria (deben tener decimales)
                        else:
                            val_num = float(val_str)
                            vals_limpios.append(str(val_num))
                    except ValueError:
                        vals_limpios.append(val_str)
                    
            if "P_neta_MW" in columnas and "Q_neta_MVAR" in columnas:
                try:
                    p_gen = float(vals_limpios[columnas.index("P_gen_MW")])
                    q_gen = float(vals_limpios[columnas.index("Q_gen_MVAR")])
                    p_car = float(vals_limpios[columnas.index("P_carga_MW")])
                    q_car = float(vals_limpios[columnas.index("Q_carga_MVAR")])

                    vals_limpios[columnas.index("P_neta_MW")] = str(p_gen - p_car)
                    vals_limpios[columnas.index("Q_neta_MVAR")] = str(q_gen - q_car)
                except Exception:
                    pass
            return vals_limpios

        def limpiar_entradas():
            for col, ent in entries.items():
                estado_previo = ent.cget("state")
                if estado_previo == "readonly":
                    ent.config(state="normal")
                ent.delete(0, tk.END)
                if col in ["P_neta_MW", "Q_neta_MVAR"]:
                    ent.insert(0, "Auto-calculado")
                    ent.config(state="readonly")
                elif estado_previo == "readonly":
                    ent.config(state="readonly")

        def agregar_fila():
            vals = [entries[c].get() for c in columnas]
            vals = procesar_valores(vals)
            tree.insert("", tk.END, values=vals)
            limpiar_entradas()

        def cargar_fila():
            seleccion = tree.selection()
            if not seleccion:
                messagebox.showwarning("Atencion", "Haz clic sobre una fila de la tabla de la derecha primero.")
                return
            valores = tree.item(seleccion[0], "value")
            for i, col in enumerate(columnas):
                ent = entries[col]
                estado_previo = ent.cget("state")
                if estado_previo == "readonly":
                    ent.config(state="normal")
                ent.delete(0, tk.END)
                ent.insert(0, valores[i])
                if estado_previo == "readonly":
                    ent.config(state="readonly")

        def guardar_cambios():
            seleccion = tree.selection()
            if not seleccion:
                messagebox.showwarning("Atencion", "No hay fila seleccionada.\nHaz clic en la tabla de la derecha antes de Guardar.")
                return
            vals = [entries[c].get() for c in columnas]
            vals = procesar_valores(vals)
            tree.item(seleccion[0], values=vals)
            limpiar_entradas()
            tree.selection_remove(seleccion)

        def eliminar_fila():
            seleccion = tree.selection()
            if not seleccion:
                messagebox.showwarning("Atencion", "Haz clic sobre una fila de la tabla de la derecha primero.")
                return
            for item in seleccion:
                tree.delete(item)

        # -------------------------------------------------------------------------
        # FUNCION INTELIGENTE DE IMPORTACION EXCEL 
        # (Tolera nombres equivocados y mapea por posicion)
        # -------------------------------------------------------------------------
        def importar_excel():
            filepath = filedialog.askopenfilename(
                title=f"Seleccionar archivo Excel para {titulo}",
                filetypes=[("Archivos Excel", "*.xlsx *.xls")]
            )
            if not filepath:
                return

            respuesta = messagebox.askyesnocancel(
                "Modo de Importación", 
                "¿Deseas BORRAR los datos que ya están en la tabla y sustituirlos por los del archivo Excel?\n\n"
                "• SÍ: Borrar tabla y cargar Excel.\n"
                "• NO: Mantener actuales y agregar el Excel abajo.\n"
                "• CANCELAR: Abortar operación."
            )
            
            if respuesta is None:
                return
                
            if respuesta is True:
                for item in tree.get_children():
                    tree.delete(item)

            try:
                df_test = pd.read_excel(filepath)
                if df_test.empty:
                    messagebox.showwarning("Archivo vacío", "El archivo Excel no contiene datos.")
                    return
                
                try:
                    float(str(df_test.columns[0]).strip())
                    tiene_headers = False
                except ValueError:
                    tiene_headers = True
                    
                if not tiene_headers:
                    df_excel = pd.read_excel(filepath, header=None).fillna(0)
                else:
                    df_excel = df_test.fillna(0)
                
                excel_cols = [str(c).strip().lower() for c in df_excel.columns]
                
                count_agregados = 0
                for _, row in df_excel.iterrows():
                    vals = []
                    for i, col in enumerate(columnas):
                        col_lower = str(col).strip().lower()
                        
                        if tiene_headers and col_lower in excel_cols:
                            idx = excel_cols.index(col_lower)
                            val = str(row.iloc[idx])
                            
                        elif i < len(df_excel.columns) and col not in ["P_neta_MW", "Q_neta_MVAR"]:
                            val = str(row.iloc[i])
                            
                        else:
                            if col in ["P_neta_MW", "Q_neta_MVAR"]:
                                val = "Auto-calculado"
                            else:
                                val = "0"
                                
                        vals.append(val)

                    vals_limpios = procesar_valores(vals)
                    tree.insert("", tk.END, values=vals_limpios)
                    count_agregados += 1

                messagebox.showinfo(
                    "Importación exitosa", 
                    f"Se importaron {count_agregados} registros desde el archivo Excel."
                )
            except Exception as e:
                messagebox.showerror(
                    "Error de lectura", 
                    f"No se pudo procesar el archivo Excel.\nVerifique que no esté corrupto.\nDetalle: {str(e)}"
                )
        # -------------------------------------------------------------------------

        frame_btns = tk.Frame(frame_form)
        frame_btns.pack(fill="x", pady=(15, 10), padx=5)

        tk.Button(frame_btns, text="Agregar Nuevo", command=agregar_fila, bg="#2ecc71", fg="white", font=("Arial", 9)).pack(fill="x", pady=2)
        tk.Button(frame_btns, text="Cargar Seleccionado", command=cargar_fila, bg="#f39c12", fg="white", font=("Arial", 9)).pack(fill="x", pady=2)
        tk.Button(frame_btns, text="Guardar Cambios", command=guardar_cambios, bg="#3498db", fg="white", font=("Arial", 9)).pack(fill="x", pady=2)
        tk.Button(frame_btns, text="Eliminar", command=eliminar_fila, bg="#e74c3c", fg="white", font=("Arial", 9)).pack(fill="x", pady=2)
        tk.Button(frame_btns, text="Importar desde Excel", command=importar_excel, bg="#9b59b6", fg="white", font=("Arial", 9)).pack(fill="x", pady=(10, 2))

        frame_tabla = tk.LabelFrame(
            frame_main, text="Base de Datos Actual",
            padx=10, pady=10, font=("Arial", 10, "bold")
        )
        frame_tabla.pack(side="right", expand=True, fill="both", padx=10, pady=10)

        scroll_y = ttk.Scrollbar(frame_tabla, orient="vertical")
        scroll_y.pack(side="right", fill="y")

        scroll_x = ttk.Scrollbar(frame_tabla, orient="horizontal")
        scroll_x.pack(side="bottom", fill="x")

        tree = ttk.Treeview(
            frame_tabla, columns=columnas, show="headings",
            yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set
        )

        for col in columnas:
            tree.heading(col, text=col)
            tree.column(col, width=110, anchor="center")

        tree.pack(expand=True, fill="both")
        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)

        for fila in datos_default:
            tree.insert("", tk.END, values=fila)

        return tree

    def crear_treeview(self, parent):
        cont = ttk.Frame(parent)
        cont.pack(expand=True, fill="both", padx=10, pady=10)

        scroll_y = ttk.Scrollbar(cont, orient="vertical")
        scroll_y.pack(side="right", fill="y")

        scroll_x = ttk.Scrollbar(cont, orient="horizontal")
        scroll_x.pack(side="bottom", fill="x")

        tree = ttk.Treeview(
            cont, show="headings",
            yscrollcommand=scroll_y.set,
            xscrollcommand=scroll_x.set
        )
        tree.pack(expand=True, fill="both")

        scroll_y.config(command=tree.yview)
        scroll_x.config(command=tree.xview)

        tree.tag_configure("bueno", background="#d5f5e3")
        tree.tag_configure("medio", background="#fdebd0")
        tree.tag_configure("malo", background="#fadbd8")
        tree.tag_configure("trafo", background="#d6eaf8")

        return tree

    def cargar_df_en_tree(self, tree, df):
        tree.delete(*tree.get_children())

        if df is None or df.empty:
            tree["columns"] = ["Info"]
            tree.heading("Info", text="Info")
            tree.column("Info", anchor="center", width=400)
            tree.insert("", tk.END, values=["Sin datos disponibles"])
            return

        tree["columns"] = list(df.columns)

        for col in df.columns:
            tree.heading(col, text=col)
            ancho = 150 if len(str(col)) > 14 else 110
            tree.column(col, anchor="center", width=ancho)

        for _, row in df.iterrows():
            valores = list(row)
            tag = ""
            texto = " ".join(str(v) for v in valores)
            if "MUY RECOMENDABLE" in texto:
                tag = "bueno"
            elif "RECOMENDABLE" in texto:
                tag = "medio"
            elif "TRANSFORMADOR" in texto or "TRAFO" in texto:
                tag = "trafo"
            elif "NO VIABLE" in texto or "NO APORTA" in texto or "ALERTA" in texto:
                tag = "malo"

            tree.insert("", tk.END, values=valores, tags=(tag,) if tag else ())

    def obtener_df_de_tree(self, tree, columnas):
        datos = [tree.item(item)['values'] for item in tree.get_children()]
        df = pd.DataFrame(datos, columns=columnas)
        return df.apply(pd.to_numeric, errors='coerce')

    def leer_entradas(self):
        cols_nodos = ["ID", "Tipo", "V_init", "P_gen_MW", "Q_gen_MVAR", "P_carga_MW", "Q_carga_MVAR", "P_neta_MW", "Q_neta_MVAR"]
        cols_lineas = ["De", "A", "R", "X", "B", "Limite_MVA"]
        cols_gens = ["Nodo", "Precio_MWh", "Pmin_MW", "Pmax_MW"]

        df_nodos = self.obtener_df_de_tree(
            self.tree_input_nodos, cols_nodos
        ).sort_values(by="ID").reset_index(drop=True)

        df_lineas = self.obtener_df_de_tree(self.tree_input_lineas, cols_lineas)

        df_gens = self.obtener_df_de_tree(
            self.tree_input_gens, cols_gens
        ).sort_values(by="Nodo").reset_index(drop=True)

        return df_nodos, df_lineas, df_gens

    def generar_casos(self):
        try:
            try:
                factor = float(self.ent_factor.get()) / 100.0
            except ValueError:
                factor = 1.20

            if factor <= 0:
                factor = 1.0

            try:
                pct_xc = float(self.ent_pct_xc.get()) / 100.0
            except ValueError:
                pct_xc = PCT_COMPENSACION

            pct_xc = max(0.0, min(pct_xc, 0.95))

            df_nodos, df_lineas, df_gens = self.leer_entradas()

            df_casos, df_detalle, linea_crit = comparar_casos(
                df_nodos, df_lineas, df_gens,
                factor_demanda=factor,
                k_compensacion=pct_xc
            )

            self.df_casos = df_casos
            self.df_casos_det = df_detalle

            self.cargar_df_en_tree(self.tree_res_casos, df_casos)
            self.cargar_df_en_tree(self.tree_res_casos_det, df_detalle)

            mejores = df_detalle[df_detalle["Red sin sobrecarga"] == "SI"]
            mejores = mejores[mejores["FACTS"] != "No"]

            if not mejores.empty:
                fila = mejores.sort_values(by="Costo ($/h)").iloc[0]
                extra = (
                    f"  |  Mejor solucion con TCSC: {fila['FACTS']} "
                    f"(Xc = {fila['Compensacion (%)']} % X = {fila['Xc (pu)']} pu), "
                    f"cargabilidad maxima "
                    f"{fila['Cargabilidad max (%)']} %"
                )
            else:
                extra = "  |  Ningun TCSC elimina por completo la sobrecarga."

            self.lbl_casos.config(
                text=(
                    f"Ecuacion: Z'_ij = R + j(X - Xc)  con  Xc = {pct_xc*100:.0f} % * X\n"
                    f"Caso 1: demanda nominal sin FACTS. Caso 2: demanda al "
                    f"{factor*100:.0f} % sin FACTS (linea critica {linea_crit}). "
                    f"Casos siguientes: un TCSC por linea con "
                    f"Xc = {pct_xc*100:.0f} % * X." + extra
                ),
                fg="#145a32", font=("Arial", 10, "bold")
            )

            self.notebook.select(self.frame_casos)

        except Exception as e:
            messagebox.showerror(
                "Error en la comparativa",
                f"No se pudo construir la tabla de casos.\nDetalle: {str(e)}"
            )

    def exportar_casos(self):
        if self.df_casos is None:
            messagebox.showinfo(
                "Sin datos", "Genera primero la tabla comparativa."
            )
            return

        try:
            self.df_casos.to_csv(
                "comparativa_casos_tcsc.csv", index=False, encoding="utf-8-sig"
            )
            self.df_casos_det.to_csv(
                "comparativa_casos_tcsc_detalle.csv", index=False, encoding="utf-8-sig"
            )
            messagebox.showinfo(
                "Exportacion completa",
                "Archivos generados:\n"
                "comparativa_casos_tcsc.csv\n"
                "comparativa_casos_tcsc_detalle.csv"
            )
        except Exception as e:
            messagebox.showerror("Error al exportar", str(e))

    def dibujar_graficas(self, flujos_raw, perdidas_P, perdidas_Q, df_tcsc):
        for widget in self.frame_graficas.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(11, 7), dpi=100)

        ax1 = fig.add_subplot(231)
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(233)
        ax4 = fig.add_subplot(234)
        ax5 = fig.add_subplot(235)
        ax6 = fig.add_subplot(236)

        etiquetas = [f"{e}-{r}" for e, r, S in flujos_raw]
        P = [S.real for e, r, S in flujos_raw]
        Q = [S.imag for e, r, S in flujos_raw]
        Smag = [abs(S) for e, r, S in flujos_raw]

        bars1 = ax1.bar(etiquetas, P, color='#3498db')
        ax1.bar_label(bars1, fmt='%.2f', padding=3, fontsize=8)
        ax1.set_title("Flujo P (pu)")

        bars2 = ax2.bar(etiquetas, Q, color='#9b59b6')
        ax2.bar_label(bars2, fmt='%.2f', padding=3, fontsize=8)
        ax2.set_title("Flujo Q (pu)")

        bars3 = ax3.bar(etiquetas, Smag, color='#e67e22')
        ax3.bar_label(bars3, fmt='%.2f', padding=3, fontsize=8)
        ax3.set_title("Potencia Aparente |S| (pu)")

        bars4 = ax4.bar(['P', 'Q'], [perdidas_P, perdidas_Q],
                        color=['#e74c3c', '#2ecc71'], width=0.5)
        ax4.bar_label(bars4, fmt='%.4f', padding=3, fontsize=8)
        ax4.set_title("Perdidas Totales (pu)")

        if df_tcsc is not None and not df_tcsc.empty:
            lineas_tcsc = df_tcsc["Linea"].tolist()

            red = pd.to_numeric(
                df_tcsc["Reduccion Perdidas (MW)"], errors="coerce"
            ).fillna(0).tolist()

            bars5 = ax5.bar(lineas_tcsc, red, color='#16a085')
            ax5.bar_label(bars5, fmt='%.2f', padding=3, fontsize=8)
            ax5.axhline(0, color='black', linewidth=0.8)
            ax5.set_title("Reduccion de Perdidas con TCSC (MW)")
            ax5.tick_params(axis='x', rotation=45)

            c_sin = pd.to_numeric(
                df_tcsc["Carga sin TCSC (%)"], errors="coerce"
            ).fillna(0).to_numpy()

            c_con = pd.to_numeric(
                df_tcsc["Carga con TCSC (%)"], errors="coerce"
            ).fillna(0).to_numpy()

            x = np.arange(len(lineas_tcsc))

            bars6_1 = ax6.bar(x - 0.2, c_sin, width=0.4, label="Sin TCSC", color='#95a5a6')
            bars6_2 = ax6.bar(x + 0.2, c_con, width=0.4, label="Con TCSC", color='#2980b9')

            ax6.bar_label(bars6_1, fmt='%.1f', padding=3, fontsize=7, rotation=90)
            ax6.bar_label(bars6_2, fmt='%.1f', padding=3, fontsize=7, rotation=90)

            ax6.set_xticks(x)
            ax6.set_xticklabels(lineas_tcsc, rotation=45)
            ax6.axhline(100, color='red', linestyle='--', linewidth=0.9)
            ax6.set_title("Cargabilidad (%)")
            ax6.legend(fontsize=7)
        else:
            ax5.text(0.5, 0.5, "TCSC no evaluado", ha="center")
            ax6.text(0.5, 0.5, "TCSC no evaluado", ha="center")

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ymin, ymax = ax.get_ylim()
            if ymax > 0:
                ax.set_ylim(top=ymax * 1.18)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.frame_graficas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def generar_reporte_pdf(self):
        if self.ultima_ejecucion is None:
            messagebox.showwarning("Faltan datos", "Para emitir el reporte tecnico, ejecute primero el analisis de flujo de potencia.")
            return
            
        if not HAS_FPDF:
            messagebox.showerror(
                "Libreria faltante", 
                "Es necesario contar con el modulo 'fpdf' para emitir documentos.\n\n"
                "Ejecute en su consola:\n"
                "pip install fpdf"
            )
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("Archivos PDF", "*.pdf")],
            title="Guardar Reporte Técnico PDF"
        )
        if not filepath:
            return

        try:
            def limpiar_texto(texto):
                reemplazos = {
                    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ñ': 'n',
                    'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U', 'Ñ': 'N',
                    'ó': 'o', 'í': 'i'
                }
                for original, nuevo in reemplazos.items():
                    texto = texto.replace(original, nuevo)
                return texto

            base = self.ultima_ejecucion["base"]
            df_flujos = self.ultima_ejecucion["df_flujos"]
            df_voltajes = self.ultima_ejecucion["df_voltajes"]
            mejor_tcsc = self.ultima_ejecucion["mejor"]

            pdf = FPDF()
            pdf.add_page()
            
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="REPORTE TECNICO: ANALISIS DE FLUJO DE POTENCIA OPTIMO", ln=True, align='C')
            pdf.cell(200, 10, txt="Y COMPENSACION MEDIANTE DISPOSITIVOS FACTS (TCSC)", ln=True, align='C')
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(200, 10, txt="1. PARAMETROS OPERATIVOS DEL CASO BASE", ln=True)
            pdf.set_font("Arial", '', 11)
            
            texto_resumen = (
                f"- Convergencia del flujo de potencia: {'Exitosa' if base['convergio'] else 'Fallida'} "
                f"({base['iteraciones']} iteraciones requeridas)\n"
                f"- Demanda Total Suministrada: {base['demanda_base_mw'] + base['perdidas_P_mw']:.2f} MW\n"
                f"- Perdidas Activas Totales de la Red: {base['perdidas_P_mw']:.3f} MW\n"
                f"- Costo Operativo del Despacho: $ {base['costo']:,.2f} / h\n"
                f"- Costo Marginal del Sistema (Nodo Slack): $ {base['lambda']:.2f} / MWh\n"
            )
            pdf.multi_cell(0, 8, txt=limpiar_texto(texto_resumen))
            pdf.ln(2)

            pdf.set_font("Arial", 'B', 11)
            pdf.cell(200, 10, txt="2. EVALUACION DE RESTRICCIONES (LIMITES TERMICOS Y DE TENSION)", ln=True)
            pdf.set_font("Arial", '', 11)
            
            observaciones_restricciones = []
            
            # Se aplica conversion numerica segura para no tronar al encontrar textos como 'N/A' (Trafos)
            lineas_sobrecargadas = df_flujos[pd.to_numeric(df_flujos["Carga (%)"], errors='coerce').fillna(0) > 100]
            if lineas_sobrecargadas.empty:
                observaciones_restricciones.append("Cargabilidad termica: Todas las ramas operan dentro de su capacidad nominal (< 100%).")
            else:
                for _, row in lineas_sobrecargadas.iterrows():
                    observaciones_restricciones.append(
                        f"Congestion detectada: La rama {row['Linea']} registra una cargabilidad del {row['Carga (%)']}% "
                        f"respecto a su limite de transferencia de {row['Limite (MVA)']} MVA."
                    )
                    
            voltajes_fuera_limite = df_voltajes[(df_voltajes["Magnitud |V| (pu)"] < V_MIN_OP) | (df_voltajes["Magnitud |V| (pu)"] > V_MAX_OP)]
            if voltajes_fuera_limite.empty:
                observaciones_restricciones.append(f"Perfil de tension: Los nodos del sistema mantienen magnitudes dentro del limite normativo ({V_MIN_OP} - {V_MAX_OP} p.u.).")
            else:
                for _, row in voltajes_fuera_limite.iterrows():
                    v = row['Magnitud |V| (pu)']
                    tipo_violacion = "Caida de tension" if v < V_MIN_OP else "Sobrevoltaje"
                    observaciones_restricciones.append(
                        f"Violacion de perfil de tension: Nodo {int(row['Nodo'])} reporta magnitud de {v:.4f} p.u. ({tipo_violacion})."
                    )

            for obs in observaciones_restricciones:
                pdf.multi_cell(0, 7, txt=limpiar_texto("- " + obs))
            pdf.ln(5)

            pdf.set_font("Arial", 'B', 11)
            pdf.cell(200, 10, txt="3. IMPACTO DE COMPENSACION SERIE CONTROLADA POR TIRISTORES (TCSC)", ln=True)
            pdf.set_font("Arial", '', 11)
            
            if mejor_tcsc is None:
                pdf.multi_cell(0, 7, txt="Evaluacion FACTS declinada: El analisis de contingencia no recomienda intervencion para este escenario operativo o la evaluacion fue deshabilitada.")
            else:
                ahorro_costo = base['costo'] - mejor_tcsc['costo']
                reduccion_perdidas = base['perdidas_P_mw'] - mejor_tcsc['perdidas']
                
                txt_facts = (
                    f"Con base en el analisis de sensibilidad del Flujo de Potencia Optimo Continuo, "
                    f"se establece la ubicacion optima del dispositivo FACTS (TCSC) bajo los siguientes parametros operativos:\n\n"
                    f"Parametros de Instalacion:\n"
                    f"- Ubicacion topologica: Rama {mejor_tcsc['linea']}\n"
                    f"- Grado de compensacion serie capacitiva (k): {mejor_tcsc['k']*100:.0f}% de la reactancia natural de la linea.\n"
                    f"- Reactancia equivalente de la compensacion (Xc): {mejor_tcsc['Xc']:.5f} p.u.\n\n"
                    f"Impacto en el Sistema de Potencia:\n"
                )
                
                impactos = []
                
                if not lineas_sobrecargadas.empty:
                    if mejor_tcsc['sobrecarga_max'] <= 100.5:
                        impactos.append("Mitigacion de congestion: Resolucion completa de restricciones termicas. Redireccionamiento de flujos de potencia activa logrado con exito.")
                    else:
                        impactos.append(f"Mitigacion parcial de congestion: La sobrecarga maxima remanente en la red desciende al {mejor_tcsc['sobrecarga_max']:.2f}%.")
                else:
                    impactos.append(f"Capacidad de transferencia: La red mantiene operacion estable con una cargabilidad maxima en estado estacionario del {mejor_tcsc['sobrecarga_max']:.2f}%.")
                
                if ahorro_costo > 0:
                    impactos.append(f"Optimizacion de despacho: Disminucion de los costos marginales de operacion, con un ahorro equivalente a $ {ahorro_costo:,.2f} / h.")
                elif ahorro_costo < 0:
                    impactos.append(f"Ajuste por redispatch: Variacion operativa con un incremento marginal de costo equivalente a $ {abs(ahorro_costo):,.2f} / h para garantizar estabilidad.")
                else:
                    impactos.append("Costo operativo sin variacion significativa ante la integracion del dispositivo.")
                    
                if reduccion_perdidas > 0:
                    impactos.append(f"Eficiencia energetica: Atenuacion de las perdidas activas de transmision I^2R en {reduccion_perdidas:.3f} MW.")
                elif reduccion_perdidas < 0:
                    impactos.append(f"Efecto colateral: Incremento marginal en perdidas activas del sistema por redisribucion de la matriz de admitancia.")
                    
                pdf.multi_cell(0, 7, txt=limpiar_texto(txt_facts))
                for imp in impactos:
                    pdf.multi_cell(0, 7, txt=limpiar_texto("  * " + imp))

            pdf.output(filepath)
            messagebox.showinfo("Emision completada", f"Reporte tecnico emitido satisfactoriamente.\nRuta: {filepath}")

        except Exception as e:
            messagebox.showerror("Fallo en emision documental", f"Se produjo una excepcion durante la generacion del PDF.\nDetalle: {str(e)}")

    def ejecutar(self):
        try:
            df_nodos, df_lineas, df_gens = self.leer_entradas()

            (
                resumen,
                df_gens_res,
                df_voltajes,
                df_flujos,
                df_flujos_perdidas,
                flujos_raw,
                per_P,
                per_Q,
                df_tcsc,
                df_barrido,
                mejor,
                base
            ) = ejecutar_analisis_opf(
                df_nodos, df_lineas, df_gens,
                evaluar_facts=self.var_facts.get()
            )

            self.ultima_ejecucion = {
                "base": base,
                "df_flujos": df_flujos,
                "df_voltajes": df_voltajes,
                "mejor": mejor,
                "df_tcsc": df_tcsc
            }

            self.lbl_resumen.config(
                text=resumen, fg="#0052cc", font=("Arial", 12, "bold")
            )

            self.cargar_df_en_tree(self.tree_res_gens, df_gens_res)
            self.cargar_df_en_tree(self.tree_res_voltajes, df_voltajes)
            self.cargar_df_en_tree(self.tree_res_flujos, df_flujos)
            self.cargar_df_en_tree(self.tree_res_flujos_perdidas, df_flujos_perdidas)
            self.cargar_df_en_tree(self.tree_res_tcsc, df_tcsc)
            self.cargar_df_en_tree(self.tree_res_barrido, df_barrido)

            if mejor is not None:
                self.lbl_tcsc.config(
                    text=(
                        "Z_ij = R + jX   ->   Z'_ij = R + j(X - Xc)   con   "
                        f"Xc = {mejor['k']*100:.0f} % * X\n"
                        f"Mejor ubicacion: linea {mejor['linea']}, Xc = "
                        f"{mejor['Xc']:.5f} pu  |  |V| en [{mejor['V_min']:.4f}, "
                        f"{mejor['V_max']:.4f}] pu  |  Cargabilidad maxima del sistema: "
                        f"{mejor['sobrecarga_max']:.2f} %"
                    ),
                    fg="#145a32", font=("Arial", 11, "bold")
                )
            else:
                self.lbl_tcsc.config(
                    text="Evaluacion TCSC desactivada o sin resultados factibles.",
                    fg="#7b241c", font=("Arial", 11)
                )

            self.dibujar_graficas(flujos_raw, per_P, per_Q, df_tcsc)

            if self.var_facts.get():
                self.generar_casos()

            self.notebook.select(self.frame_resumen)

        except Exception as e:
            messagebox.showerror(
                "Error de Calculo",
                "Revisa que los datos sean numeros correctos.\n"
                f"Detalle: {str(e)}"
            )

if __name__ == "__main__":
    app = SimuladorOPF()
    app.mainloop()