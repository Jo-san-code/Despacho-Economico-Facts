# Despacho económico con dispositivos FACTS (TCSC)

Herramienta en Python para calcular el despacho económico de un sistema eléctrico de potencia, verificar su factibilidad técnica mediante un estudio de flujos de potencia y evaluar el efecto de un Compensador Serie Controlado por Tiristores (TCSC) instalado en cada línea de transmisión.

Desarrollada como parte del proyecto de investigación *«Despacho económico con sistemas de transmisión flexible en corriente alterna»*, Escuela Superior de Ingeniería Mecánica y Eléctrica, Unidad Zacatenco, Instituto Politécnico Nacional (2026).

- **Autores:** Santiago González Martínez y Josue Humberto Olvera Alvarez
- **Asesores:** Cain Álvarez García y Jaime Robles García

## Funciones principales

- Construcción de la matriz de admitancia nodal (Y<sub>bus</sub>) con el modelo π de las líneas.
- Despacho económico con funciones de costo cuadráticas y límites de generación, resuelto con el método lambda iterativo (bisección sobre λ) a partir de las condiciones de optimalidad de Lagrange.
- Flujo de potencia por el método de Gauss-Seidel (factor de aceleración 1.4 en nodos PQ, tolerancia 1×10⁻⁸ p.u.).
- Lazo de convergencia despacho–red: las pérdidas calculadas por el flujo de potencia se incorporan a la demanda del despacho hasta que convergen (tolerancia 1×10⁻⁵ MW).
- Evaluación del TCSC línea por línea (compensación nominal del 40 % de la reactancia) y barrido de compensación de 10 % a 70 %.
- Tabla comparativa de casos: demanda nominal, demanda escalada (por ejemplo, 120 % o 150 %) y un caso por cada línea compensada.
- Resultados de tensiones, flujos, cargabilidad y pérdidas, gráficas, exportación a CSV y reporte técnico en PDF.

## Requisitos

- Python 3.10 o superior.
- Tkinter (incluido en Python para Windows y macOS; en Linux: `sudo apt install python3-tk`).
- Librerías:

```bash
pip install -r requirements.txt
```

## Uso

```bash
python simulador_opf_tcsc.py
```

1. Capture o importe los datos en las pestañas **Nodos**, **Lineas** y **Generadores**. Al abrir el programa aparecen cargados los datos del sistema de 9 nodos.
2. Presione **EJECUTAR ANALISIS OPF + TCSC**. Si la casilla «Evaluar TCSC en cada linea» está activa, también se evalúa el compensador en cada línea.
3. Revise los resultados en las pestañas *Resumen y Despacho*, *Voltajes*, *Flujos y Cargabilidad*, *Flujos y Perdidas*, *FACTS TCSC por Linea*, *Barrido de Compensacion* y *Graficas*.
4. En *Comparativa de Casos*, indique el porcentaje de demanda del escenario estresado y el porcentaje de compensación, y genere la tabla. Puede exportarla a CSV.
5. Con **GENERAR REPORTE PDF** se obtiene un reporte técnico del último análisis.

## Formato de los datos

Todas las impedancias se expresan en por unidad sobre una base de 100 MVA.

**Nodos**

| Columna | Descripción |
|---|---|
| `ID` | Número de nodo (consecutivo, empezando en 1) |
| `Tipo` | 0 = slack, 1 = generación (PV), 2 = carga (PQ) |
| `V_init` | Magnitud de tensión inicial en p.u.; en nodos slack y PV es la tensión especificada |
| `P_gen_MW`, `Q_gen_MVAR` | Generación del nodo. En nodos PV con generador despachable, la P la asigna el despacho |
| `P_carga_MW`, `Q_carga_MVAR` | Demanda del nodo |

Las columnas `P_neta_MW` y `Q_neta_MVAR` se calculan automáticamente.

**Líneas**

| Columna | Descripción |
|---|---|
| `De`, `A` | Nodos de envío y recepción |
| `R`, `X` | Resistencia y reactancia serie (p.u.) |
| `B` | Susceptancia **total** de la línea (p.u.); el programa coloca B/2 en cada extremo |
| `Limite_MVA` | Límite térmico de la línea |

Las ramas con `R = 0` se identifican como transformadores: no se les asigna límite térmico ni se evalúa en ellas el TCSC.

**Generadores**

| Columna | Descripción |
|---|---|
| `Nodo` | Nodo donde se conecta la unidad |
| `Precio_MWh` | Coeficiente lineal de costo b ($/MWh) |
| `Pmin_MW`, `Pmax_MW` | Límites de generación |

La función de costo de cada unidad es C = a + bP + cP², donde a y c se calculan a partir del precio b con las relaciones a = 1500/b + 20 y c = 0.0004·b.

## Cargar otro sistema de potencia

1. Prepare tres archivos de Excel (`nodos.xlsx`, `lineas.xlsx` y `generadores.xlsx`) con los encabezados indicados arriba en la primera fila. Puede usar como plantilla los de `datos/sistema_9_nodos/`.
2. En cada pestaña, presione **Importar desde Excel** y elija el archivo correspondiente. Elija «Sí» para reemplazar los datos actuales.
3. Ejecute el análisis.

## Supuestos del modelo

- Potencia base: 100 MVA. Banda de tensión de referencia: 0.95–1.05 p.u.
- El TCSC se representa como una reactancia capacitiva fija: X' = X − k·X, con k = 0.40 por defecto.
- Los límites térmicos de las líneas se verifican con los resultados del flujo de potencia, pero no se imponen como restricción en el despacho.
- No se consideran límites de potencia reactiva en los nodos PV.

## Cómo citar

S. González Martínez y J. H. Olvera Alvarez, *Herramienta de despacho económico con dispositivos FACTS (TCSC)*, versión 1.0, ESIME Zacatenco, IPN, 2026. [Código fuente]. Disponible en: https://github.com/Jo-san-code/Despacho-Economico-Facts
