# Respuestas del workshop RAG

Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ cálculo a mano + simulación (`numpy`, semilla 0).

**Cálculo a mano.** Con `mL = 1/ln(M)`, la probabilidad de que un vector alcance al menos la capa `k` es `P(l ≥ k) = e^(−k/mL) = M^(−k)`. Con `M = 16` y `N = 1.000.000`:

| Capa | Probabilidad | Esperados = N · 16^(−k) |
|---|---|---:|
| ≥ 1 | 1/16 = 0,0625 | 62.500 |
| ≥ 2 | 1/256 ≈ 0,0039 | ≈ 3.906 |
| ≥ 3 | 1/4096 ≈ 0,00024 | ≈ 244 |

**Simulación:**

```
capa >= 1: simulado= 62,675  teórico=   62,500
capa >= 2: simulado=  4,005  teórico=    3,906
capa >= 3: simulado=    270  teórico=      244
capa máxima observada: 5
```

Las diferencias caben en el ruido de muestreo: la desviación estándar de una binomial `√(N·p·(1−p))` es ≈ 242, 62 y 16 nodos para cada fila, y las diferencias observadas (175, 99 y 26) están entre 0,7 y 1,7 desviaciones. La capa máxima observada (5) coincide con `log₁₆(10⁶) ≈ 5`.

Un detalle que noté: la presentación de inserción dice que `l = 0` ocurre ~63 % de las veces y que la capa 1 tiene ~37 % de los nodos. Esos porcentajes corresponden a `mL = 1` (`e^(−1) ≈ 0,37`). Con `M = 16` y `mL = 1/ln 16`, el 93,75 % de los vectores se queda solo en la capa 0.

**Por qué conviene tener pocos hubs arriba.** Cada capa tiene unas `M` veces menos nodos que la de abajo, así que el descenso greedy empieza en un grafo diminuto con enlaces largos que acerca la búsqueda a la zona correcta en pocos saltos, y hay solo ~`log_M N` capas que atravesar. El trabajo por capa es casi constante, de modo que el costo total crece como `O(log N)` en vez de `O(N)`, y además las capas altas ocupan muy poca memoria extra.

## T2 · ef, k y el tamaño de la cola

_Evidencia:_ presentaciones [Búsqueda en HNSW](https://ia-eafit-rag.netlify.app/teoria/hnsw_busqueda) y [Small World y HNSW](https://ia-eafit-rag.netlify.app/teoria/hnsw_explainer).

1. **k = 10 con ef = 5.** `ef` es el tamaño de la cola de candidatos en la capa 0 y los `k` resultados se eligen *de esa cola*. Con solo 5 candidatos es imposible devolver 10 vecinos: la búsqueda devuelve menos resultados (o la librería sube `ef` a `k` por debajo). Por eso siempre `ef ≥ k`.
2. **Valor inicial.** La presentación de búsqueda recomienda empezar con `ef = 2·k` y subirlo solo si el recall es insuficiente; para RAG típico, `ef = 50–100` con `k = 5–10` da > 95 % de recall. La de Small World da la receta de producción `M = 16, ef_construction = 200, ef = 50`: si el recall es bajo, subir primero `ef`, porque es más barato que reconstruir el índice.
3. **Se puede cambiar `ef` sin reconstruir.** `ef` solo afecta cómo se explora el grafo en cada consulta y se puede fijar por búsqueda. `ef_construction` decidió qué aristas se crearon al insertar cada vector; cambiarlo solo tiene efecto si se vuelve a construir el grafo.

## T3 · Inserción frente a búsqueda

_Evidencia:_ presentaciones [Inserción](https://ia-eafit-rag.netlify.app/teoria/hnsw_insercion) y [Búsqueda](https://ia-eafit-rag.netlify.app/teoria/hnsw_busqueda).

|                                | Inserción | Búsqueda |
|--------------------------------|-----------|----------|
| ¿Se sortea la capa? | Sí, con `l = ⌊−ln(U)·mL⌋`. **Porqué:** hay que decidir en cuántas capas va a vivir el nuevo nodo; hacerlo al azar hace que los hubs no dependan del orden de inserción. | No. **Porqué:** la consulta no se guarda en el grafo; siempre se entra por el nodo de entrada global de la capa más alta. |
| Valor de ef en las capas altas | `ef = 1` desde la capa máxima hasta `l+1`. **Porqué:** solo se busca un buen punto de entrada, no vecinos definitivos. | `ef = 1` desde la capa máxima hasta la capa 1. **Porqué:** igual que en la inserción, basta con aproximarse a la zona correcta rápido. |
| Parámetro de la fase principal | `ef_construction`, en cada capa desde `l` hasta 0. **Porqué:** de esos candidatos salen los `M` vecinos que se conectan; más candidatos = aristas de mejor calidad. | `ef`, solo en la capa 0. **Porqué:** es donde están todos los nodos y donde se eligen los `k` mejores; controla el compromiso recall/latencia. |
| ¿Modifica el índice? | Sí: crea aristas bidireccionales y poda vecinos que superan `M_max`. **Porqué:** es la construcción del grafo. | No. **Porqué:** es una operación de solo lectura, por eso muchas consultas pueden correr en paralelo. |

En el notebook 03, `vectorstore.add_documents(all_docs)` dispara la **inserción** (cada chunk se convierte en vector y Qdrant lo conecta en su grafo HNSW) y `retriever.invoke(pregunta)` dispara la **búsqueda** (se calcula `embed_query` de la pregunta y se recorre el grafo para traer los `k` vecinos).

## T4 · El invariante de FAISSVectorStore

_Evidencia:_ presentación [FAISSVectorStore](https://ia-eafit-rag.netlify.app/teoria/faiss_explicado).

| Posición | Lo que FAISS cree que hay (vectores) | `self.documents` antes | `self.documents` después de borrar B |
|---:|---|---|---|
| 0 | vA | A | A |
| 1 | vB | B | **C** |
| 2 | vC | C | **D** |
| 3 | vD | D | — (no existe) |

1. La búsqueda sigue devolviendo los enteros `[1, 2, 3]` porque el índice FAISS no cambió. Ahora `documents[1]` es **C** (el vector que coincidió era el de B), `documents[2]` es **D** (el que coincidió era el de C) y `documents[3]` lanza **`IndexError`**, porque la lista solo tiene 3 elementos.
2. Las posiciones 1 y 2 son **errores silenciosos**: devuelven un documento real pero equivocado, con el score de otro, sin ninguna excepción. La posición 3 es la que **lanza la excepción**. Si el código tuviera un guard `idx < len(self.documents)`, el tercero también se volvería silencioso (se descartaría sin avisar).
3. `IndexFlatIP` no tiene operación de borrado, así que mantener dos estructuras paralelas sincronizadas queda en manos del programador. Bases vectoriales como Qdrant guardan el vector y su payload (texto y metadata) bajo el **mismo id de punto** y soportan borrado/actualización nativos, así que no hay listas paralelas que se puedan desalinear. Por eso en los notebooks `title`, `source` y `chunk_index` viajan dentro de la metadata del propio punto.

## T5 · El costo de la fuerza bruta

_Evidencia:_ presentación [Small World y HNSW](https://ia-eafit-rag.netlify.app/teoria/hnsw_explainer).

1. `768 × 10⁶` multiplicaciones por consulta × `100` consultas/s = **76.800.000.000 multiplicaciones por segundo**.
2. En notación científica: **7,68 × 10¹⁰ mult/s** (y otras tantas sumas). El problema no es solo el cómputo: cada consulta lee la matriz completa, `10⁶ × 768 × 4 bytes ≈ 3,07 GB`, así que 100 consultas/s exigen ~307 GB/s de ancho de banda de memoria, más de lo que da un servidor típico. Además el costo crece linealmente con `N`: con 10 millones de vectores sería 10 veces peor. Por eso «ser exacto» deja de ser viable.
3. En ANN se sacrifica la garantía de obtener exactamente los `k` vecinos verdaderos (recall < 100 %, típicamente ~95 %) y se gana velocidad de órdenes de magnitud, porque el costo pasa de `O(N·d)` a aproximadamente `O(log N)`.

---

# Bloque A · App RAG empresarial

Notebook: [`notebooks/02b_rag_app_nvidia_nim_colab.ipynb`](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) (ejecutado en Colab el 22-oct-2026).

**Modelos usados:** embeddings `nvidia/nemotron-3-embed-1b` (dimensión 2048) y generación **`nvidia/nemotron-3.5-lightning-30b-a3b`**. El modelo del curso, `nvidia/nemotron-3-super-120b-a12b`, devolvió `404 Function … Not found for account` (salió del catálogo gratuito de build.nvidia.com), así que la celda de configuración prueba una lista de modelos vigentes y usa el primero que responde. Además apagué el modo razonamiento (`with_thinking_mode(enabled=False)`) porque al principio el modelo gastaba los tokens razonando y devolvía respuestas vacías.

Separé `build_pipeline` en `build_index` (chunking + embeddings + Qdrant) y `build_chain` (retriever + prompt + LLM) para que cambiar el prompt o el umbral no vuelva a pagar embeddings.

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_ `evidencias/A1_gradio.png` · notebook 02b, sección 4.1 · índice demo: 4 documentos, 4 chunks.

Configuración original del curso: `chunk_size=400`, `overlap=40`, `k=3`, `score_threshold=0.3` y prompt original.

| # | Pregunta | Respuesta del RAG | Fuentes recuperadas | Top-1 (coseno) | ¿Correcta? |
|---|---|---|---|---|---|
| 1 | ¿Cuál es el precio del plan Enterprise de CloudSync Pro? | $49/mes por usuario con compromiso anual. | CloudSync Pro — Precios, Onboarding, Política de reembolsos | CloudSync Pro — Precios (0.479) | Sí |
| 2 | ¿Cuántos días pueden trabajar remotamente los empleados? | Hasta 3 días por semana. | Política de trabajo remoto, Onboarding, Política de reembolsos | Política de trabajo remoto (0.639) | Parcial: omite «con aprobación del jefe directo» |
| 3 | ¿Qué pasa con los reembolsos mayores a $100? | Requieren aprobación del supervisor y se procesan en 3 a 5 días hábiles al método de pago original. | Política de reembolsos, Política de trabajo remoto, CloudSync Pro — Precios | Política de reembolsos (0.644) | Sí |
| 4 | ¿Qué ocurre durante la primera semana de onboarding? | Se completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa. | Onboarding, Política de trabajo remoto, Política de reembolsos | Onboarding (0.435) | Sí |
| 5 | ¿Cuántos días de vacaciones pagadas tienen los empleados al año? *(fuera de contexto)* | No tengo esa información. | Política de trabajo remoto, CloudSync Pro — Precios, Onboarding | Política de trabajo remoto (0.255) | Sí: se negó |

_Mi análisis:_ las cuatro preguntas de ejemplo recuperaron el documento correcto en el primer lugar, y tres respuestas fueron completas. En la de trabajo remoto el modelo recortó la condición de aprobación del jefe, un detalle secundario pero importante en una política. Con la pregunta de vacaciones no inventó nada: respondió que no tenía la información, aunque igual le llegaron 3 fragmentos. Eso pasa porque con solo 4 chunks y `k=3` el retriever siempre devuelve casi todo el índice: el umbral 0.3 no filtró nada (ver A4), y la honestidad dependió por completo del prompt. El top-1 de la pregunta fuera de contexto (0.255) quedó claramente por debajo de las buenas (0.435–0.644).

## A2 · Tu propio corpus de documentos

_Evidencia:_ [`data/company_documents.json`](data/company_documents.json) · `evidencias/A2_corpus.png` · notebook 02b, sección 4.2.

**Corpus:** *Andina Bikes S.A.S.*, una tienda y taller de bicicletas inventada en Medellín. 10 documentos (catálogo, descuentos, cuotas, mantenimiento, garantía, devoluciones, horarios, envíos, turnos y movilidad sostenible) con dos claves de metadata: `department` (ventas, finanzas, taller, servicio_cliente, logistica, recursos_humanos) y `type` (catalogo, politica, servicio, informacion, procedimiento, beneficio).

**Ajuste de `chunk_size`.** Los documentos miden entre 312 y 572 caracteres (media 429) y cada uno tiene dos párrafos con subtemas distintos.

| chunk_size / overlap | Chunks | Documentos partidos |
|---|---:|---:|
| 400 / 40 | 16 | 6 |
| **500 / 50 (elegido)** | **12** | **2** |
| 700 / 70 | 10 | 0 |

Con 700 cada documento queda en un solo chunk y el embedding mezcla dos subtemas (p. ej. precios de los modelos y el regalo por compra); con 400 se parten párrafos que debían quedar juntos. Con 500 solo se parten los dos documentos más largos, justo en el salto de párrafo.

Índice: 10 documentos → 12 chunks. Prompt original, `k=3`, `score_threshold=0.3`.

| # | Pregunta | Respuesta del RAG | Fuentes recuperadas | Top-1 (coseno) | ¿Correcta? |
|---|---|---|---|---|---|
| 1 | ¿Cuánto cuesta la Palmas E-Bike y qué autonomía tiene? | $6.900.000 COP y una autonomía aproximada de 60 km por carga. | Catálogo y precios…, Garantía…, Servicios de mantenimiento… | Catálogo y precios de bicicletas urbanas (0.435) | Sí |
| 2 | ¿Quién debe aprobar un descuento del 15 %? | El gerente comercial, según la política de descuentos. | Política de descuentos, Turnos y horas extra…, Plan de pagos… | Política de descuentos (0.532) | Sí |
| 3 | ¿Qué interés tiene la financiación a 6 cuotas? | El interés mensual es del 1,5 %. | Plan de pagos a cuotas, Servicios de mantenimiento…, Envíos… | Plan de pagos a cuotas (0.478) | Sí |
| 4 | ¿Cuánto dura la garantía de la batería de la bicicleta eléctrica? | 2 años o 800 ciclos de carga, lo que ocurra primero. | Garantía…, Servicios de mantenimiento…, Política de devoluciones… | Garantía de bicicletas y componentes (0.604) | Sí |
| 5 | ¿Puedo devolver una bicicleta y recibir mi dinero de vuelta? | No: solo se cambia por otro modelo en los primeros 7 días y no hay reembolsos en dinero. | Política de devoluciones y cambios, Catálogo…, Plan de pagos… | Política de devoluciones y cambios (0.577) | Sí (omite el límite de 20 km) |
| 6 | ¿Venden patinetas eléctricas? *(fuera de contexto)* | «No, soy un modelo de lenguaje creado por investigadores de NVIDIA.» | Servicios de mantenimiento…, Catálogo…, Garantía… | Servicios de mantenimiento del taller (0.282) | **No**: respuesta sin sentido |
| 7 | ¿Cuál es el NIT de Andina Bikes? *(fuera de contexto)* | No tengo esa información. | Plan de pagos…, Catálogo…, Horarios y sedes | Plan de pagos a cuotas (0.439) | Sí: se negó |

_Mi análisis:_ las 5 preguntas con respuesta en el corpus fueron correctas y el chunk correcto siempre quedó en el primer lugar, así que con documentos cortos y bien separados por tema la recuperación no fue el cuello de botella. De las 2 preguntas fuera de contexto, el NIT se resolvió bien. Las patinetas produjeron una respuesta absurda: el modelo interpretó «¿Venden…?» como si le preguntaran qué es él y contestó con su identidad. No inventó un dato de la tienda, pero tampoco usó la fórmula de «no tengo esa información»; eso justifica endurecer el prompt en A3. Algo que no esperaba: la pregunta del NIT tuvo un top-1 de 0.439, más alto que la buena pregunta de la Palmas E-Bike (0.435). Mi explicación es que «Andina Bikes» aparece en varios chunks y el embedding empareja el nombre de la empresa aunque el dato pedido no exista.

## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_ notebook 02b, sección 4.3 (mismo índice, solo cambia el prompt).

```diff
- "Eres un asistente empresarial. Responde únicamente con información "
- "del contexto proporcionado. Si el contexto no contiene la respuesta, "
- "indica que no tienes esa información. Responde en español y de forma concisa."
+ "Eres un asistente empresarial. Responde únicamente con información "
+ "del contexto proporcionado. Cita siempre el título del documento entre "
+ "corchetes, por ejemplo [Política de reembolsos]. Si el contexto no alcanza, "
+ "responde exactamente: «No encuentro esa información en los documentos.» "
+ "Responde en español, en máximo 3 oraciones."
```

Preguntas: fácil «¿Qué interés tiene la financiación a 6 cuotas?», ambigua «¿Cuánto cuesta el mantenimiento?» (hay tres paquetes) y fuera de contexto «¿Venden patinetas eléctricas?».

| Tipo | Pregunta | Respuesta ANTES | Respuesta DESPUÉS | ¿Cita [título]? | ¿Frase fija? | Oraciones antes → después |
|---|---|---|---|---|---|---|
| Fácil | ¿Qué interés tiene la financiación a 6 cuotas? | El interés mensual es del 1,5 %. | Las compras a 6 cuotas tienen un interés del 1,5 % mensual. | No | No aplica | 1 → 1 |
| Ambigua | ¿Cuánto cuesta el mantenimiento? | Lista en Markdown con los tres paquetes ($60.000, $140.000 y $190.000 COP) más una nota adicional. | Varía según el tipo: básico $60.000, completo $140.000 y eléctricas $190.000 COP. [Servicios de mantenimiento del taller] | Sí | No aplica | 1 → 2 |
| Fuera de contexto | ¿Venden patinetas eléctricas? | «No, soy un modelo de lenguaje creado por investigadores de NVIDIA.» | No encuentro esa información en los documentos. | No aplica | **Sí** | 1 → 1 |

_Mi análisis:_ la instrucción con más efecto fue la **frase fija**. Convirtió la respuesta absurda de las patinetas en una negativa exacta y verificable, que además se puede detectar automáticamente en una evaluación. La restricción de **máximo 3 oraciones** también se cumplió: en la pregunta ambigua el modelo pasó de una lista con viñetas y una nota extra a un resumen compacto que sigue cubriendo los tres paquetes. La instrucción que el modelo **ignoró a medias** fue la de **citar siempre el título**: la usó en la ambigua, pero no en la fácil, aunque el prompt dice «siempre». Mi conclusión es que el modelo obedece mejor las instrucciones que cambian el contenido de la respuesta (qué decir cuando falta información, cuánto escribir) que las de formato. Si la cita fuera obligatoria, la agregaría en el código desde la metadata de los documentos recuperados, en vez de confiar en el LLM.

## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_ `evidencias/A4_scores.png` · notebook 02b, sección 4.4 y función `chat()`.

`build_pipeline` devuelve `(pipeline, chunks, vectorstore)` y `chat()` muestra cada fuente con su score coseno de `similarity_search_with_score(question, k=3)`, marcando con ✅ las que superan el umbral.

**Hallazgo al implementarlo (verificado en el código de `langchain-qdrant` 1.1.0).** El retriever con `similarity_score_threshold` no compara el umbral contra el coseno, sino contra una «relevancia» `(coseno + 1) / 2`. Entonces `score_threshold = 0.3` equivale a **coseno ≥ −0,4**, que en la práctica no descarta nada. La guía dice «descarta los que tengan score coseno menor a 0,3», pero no es lo que pasa. Por eso el notebook convierte mi umbral propuesto con `to_relevance()` antes de pasarlo al retriever, y la primera línea de la sección 4.4 imprime coseno crudo vs. relevancia para comprobarlo.

La comprobación en el notebook lo confirma: para la pregunta de la Palmas E-Bike, coseno crudo = 0.435 y relevancia usada por el retriever = 0.718 = (0.435 + 1) / 2.

| Tipo | Pregunta | Fuente 1 (coseno) | Fuente 2 (coseno) | Fuente 3 (coseno) |
|---|---|---|---|---|
| Buena | ¿Cuánto cuesta la Palmas E-Bike y qué autonomía tiene? | Catálogo y precios (0.435) | Garantía (0.426) | Servicios de mantenimiento (0.326) |
| Buena | ¿Quién debe aprobar un descuento del 15 %? | Política de descuentos (0.532) | Turnos y horas extra (0.197) | Plan de pagos a cuotas (0.183) |
| Buena | ¿Cuánto dura la garantía de la batería de la bicicleta eléctrica? | Garantía (0.604) | Servicios de mantenimiento (0.410) | Política de devoluciones (0.303) |
| Fuera | ¿Venden patinetas eléctricas? | Servicios de mantenimiento (0.282) | Catálogo y precios (0.275) | Garantía (0.170) |
| Fuera | ¿Cuál es el NIT de Andina Bikes? | Plan de pagos a cuotas (0.439) | Catálogo y precios (0.436) | Horarios y sedes (0.400) |

Top-1 mínimo de las preguntas buenas: **0.435**. Top-1 máximo de las fuera de contexto: **0.439** (el NIT).

_Mi análisis y umbral propuesto:_ no hay una separación limpia. El NIT supera por 0.004 a una pregunta buena, así que ningún umbral separa perfectamente los dos grupos. La regla automática de la celda tomó el máximo de las fuera de contexto (0.44), pero eso habría descartado la pregunta buena de la Palmas E-Bike (0.435). En una app de soporte prefiero perder un poco de filtrado antes que dejar sin contexto una pregunta legítima, porque el prompt de A3 ya resuelve las negativas. Por eso propongo **umbral coseno = 0.42**, que en el retriever se pasa como `score_threshold = (0.42 + 1) / 2 = 0.71`. Ese valor:
- deja pasar el top-1 de las tres preguntas buenas (0.435, 0.532 y 0.604);
- descarta todos los chunks de la pregunta de patinetas (máximo 0.282), que era justo la que hizo fallar al modelo en A2;
- deja pasar el NIT (0.439), que el prompt de A3 contesta con la frase fija.

Con solo 5 preguntas esta propuesta es frágil. Para fijarla de verdad la calibraría con más preguntas de cada tipo y miraría la distribución de los scores, no solo los extremos. También queda claro que el valor original del curso (0.3 de relevancia = coseno ≥ −0.4) no filtraba nada en ninguno de los dos corpus.

---

# Bloque B · RAG sobre papers

Notebook: [`notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb`](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) (ejecutado en Colab el 22-oct-2026). Mismos modelos que en el Bloque A: embeddings `nvidia/nemotron-3-embed-1b` (dimensión 2048) y generación `nvidia/nemotron-3.5-lightning-30b-a3b`. Índice base: 150 chunks (55 de Attention + 95 de RAG) con `chunk_size=900`, `overlap=120`.

Observación previa: MarkItDown extrae los PDFs **pegando muchas palabras sin espacio** (p. ej. «WeusedtheAdamoptimizer…», «EachWikipediaarticleissplitintodisjoint100-wordchunks»). Eso afecta tanto a los embeddings como a cualquier búsqueda por palabra clave, y lo tuve en cuenta al elegir las palabras clave de B1/B2 y C4.

La columna «auto» de las tablas es un chequeo por palabras clave del ground truth; la columna «Correcta» es mi revisión manual.

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_ notebook 03, Parte 5, Bonus y celda «B1 · Evaluación contra el ground truth».

| # | Pregunta (resumen) | Respuesta del RAG (resumen) | Auto | Correcta | Paper correcto | Top-5 recuperado (paper#chunk, coseno) |
|---|---|---|---|---|---|---|
| Q1 | BLEU del Transformer big en WMT14 En→De | 28.4 | sí | **Sí** (no menciona que supera en más de 2 BLEU al estado del arte) | Sí | attention#28 (0.71), #1 (0.66), #26 (0.58), #40 (0.52), #31 (0.50) |
| Q2 | Memoria paramétrica y no paramétrica en RAG | Seq2seq preentrenado como memoria paramétrica + índice denso de Wikipedia con un retriever neuronal como no paramétrica, entrenados end-to-end | parcial | **Parcial**: la idea es correcta, pero no nombra BART ni DPR | Sí | rag#1 (0.59), #8 (0.55), #32 (0.52), #7 (0.51), #50 (0.48) |
| Q3 | Propósito de 1/√dk | Con dk grande los productos punto crecen y empujan el softmax a gradientes muy pequeños | sí | **Sí** | Sí | attention#11 (0.55), #12 (0.48), #13 (0.38), #16 (0.30), #20 (0.28) |
| Q4 | RAG-Sequence vs RAG-Token | Sequence usa el mismo documento para toda la salida; Token puede usar uno distinto por token | sí | **Sí** (no menciona el costo computacional) | Sí | rag#10 (0.51), #11 (0.47), #1 (0.47), #50 (0.46), #7 (0.45) |
| Q5 | Cabezas y dk/dv del Transformer base | 8 cabezas, dk = dv = 64 | sí | **Sí** | Sí | attention#14 (0.55), #15 (0.49), #9 (0.49), #0 (0.47), #40 (0.46) |
| Bonus | Generador de RAG y relación con el Transformer | BART, que está basado en la arquitectura Transformer | sí | **Sí** | Sí, aparecen los dos papers | rag#7 (0.53), #11 (0.51), #10 (0.51), #12 (0.47), attention#0 (0.47) |

**Fragmentos de la bonus:** rag_paper.pdf chunks 7, 11, 10 y 12, y attention_paper.pdf chunk 0. **Sí aparecieron los dos papers**, aunque 4 de los 5 fragmentos son del paper de RAG y el único de Attention es el chunk 0 (título, autores y abstract).

_Mi análisis:_ 5 de 6 respuestas fueron correctas y 1 parcial, y en todas el contexto vino del paper correcto: en Q1–Q5 los 5 chunks recuperados son del paper esperado. La única parcial (Q2) no es una alucinación sino falta de especificidad: el modelo describió la arquitectura con las palabras genéricas del abstract («pre-trained seq2seq model», «neural retriever») sin bajar a los nombres BART y DPR, que están en la sección 2, en otros chunks. En la bonus, la parte de Attention entró solo a través del abstract; alcanzó porque bastaba con saber que el Transformer existe, pero muestra que una pregunta que cruza papers queda sesgada hacia el paper que más se parece a la pregunta. Los scores más altos (0.71 en Q1) aparecen cuando la pregunta repite términos literales del texto, como «BLEU», «WMT 2014» y «English-to-German».

## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ notebook 03, celdas «B2» (`build_store`, `build_chain`, tabla de configuraciones).

Número de chunks por configuración:

| chunk_size / overlap | Attention | RAG | Total |
|---|---:|---:|---:|
| 400 / 52 | 121 | 205 | 326 |
| 900 / 120 | 55 | 95 | 150 |
| 1500 / 195 | 38 | 58 | 96 |

Por el hallazgo de A4, los umbrales 0.15, 0.25 y 0.40 que pide el enunciado están en la escala de relevancia `(coseno+1)/2`, es decir, coseno ≥ −0.70, −0.50 y −0.20. Agregué el experimento **3b** con umbrales en coseno real (0.15, 0.25, 0.40 → relevancia 0.575, 0.625, 0.70).

| Exp. | chunk_size | k | score_threshold | # chunks índice | Chunks recuperados (prom.) | Aciertos (de 5) | Parciales | Observación |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 · chunk_size | 400 | 5 | 0.250 | 326 | 5.0 | 4 | 1 | Q2 parcial |
| 1 · chunk_size | 900 | 5 | 0.250 | 150 | 5.0 | 4 | 1 | Q2 parcial |
| 1 · chunk_size | **1500** | 5 | 0.250 | 96 | 5.0 | **5** | 0 | todas correctas |
| 2 · k | 900 | 3 | 0.250 | 150 | 3.0 | 4 | 1 | Q2 parcial |
| 2 · k | 900 | 5 | 0.250 | 150 | 5.0 | 4 | 1 | Q2 parcial |
| 2 · k | 900 | 8 | 0.250 | 150 | 8.0 | 4 | 1 | Q2 parcial |
| 3 · threshold (relevancia) | 900 | 5 | 0.150 | 150 | 5.0 | 4 | 1 | Q2 parcial |
| 3 · threshold (relevancia) | 900 | 5 | 0.250 | 150 | 5.0 | 4 | 1 | Q2 parcial |
| 3 · threshold (relevancia) | 900 | 5 | 0.400 | 150 | 5.0 | 4 | 1 | Q2 parcial |
| 3b · threshold (coseno 0.15) | 900 | 5 | 0.575 | 150 | 5.0 | 4 | 1 | Q2 parcial |
| 3b · threshold (coseno 0.25) | 900 | 5 | 0.625 | 150 | 5.0 | 4 | 1 | Q2 parcial |
| 3b · threshold (coseno 0.40) | 900 | 5 | 0.700 | 150 | 4.4 | 4 | 1 | Q2 parcial; algunas preguntas reciben menos de 5 chunks |

_Mi conclusión:_
- El parámetro que más movió los resultados fue **`chunk_size`**: con 1500 caracteres Q2 pasó de parcial a correcta, porque un chunk grande junta en el mismo fragmento la descripción general de RAG y los nombres concretos (BART y DPR).
- **`k`** no cambió nada (3, 5 u 8 dan 4/5). Las respuestas de estas preguntas están en 1–2 chunks muy bien rankeados, así que agregar más contexto no aporta y solo gasta tokens.
- **`score_threshold`** en la escala del enunciado tampoco filtró nada: con 0.15, 0.25 y 0.40 siempre llegan los 5 chunks, lo que confirma el hallazgo de A4. Solo con un umbral real de coseno 0.40 empezaron a llegar menos chunks (4.4 en promedio), sin cambiar la calidad.
- Sí hay una configuración mejor que la del notebook: **1500 / 195, k=5**. Tiene un costo: cada consulta manda ~7.500 caracteres de contexto en vez de ~4.500, y con solo 5 preguntas la diferencia es de una respuesta, así que no es concluyente. Si quisiera bajar costo sin perder calidad, probaría 1500 con k=3.

## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_ notebook 03, celda «B3».

| # | Pregunta trampa | Chunks que llegaron al LLM (coseno) | Respuesta prompt v1 | Clasificación v1 | Respuesta prompt v2 | Clasificación v2 |
|---|---|---|---|---|---|---|
| T1 | What BLEU score does GPT-4 achieve on WMT 2014 English-to-German according to the paper? | attention#28 (0.56), #1 (0.55), #26 (0.49), #24 (0.42), #40 (0.41) | 28.4 | **Mezcla**: toma el BLEU real del Transformer y se lo atribuye a GPT-4 | The provided papers do not contain this information. | **Admite que no sabe** |
| T2 | By how many exact-match points does RAG outperform Llama 2 70B on Natural Questions? | rag#29 (0.54), #30 (0.50), #25 (0.48), #9 (0.47), #42 (0.46) | 11.8 | **Inventa**: Llama 2 no aparece en el paper | 25.0 | **Inventa**: otro número distinto |
| T3 | Which FlashAttention version does the Transformer use to reduce the memory cost of attention? | attention#5 (0.45), #6 (0.42), #7 (0.40), #15 (0.40), #1 (0.39) | «The Transformer does not use FlashAttention. It uses Multi-Head Attention to reduce the memory cost of attention.» | **Mezcla**: lo primero es cierto; lo segundo es falso (multi-head no se propone para reducir memoria) | The provided papers do not contain this information. | **Admite que no sabe** |

Como con el prompt original las tres respuestas inventaron o mezclaron, cambié el system prompt (v2):

```diff
- "Answer using only the supplied academic-paper excerpts. "
- "If the excerpts do not contain enough evidence, say so explicitly. "
+ "Answer using only the supplied academic-paper excerpts; never use outside knowledge. "
+ "Before answering, check that every model, dataset, number and name in the question "
+ "appears literally in the excerpts. If any of them is missing, reply exactly: "
+ "'The provided papers do not contain this information.' and nothing else. "
  "Answer in the same language as the question and be concise. "
  "Mention the paper title supporting the answer."
```

_Mi análisis:_ el prompt original **no admitió que no sabía en ninguna de las tres**. En todos los casos le llegaron 5 chunks con coseno entre 0.39 y 0.56 (scores parecidos a los de las preguntas legítimas de B1), así que el modelo tenía «material» y lo usó: en T1 reutilizó el 28.4 que está en los chunks, en T2 sacó un número de las tablas de resultados de RAG y en T3 justificó con un dato del paper algo que el paper no dice. El prompt v2, que obliga a verificar que cada entidad de la pregunta aparezca literalmente en el contexto, corrigió T1 y T3, pero **falló en T2**: hasta inventó un número diferente. Mi hipótesis es que en T2 los chunks recuperados sí contienen «RAG», «Natural Questions» y muchos números de exact match, y el modelo no detecta que la única entidad faltante es «Llama 2 70B». Como el umbral de similitud no separa estas preguntas (sus scores se parecen a los de las buenas), la verificación de entidades debería hacerse en código antes de llamar al LLM, por ejemplo extrayendo los nombres de modelos de la pregunta y comprobando que aparezcan en los chunks, en vez de confiar solo en el prompt.

## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ notebook 03, celda «B4».

Traducciones: Q1 «¿Qué puntaje BLEU obtuvo el Transformer (modelo big) en WMT 2014 inglés-alemán?», Q3 «¿Cuál es el propósito del factor de escala sqrt(dk) en la atención de producto punto escalado?», Q4 «¿Cuáles son las dos variantes del modelo RAG propuestas en el paper y en qué se diferencian?».

| # | Overlap top-5 (EN∩ES) | Coseno medio EN | Coseno medio ES | Top-1 EN | Top-1 ES | Correcta EN | Correcta ES (manual) | Respuesta ES |
|---|---:|---:|---:|---|---|---|---|---|
| Q1 | 4/5 | 0.592 | 0.584 | attention#28 (0.71) | attention#28 (0.71) | Sí | Sí | 28.4 |
| Q3 | 4/5 | 0.398 | 0.270 | attention#11 (0.55) | attention#11 (0.35) | Sí | Sí | Contrarrestar que los productos punto crezcan con dk grande y empujen el softmax a gradientes pequeños |
| Q4 | 5/5 | 0.474 | 0.453 | rag#10 (0.51) | rag#10 (0.49) | Sí | Sí | RAG-Sequence (mismo documento) y RAG-Token (documento distinto por token) |

El chequeo automático marcó Q3 y Q4 en español como «parcial», pero leyéndolas son correctas. El problema es que las palabras clave del ground truth están en inglés («magnitude», «same», «each token») y la respuesta llegó en español: eso es una limitación de mi evaluación, no del RAG.

Prueba de la hipótesis (coseno entre embeddings de pregunta):

| # | coseno(EN, ES) | coseno(EN, pregunta no relacionada) |
|---|---:|---:|
| Q1 | 0.957 | 0.001 |
| Q3 | 0.727 | 0.013 |
| Q4 | 0.979 | 0.004 |

_Mi hipótesis:_ `nemotron-3-embed-1b` es un modelo de embeddings **multilingüe** que proyecta una pregunta y su traducción casi al mismo punto del espacio: coseno 0.96–0.98 entre EN y ES, frente a ~0.00 con una pregunta sin relación. Por eso la búsqueda entre idiomas funciona: se recuperaron casi los mismos chunks (4/5, 4/5 y 5/5, con el mismo top-1 en los tres casos) y las respuestas en español fueron igual de correctas, incluso redactadas en español como pide el prompt. La alineación no es perfecta: Q3 tuvo el coseno EN–ES más bajo (0.727) y sus scores de recuperación bajaron de 0.398 a 0.270 en promedio. Creo que se debe a la traducción de términos técnicos como «Scaled Dot-Product Attention» → «atención de producto punto escalado», que no es una frase estándar en español. Para comprobarlo más a fondo: (1) repetiría con 20–30 preguntas y mediría el overlap promedio; (2) probaría variantes de Q3 dejando los términos técnicos en inglés dentro de la pregunta en español, y si el coseno sube, confirmaría que el problema es la terminología; (3) compararía con un modelo de embeddings solo en inglés, donde esperaría que el overlap caiga mucho.

## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ notebook 03, sección «B5» (tres celdas de código con su bloque `<details>`). Verifiqué cada ground truth en el texto de los PDFs.

| # | Pregunta | Ground truth | Respuesta del RAG | Correcta | Top-5 recuperado |
|---|---|---|---|---|---|
| B5-1 | Optimizador, β1, β2, ε y warmup del Transformer | Adam, β1 = 0.9, β2 = 0.98, ε = 10⁻⁹, `warmup_steps = 4000` (Attention §5.3) | Adam con β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹ y warmup_steps = 4000 | **Sí** | Solo Attention: #25 (0.52), #29, #39, #1, #27 |
| B5-2 | Dump de Wikipedia, cómo se parte y cuántos documentos | Dump de diciembre de 2018, fragmentos disjuntos de 100 palabras, 21M documentos (RAG §3) | Dump de diciembre de 2018, fragmentos disjuntos de 100 palabras, 21M documentos | **Sí** | Solo RAG: #18 (0.51), #1, #7, #42, #37 |
| B5-3 *(combina ambos papers)* | Parámetros del Transformer big vs parámetros entrenables de RAG | Transformer big: 213M (Attention Tabla 3). RAG: 626M (RAG Apéndice G). RAG ≈ 2,9× más grande | «El Transformer big tiene 11 mil millones de parámetros entrenables y RAG 626 millones; el Transformer big es más grande» | **No** | Solo RAG: #91 (0.60), #10, #93, #1, #7 |

_Mi análisis:_ las dos preguntas de un solo paper fueron perfectas, incluso con varios números exactos en la misma respuesta. La pregunta que combina los dos papers **falló, y de una forma muy instructiva**:
1. **La recuperación no trajo nada del paper de Attention.** Los 5 chunks son del paper de RAG, y el dato de 213M está en una fila de la Tabla 3 («| big | 6 | 1024 | … | 213 |»). Una fila de tabla convertida a texto, sin palabras como «parameters» cerca, genera un embedding que casi no se parece a la pregunta.
2. **El modelo llenó el hueco con un dato equivocado del otro paper.** El chunk rag#91 (el de mayor score, 0.60) es el Apéndice G, que menciona RAG con 626M y también **T5-11B, con 11 mil millones de parámetros**. El modelo confundió «T5-11B» con «el Transformer big» y concluyó lo contrario de la verdad.

El 626M de RAG sí es correcto. Este es el caso más peligroso: una respuesta mitad correcta, mitad inventada, con apariencia segura. Mejoras que probaría: subir `k` o usar MMR (reto C5) para obligar a traer chunks de ambos papers, filtrar por `metadata.source` y hacer una consulta por paper, y tratar las tablas aparte al indexar (por ejemplo, convertir cada fila en una oración con el encabezado de la columna).

---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ tablas de A1–A4 y B1–B5 de este documento.

**1. ¿Qué ocurrió cuando el contexto no contenía información suficiente? ¿El modelo reconoció la limitación o inventó?**
Dependió mucho de qué tan «tentador» era el contexto recuperado. Cuando la pregunta no tenía nada parecido en el corpus (vacaciones en A1, NIT en A2), el modelo dijo «No tengo esa información» incluso con el prompt original. Pero cuando los chunks recuperados contenían números y entidades parecidos, inventó o mezcló: en B3 las tres preguntas trampa fallaron con el prompt original (28.4 atribuido a GPT-4, un 11.8 inventado para Llama 2 y una afirmación falsa sobre FlashAttention), y en B5-3 confundió T5-11B con el Transformer big. El prompt v2, que exige verificar las entidades, corrigió 2 de las 3 trampas, y la frase fija de A3 convirtió una respuesta absurda («soy un modelo de NVIDIA») en una negativa exacta. Conclusión: el prompt ayuda mucho, pero no garantiza honestidad cuando el contexto se parece a la pregunta.

**2. ¿Qué fragmentos se recuperaron para la pregunta bonus? ¿Aparecieron ambos papers?**
Se recuperaron los chunks 7, 11, 10 y 12 de `rag_paper.pdf` y el chunk 0 de `attention_paper.pdf`, así que sí aparecieron los dos papers. Pero la representación fue desigual: 4 de 5 fragmentos son de RAG y el de Attention es el abstract, que entró con un score de 0.47, el más bajo del top-5. Alcanzó para la bonus porque la respuesta solo necesitaba saber que BART es un Transformer, algo que el propio paper de RAG ya dice. En B5-3, donde sí hacía falta un dato concreto de Attention (213M), ese paper no apareció y la respuesta falló. Una pregunta que cruza documentos no garantiza que el retriever traiga ambos.

**3. ¿Qué efecto tuvo modificar `chunk_size`, `chunk_overlap`, `k` o `score_threshold`?**
En B2, `chunk_size` fue el único parámetro con efecto: con 1500/195 (96 chunks) las 5 preguntas fueron correctas, contra 4/5 con 400 y 900, porque un chunk grande juntó la idea general de RAG con los nombres BART y DPR que le faltaban a Q2. Cambiar `k` entre 3, 5 y 8 no movió el resultado, porque la evidencia estaba en los primeros 1–2 chunks. Con `score_threshold` descubrí que `langchain-qdrant` lo compara contra `(coseno+1)/2`: los valores del curso (0.15–0.40) equivalen a coseno negativo y nunca filtraron nada; solo un umbral real de coseno 0.40 empezó a recortar chunks (4.4 en promedio). En mi corpus de A2, los documentos cortos con dos párrafos me llevaron a elegir 500/50 para que cada subtema quedara en su propio chunk, y con eso el chunk correcto quedó primero en las 5 preguntas.

**4. ¿Por qué conviene separar el modelo de embeddings del modelo generativo?**
Porque hacen trabajos distintos con costos distintos. El de embeddings se ejecuta sobre todos los chunks al indexar y sobre cada pregunta, y tiene que ser barato, rápido y estable. El generativo solo se llama una vez por respuesta y es donde está la calidad del lenguaje. Lo viví hoy: el modelo generativo del curso (`nemotron-3-super`) dejó de estar disponible y lo cambié por `nemotron-3.5-lightning` **sin tocar el índice**, porque los vectores dependen solo de `nemotron-3-embed-1b`. Si fueran el mismo modelo, cambiar el LLM habría obligado a reindexar todo. Además, cada uno se escoge por su propia virtud: el embedding multilingüe permitió buscar en inglés con preguntas en español (coseno EN–ES de 0.96–0.98 en B4), independientemente del idioma en que responda el LLM.

**5. ¿Qué cambiarías para llevar este prototipo a producción?**
- **Persistencia:** hoy el índice vive en `QdrantClient(":memory:")` y cada reinicio vuelve a pagar embeddings. Usaría un Qdrant persistente o un servidor (reto C6), con respaldos y versionado del índice por modelo de embeddings y `chunk_size`.
- **Control de acceso:** mi corpus ya tiene `department` en la metadata. Lo usaría para filtrar en Qdrant según el rol del usuario, de modo que alguien de ventas no recupere documentos de recursos humanos.
- **Evaluación:** un conjunto de preguntas con ground truth más grande, con métricas de recuperación (hit@k, reto C4) y de respuesta. Mi chequeo por palabras clave falló con respuestas en español (B4), así que agregaría revisión humana o un juez automático, e incluiría preguntas trampa como las de B3 en cada versión.
- **Observabilidad:** registrar por cada consulta los chunks recuperados con su score, el modelo usado y la latencia. Así se habría visto de inmediato que el umbral no filtraba nada, y también las respuestas que no citan fuente.
- **Costos y disponibilidad:** el plan gratuito tiene rate limit (vi varios 429) y los modelos cambian sin aviso (404 y 410 hoy). Pondría reintentos, una lista de modelos de respaldo como la que agregué, caché de respuestas frecuentes y un `chunk_size`/`k` elegido midiendo tokens por consulta, no solo aciertos.

---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

_Evidencia:_ [`notebooks/04_challenges.ipynb`](notebooks/04_challenges.ipynb), sección C1 (ejecutada, con gráficas). FAISS 1.15.1, 2 hilos de CPU, `d = 64`, 200 consultas, `k = 10`, `efConstruction = 100`. Las latencias dependen de la máquina; en Colab cambiarán los valores absolutos, no las tendencias.

**Barrido con n = 20.000** (búsqueda exacta: 0,215 ms/consulta):

| M | efSearch | recall@10 | ms/consulta | construcción (s) |
|---:|---:|---:|---:|---:|
| 8 | 10 | 0.168 | 0.005 | 0.88 |
| 8 | 16 | 0.242 | 0.006 | 0.88 |
| 8 | 32 | 0.355 | 0.011 | 0.88 |
| 8 | 64 | 0.510 | 0.022 | 0.88 |
| 8 | 128 | 0.683 | 0.068 | 0.88 |
| 8 | 256 | 0.819 | 0.083 | 0.88 |
| 16 | 10 | 0.329 | 0.007 | 1.12 |
| 16 | 16 | 0.425 | 0.011 | 1.12 |
| 16 | 32 | 0.599 | 0.020 | 1.12 |
| 16 | 64 | 0.770 | 0.036 | 1.12 |
| 16 | 128 | 0.905 | 0.073 | 1.12 |
| 16 | 256 | 0.971 | 0.132 | 1.12 |
| 32 | 10 | 0.468 | 0.010 | 1.35 |
| 32 | 16 | 0.579 | 0.011 | 1.35 |
| 32 | 32 | 0.761 | 0.019 | 1.35 |
| 32 | 64 | 0.909 | 0.035 | 1.35 |
| 32 | 128 | 0.983 | 0.069 | 1.35 |
| 32 | 256 | 0.998 | 0.143 | 1.35 |

**Barrido de n** (`M = 16`; menor `efSearch` de la grilla 16…2048 con recall ≥ 0,95; latencia en lote de 200 y consulta por consulta):

| n | exacto ms/q (lote) | exacto ms/q (1 a 1) | HNSW build (s) | efSearch p/ recall ≥ 0,95 | recall | HNSW ms/q (lote) | HNSW ms/q (1 a 1) | speedup 1 a 1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20.000 | 0.122 | 0.462 | 1.1 | 256 | 0.972 | 0.109 | 0.220 | 2.1× |
| 100.000 | 0.573 | 1.996 | 7.3 | 1024 | 0.974 | 0.654 | 1.358 | 1.5× |
| 500.000 | 2.730 | 9.127 | 47.8 | 2048 (no alcanza) | 0.939 | 2.357 | 4.841 | 1.9× |

_Mi respuesta / análisis:_

- **efSearch para recall ≥ 0,95 con M = 16 (n = 20.000): 256** (recall 0,971). Con `M = 32` basta `efSearch = 128` (0,983) y con `M = 8` ni siquiera 256 alcanza (0,819). Subir `M` desplaza toda la curva hacia arriba a cambio de algo más de tiempo de construcción (0,88 → 1,12 → 1,35 s) y más memoria por las aristas extra.
- **Recall vs latencia:** a igual latencia, `M = 32` domina a `M = 16` y esta a `M = 8`. Con n = 20.000 todas las configuraciones HNSW siguen por debajo de la línea roja de la búsqueda exacta, pero las de recall alto se le acercan.
- **¿Desde qué n gana HNSW?** Depende de cómo se mida. Consulta por consulta (como llega una pregunta a un RAG) HNSW ya es ~2× más rápido desde 20.000. En lote, la búsqueda exacta es una multiplicación de matrices muy optimizada (BLAS) y empata o gana: con 100.000 la exacta (0,57 ms) es más rápida que HNSW con el `efSearch = 1024` que necesita para llegar a 0,95. Lo más importante es que el `efSearch` necesario crece con `n` (256 → 1024 → > 2048), y eso se come la ventaja: con 500.000 vectores ni `efSearch = 2048` llega a 0,95. Con esta exigencia de recall, en datos aleatorios la ganancia de HNSW se queda en ~1,5–2×.
- **Por qué los datos aleatorios son un caso difícil.** Vectores uniformes en 64 dimensiones no tienen clusters ni estructura de baja dimensión, y por la concentración de distancias el vecino 10 está casi a la misma distancia que el 100. El grafo no tiene «barrios» que explotar y la búsqueda greedy necesita una cola enorme para no perder vecinos casi empatados. Los embeddings de texto, en cambio, se agrupan por tema y viven cerca de una variedad de dimensión intrínseca baja, así que espero recall ≥ 0,95 con `efSearch` mucho menor (del orden de 50–100, como dice la presentación) y una ventaja de HNSW mucho mayor a medida que crece `n`.

## C2 · Un tercer paper: el RAG que explica HNSW

No elegido.

## C3 · App Gradio para los papers con filtro por paper

No elegido.

## C4 · Evaluación automática de la recuperación

_Evidencia:_ [`eval/preguntas.csv`](eval/preguntas.csv) (10 preguntas) y `notebooks/04_challenges.ipynb`, sección C4 (función `hit_at_k`).

Elegí palabras clave robustas al texto que produce MarkItDown, que pega palabras sin espacios: «0.98», «P100», «21M», «626M», «dotproducts», «h = 8». La celda de control confirma que cada palabra clave aparece en al menos un chunk de la fuente esperada con los tres `chunk_size`. La recuperación es por similitud pura con `k`, sin umbral, para medir solo el ranking.

| chunk_size | # chunks | hit@3 | hit@5 |
|---:|---:|---:|---:|
| 400 | 326 | 90 % | 90 % |
| 900 | 150 | 90 % | 100 % |
| 1500 | 96 | 100 % | 100 % |

Posición del primer chunk correcto con `chunk_size=900`, `k=5`:

| id | Pregunta | Fuente | Posición |
|---|---|---|---:|
| 1 | Attention heads del Transformer base | attention | 1 |
| 2 | Las dos variantes de RAG | rag | 2 |
| 3 | BLEU del Transformer big | attention | 1 |
| 4 | Por qué se escalan los productos punto | attention | 1 |
| 5 | β2 de Adam | attention | **4** |
| 6 | Label smoothing | attention | 1 |
| 7 | GPUs de entrenamiento | attention | 1 |
| 8 | Documentos del índice de Wikipedia | rag | 2 |
| 9 | Arquitectura del retriever DPR | rag | 1 |
| 10 | Parámetros entrenables de RAG | rag | 1 |

_Mi análisis:_ la recuperación es buena en general: con 900 caracteres, 7 de 10 preguntas tienen el chunk correcto en el primer lugar. El único fallo de hit@3 con 900 es la pregunta de β2 de Adam: el chunk correcto aparece en el puesto 4, probablemente porque la sección de optimizador quedó partida y el texto pegado («WeusedtheAdamoptimizer…») hace que «Adam» y «beta2» no se reconozcan bien. El patrón coincide con B2: **chunks más grandes, mejor recuperación** (1500 da 100 % ya con k=3).

_Limitaciones de la métrica:_
1. **Favorece a los chunks grandes.** Un chunk de 1500 caracteres tiene más probabilidad de contener la palabra clave aunque no sea el fragmento más preciso, así que parte de la ventaja de 1500 puede ser un efecto de la métrica y no de una mejor recuperación.
2. **Falsos positivos y negativos.** Una palabra clave puede aparecer en un chunk que no responde la pregunta (por ejemplo, «P100» en una nota al pie o en la bibliografía), y el chunk correcto puede no tenerla de forma literal por cómo se extrajo el PDF.
3. **Mide solo la recuperación,** no si la respuesta generada es correcta: en B5-3 hubo chunks útiles y aun así el LLM mezcló datos.
4. **10 preguntas es poco:** cada pregunta vale 10 puntos porcentuales, así que 90 % vs 100 % es una sola pregunta.

## C5 · MMR frente a similitud pura

_Evidencia:_ `notebooks/04_challenges.ipynb`, sección C5 (`search_type="mmr"`, `k=5`, `fetch_k=20`, `lambda_mult=0.5`; índice de 900). Redundancia = coseno medio entre pares de los 5 chunks recuperados.

| Pregunta | Estrategia | Chunks (paper#chunk) | Redundancia |
|---|---|---|---:|
| What is multi-head attention? | Similitud | attention#14, #15, #11, #13, #7 | 0.520 |
| | MMR λ=0.5 | attention#14, **rag#78**, **attention#52**, attention#7, #11 | **0.390** |
| Which datasets and tasks were used to evaluate RAG? | Similitud | rag#1, #20, #25, #84, #24 | 0.526 |
| | MMR λ=0.5 | rag#1, #35, #41, #85, #94 | **0.388** |
| How does RAG compare to BART on Jeopardy question generation? | Similitud | rag#30, #35, #36, #21, #9 | 0.557 |
| | MMR λ=0.5 | rag#30, #21, #25, #36, #42 | **0.478** |

**Comparación lado a lado (pregunta 1).** Con similitud, los 5 chunks son de la sección 3.2 del paper de Attention: la definición de multi-head, sus tres usos, la Figura 2, la transición desde scaled dot-product y la motivación en la introducción. Todos son relevantes y complementarios. MMR conserva el top-1 (#14), pero reemplaza dos chunks útiles por **rag#78** (una entrada de la bibliografía del paper de RAG que cita «Attention is all you need») y **attention#52** (texto ilegible extraído de la figura de visualización de atención, con palabras invertidas como «tI si ni»).

_Mi conclusión:_ MMR **bajó la redundancia en las tres preguntas** (0.52→0.39, 0.53→0.39 y 0.56→0.48), así que sí hace lo que promete: más diversidad. Pero la diversidad no siempre es cobertura útil.
- **Ayuda más** en preguntas amplias o enumerativas, cuya respuesta está repartida por el documento. En «datasets y tareas» MMR trajo chunks de secciones más lejanas (#35, #41, #85, #94), donde aparecen distintos benchmarks, en vez de cinco fragmentos de la misma zona. También podría ayudar en preguntas que cruzan papers como B5-3, donde la similitud trajo solo chunks de RAG.
- **Empeora** en preguntas definicionales y concretas, donde los primeros chunks por similitud ya son todos relevantes. Penalizar que se parezcan entre sí obliga a meter ruido: bibliografía y basura de figuras, como pasó con multi-head attention.

Usaría MMR con un `lambda_mult` más alto (0.7–0.8) o solo en preguntas abiertas, y limpiaría del índice la bibliografía y el texto de figuras para que la diversidad no se llene de ruido.

## C6 · Persistencia del índice en disco

_Evidencia:_ `notebooks/04_challenges.ipynb`, sección C6 (`QdrantClient(path="./qdrant_data")` + `collection_exists`).

| Acción | Tiempo (s) | Puntos en la colección | Top-1 de prueba |
|---|---:|---:|---|
| Indexar desde cero (primera ejecución) | 6.89 | 150 | attention_paper.pdf#28 (0.700) |
| Reabrir sin reindexar (después de reiniciar el runtime) | 0.24 | 150 | attention_paper.pdf#28 (0.700) |

_Mi análisis:_ reabrir la colección fue **~29 veces más rápido** que indexarla (0.24 s vs 6.89 s), y sin llamar a la API de embeddings. Después de reiniciar el runtime, la colección tenía los mismos 150 puntos y la consulta de prueba devolvió exactamente el mismo chunk con el mismo score (0.700), así que el índice sobrevivió intacto. Con solo 150 chunks la diferencia son segundos, pero el costo de indexar crece con el corpus y con el rate limit de NVIDIA (40 consultas/min), mientras que reabrir se mantiene casi constante.

_Qué sigue faltando para producción:_
- **Durabilidad real:** en Colab la carpeta `./qdrant_data` se borra al terminar la sesión. Habría que montar Google Drive o, mejor, usar un servidor de Qdrant con respaldos y snapshots.
- **Concurrencia:** el modo local bloquea la carpeta para un solo cliente (por eso el notebook crea un único `persist_client`). Varios usuarios o procesos necesitan Qdrant como servicio.
- **Acceso:** un servidor con API key o autenticación, y filtros por metadata (por ejemplo `department`) para que cada usuario solo recupere lo que le corresponde.
- **Actualización y versionado:** cuando cambian los documentos hay que hacer upsert y borrado con ids estables. Además, si cambio el modelo de embeddings o el `chunk_size`, el índice queda inválido y hay que reconstruirlo; por eso guardaría esa configuración junto a la colección.
