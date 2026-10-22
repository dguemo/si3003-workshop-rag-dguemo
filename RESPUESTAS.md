# Respuestas del workshop RAG

Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

> Las secciones marcadas con ⏳ dependen de las llamadas a NVIDIA NIM: se completan con la tabla que imprime la celda indicada del notebook al ejecutarlo en Colab.

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

Notebook: [`notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb`](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb). Índice base: 150 chunks (55 de Attention + 95 de RAG) con `chunk_size=900`, `overlap=120`.

Observación previa: MarkItDown extrae los PDFs **pegando muchas palabras sin espacio** (p. ej. «WeusedtheAdamoptimizer…», «EachWikipediaarticleissplitintodisjoint100-wordchunks»). Eso afecta tanto a los embeddings como a cualquier búsqueda por palabra clave, y lo tuve en cuenta al elegir las palabras clave de B1/B2 y C4.

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_ notebook 03, celda «B1 · Evaluación contra el ground truth».

⏳ *Pegar la tabla de 6 filas (Q1–Q5 y Bonus) y la lista de fuentes del bonus.*

| # | Pregunta (resumen) | Respuesta del RAG (resumen) | Correcta | Paper correcto |
|---|---|---|---|---|
| Q1 | BLEU del Transformer big en WMT14 En→De | ⏳ | ⏳ | ⏳ |
| Q2 | Memoria paramétrica y no paramétrica en RAG | ⏳ | ⏳ | ⏳ |
| Q3 | Propósito de 1/√dk | ⏳ | ⏳ | ⏳ |
| Q4 | RAG-Sequence vs RAG-Token | ⏳ | ⏳ | ⏳ |
| Q5 | Cabezas y dk/dv del Transformer base | ⏳ | ⏳ | ⏳ |
| Bonus | Generador de RAG y relación con el Transformer | ⏳ | ⏳ | ⏳ |

_Mi respuesta / análisis:_ ⏳

## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ notebook 03, celdas «B2» (`build_store`, `build_chain`, tabla de configuraciones).

Número de chunks por configuración (calculado con el mismo splitter y filtro de ≥ 30 caracteres):

| chunk_size / overlap | Attention | RAG | Total |
|---|---:|---:|---:|
| 400 / 52 | 121 | 205 | 326 |
| 900 / 120 | 55 | 95 | 150 |
| 1500 / 195 | 38 | 58 | 96 |

Por el hallazgo de A4, los umbrales 0.15, 0.25 y 0.40 que pide el enunciado equivalen a coseno ≥ −0,70, −0,50 y −0,20; casi seguro no filtran nada y las tres filas deberían salir iguales. Agregué el experimento **3b** con umbrales en coseno real (0.15, 0.25, 0.40 → relevancia 0.575, 0.625, 0.70) para ver el efecto que el enunciado esperaba.

⏳ *Pegar la tabla de configuraciones (aciertos de 5, parciales, chunks recuperados promedio, observación).*

_Mi conclusión:_ ⏳ (5–8 líneas: qué parámetro movió más los resultados y si hay una configuración mejor que 900/120, k=5, 0.25).

## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_ notebook 03, celda «B3».

Preguntas (plausibles, pero sobre modelos o técnicas posteriores a los papers):
1. *What BLEU score does GPT-4 achieve on WMT 2014 English-to-German according to the paper?*
2. *By how many exact-match points does RAG outperform Llama 2 70B on Natural Questions?*
3. *Which FlashAttention version does the Transformer use to reduce the memory cost of attention?*

Prompt v2 (se prueba en las tres preguntas):

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

⏳ *Pegar la tabla (chunks que llegaron al LLM, respuesta y clasificación con prompt v1 y v2).*

_Mi respuesta / análisis:_ ⏳ clasificar cada una como admite / inventa / mezcla.

## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ notebook 03, celda «B4».

Traducciones usadas: Q1 «¿Qué puntaje BLEU obtuvo el Transformer (modelo big) en WMT 2014 inglés-alemán?», Q3 «¿Cuál es el propósito del factor de escala sqrt(dk) en la atención de producto punto escalado?», Q4 «¿Cuáles son las dos variantes del modelo RAG propuestas en el paper y en qué se diferencian?».

**Cómo compruebo la hipótesis:** además de comparar los top-5 chunks EN vs ES, la celda mide el coseno entre el embedding de la pregunta en inglés y el de su traducción, y lo compara con el de una pregunta no relacionada. Si el modelo es multilingüe, `coseno(EN, ES)` debe ser claramente mayor que `coseno(EN, no relacionada)`.

⏳ *Pegar la tabla EN vs ES y la tabla de cosenos.*

_Mi hipótesis:_ ⏳

## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ notebook 03, sección «B5» (tres celdas de código con su bloque `<details>`). Verifiqué cada ground truth en el texto de los PDFs.

| # | Pregunta | Ground truth | Sección |
|---|---|---|---|
| B5-1 | Optimizador, β1, β2, ε y warmup del Transformer | Adam, β1 = 0.9, β2 = 0.98, ε = 10⁻⁹, `warmup_steps = 4000` | Attention, §5.3 |
| B5-2 | Dump de Wikipedia, cómo se parte y cuántos documentos | Dump de diciembre de 2018, fragmentos disjuntos de 100 palabras, 21 millones de documentos (FAISS + HNSW) | RAG, §3 |
| B5-3 (combina ambos) | Parámetros del Transformer big vs parámetros entrenables de RAG | 213M vs 626M (110M + 110M de BERT en DPR + 406M de BART-large); RAG ≈ 2,9× más grande | Attention Tabla 3 + RAG Apéndice G |

⏳ *Pegar la tabla de resultados (respuesta, correcta, papers en el top-5).*

_Mi respuesta / análisis:_ ⏳

---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ tablas de A1–A4 y B1–B5.

⏳ *Responder con los resultados de Colab (mínimo 3 oraciones cada una):*

1. ¿Qué ocurrió cuando el contexto no contenía información suficiente? ¿El modelo reconoció la limitación o inventó?
2. ¿Qué fragmentos se recuperaron para la pregunta bonus? ¿Aparecieron ambos papers?
3. ¿Qué efecto tuvo modificar `chunk_size`, `chunk_overlap`, `k` o `score_threshold`?
4. ¿Por qué conviene separar el modelo de embeddings del modelo generativo?
5. ¿Qué cambiarías para llevar este prototipo a producción? (persistencia, control de acceso, evaluación, observabilidad y costos)

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

_Evidencia:_ [`eval/preguntas.csv`](eval/preguntas.csv) (10 preguntas), `notebooks/04_challenges.ipynb`, sección C4 (función `hit_at_k`).

Elegí palabras clave robustas al texto que produce MarkItDown, que pega palabras sin espacios: «0.98», «P100», «21M», «626M», «dotproducts», «h = 8». Comprobé que cada palabra clave aparece en al menos un chunk de la fuente esperada con los tres `chunk_size` (control incluido en el notebook). La recuperación es por similitud pura con `k`, sin umbral, para medir solo el ranking.

⏳ *Pegar la tabla `hit@3` / `hit@5` por `chunk_size` y el detalle por pregunta.*

_Limitaciones de la métrica:_ ⏳ (al menos dos; candidatas: una palabra clave puede aparecer en un chunk que no responde la pregunta —falso positivo— y el chunk correcto puede no contener la palabra literal o tenerla partida por la extracción del PDF —falso negativo—; además mide solo recuperación, no si la respuesta generada es correcta).

## C5 · MMR frente a similitud pura

_Evidencia:_ `notebooks/04_challenges.ipynb`, sección C5 (`search_type="mmr"`, `k=5`, `fetch_k=20`, `lambda_mult=0.5`). Redundancia = coseno medio entre pares de los 5 chunks recuperados.

⏳ *Pegar la tabla de 3 preguntas × 2 estrategias y la comparación lado a lado.*

_Conclusión:_ ⏳

## C6 · Persistencia del índice en disco

_Evidencia:_ `notebooks/04_challenges.ipynb`, sección C6 (`QdrantClient(path="./qdrant_data")` + `collection_exists`).

⏳ *Pegar las dos filas de la tabla de tiempos: «indexar desde cero» (primera ejecución) y «reabrir (sin reindexar)» (después de reiniciar el runtime).*

_Qué sigue faltando para producción:_ ⏳
