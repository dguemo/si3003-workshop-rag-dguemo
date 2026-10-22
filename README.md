# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** David Guerra Morales
- **Código estudiantil:** …
- **Usuario de GitHub:** dguemo
- **Fecha:** 22 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

✅ hecha · 🟡 código listo en el notebook, falta ejecutar en Colab y pegar resultados · ⬜ no elegida

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | [RESPUESTAS.md#T1](RESPUESTAS.md#t1--cuántos-nodos-viven-en-cada-capa) |
| T2 | ef, k y el tamaño de la cola | 2 | ✅ | [RESPUESTAS.md#T2](RESPUESTAS.md#t2--ef-k-y-el-tamaño-de-la-cola) |
| T3 | Inserción frente a búsqueda | 2 | ✅ | [RESPUESTAS.md#T3](RESPUESTAS.md#t3--inserción-frente-a-búsqueda) |
| T4 | El invariante de FAISSVectorStore | 2 | ✅ | [RESPUESTAS.md#T4](RESPUESTAS.md#t4--el-invariante-de-faissvectorstore) |
| T5 | El costo de la fuerza bruta | 2 | ✅ | [RESPUESTAS.md#T5](RESPUESTAS.md#t5--el-costo-de-la-fuerza-bruta) |
| A1 | Ejecuta la app RAG de punta a punta | 4 | 🟡 | [02b §4.1](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) · `evidencias/A1_gradio.png` |
| A2 | Tu propio corpus de documentos | 6 | 🟡 | [data/company_documents.json](data/company_documents.json) · [02b §4.2](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) · `evidencias/A2_corpus.png` |
| A3 | Ingeniería del prompt (antes y después) | 5 | ✅ | [02b §4.3](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) |
| A4 | Muestra el score de similitud en las fuentes | 5 | 🟡 | [02b §4.4 y chat()](notebooks/02b_rag_app_nvidia_nim_colab.ipynb) · `evidencias/A4_scores.png` |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | ✅ | [03 · B1](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | ✅ | [03 · B2](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ✅ | [03 · B3](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| B4 | Preguntas en español sobre papers en inglés | 5 | ✅ | [03 · B4](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| B5 | Escribe tu propio quiz con ground truth | 5 | ✅ | [03 · B5](notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb) |
| R1 | Reflexión final con tus datos | 10 | ✅ | [RESPUESTAS.md#Reflexión](RESPUESTAS.md#reflexión) |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ✅ | [04 · C1](notebooks/04_challenges.ipynb) · [RESPUESTAS.md#C1](RESPUESTAS.md#c1--hnsw-contra-fuerza-bruta-con-faiss) |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ⬜ | |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ⬜ | |
| C4 | Evaluación automática de la recuperación (reto) | 12 | 🟡 | [eval/preguntas.csv](eval/preguntas.csv) · [04 · C4](notebooks/04_challenges.ipynb) |
| C5 | MMR frente a similitud pura (reto) | 8 | 🟡 | [04 · C5](notebooks/04_challenges.ipynb) |
| C6 | Persistencia del índice en disco (reto) | 8 | 🟡 | [04 · C6](notebooks/04_challenges.ipynb) |

**Autoevaluación:** … / 100 puntos (núcleo 70 + retos con tope de 30: C1 + C4 + C5 + C6 = 43).

## Cómo ejecutar mis notebooks

1. Abre cada notebook de `notebooks/` en Google Colab.
2. Crea el secreto `NVIDIA_API_KEY` en Colab (panel Secrets) con acceso para el notebook.
3. Ejecuta las celdas en orden. El 02b descarga solo `data/company_documents.json` y el 04 descarga `eval/preguntas.csv` desde este repositorio si no están en la sesión.

> No incluyas tu API key en ningún archivo de este repositorio.

## Decisiones y aprendizajes

- El `score_threshold` del retriever de `langchain-qdrant` no se compara contra el coseno sino contra `(coseno + 1) / 2`: el 0.3 del notebook equivale a coseno ≥ −0,4 y casi no filtra. Elegir el umbral con los scores crudos (A4) y convertirlo evita ese error.
- Separar el índice (`build_index` / `build_store`) de la cadena (`build_chain`) permite experimentar con prompt, `k` y umbral sin volver a pagar embeddings ni chocar con el rate limit.
- MarkItDown extrae los PDFs con muchas palabras pegadas; eso afecta los embeddings y obliga a escoger con cuidado las palabras clave de una evaluación automática (C4).
- En HNSW con datos aleatorios, el `efSearch` necesario para recall ≥ 0,95 crece con `n` (256 → 1024 → > 2048) y se come la ventaja frente a la búsqueda exacta; subir `M` desplaza la curva recall–efSearch hacia arriba (C1).
- `ef` se puede ajustar por consulta y `ef_construction` no; por eso la primera palanca ante poco recall es `ef` (T2).
