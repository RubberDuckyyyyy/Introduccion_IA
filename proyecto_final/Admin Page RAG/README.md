# Admin Page RAG

Sistema RAG (generación aumentada por recuperación) que responde preguntas
sobre cómo configurar el ambiente de desarrollo de *Banner Admin Pages*,
usando solo la evidencia de los documentos indexados.

**Stack:** Streamlit (UI) → FastAPI (API) → ChromaDB (índice persistente) +
Google AI (embeddings `gemini-embedding-001` y generación con Gemini).

- Corpus de ejemplo: [CORPUS.md](CORPUS.md)
- Reporte del proyecto: [REPORTE.md](REPORTE.md)

## Arquitectura

```text
Usuario
  └── Streamlit  app.py           (puerto 8501)
        └── HTTP JSON
              └── FastAPI  api.py (puerto 8000)
                    ├── Google AI → embeddings de chunks y preguntas (mismo modelo)
                    ├── ChromaDB  → persistencia en chroma/ y búsqueda top-k
                    └── Google AI → respuesta en español con citas [n] (Gemini)
```

Streamlit no accede a Chroma ni a Google AI: todo pasa por la API.

## Estructura

El proyecto vive en `RAG-proyecto/Admin Page RAG/`. Los nombres difieren de la
estructura sugerida en la consigna; esta es la equivalencia:

| Archivo | Equivale a | Qué hace |
|---|---|---|
| [api.py](api.py) | `app/main.py` | FastAPI: `/health`, `/ingest`, `/query` |
| [rag/chunk.py](rag/chunk.py) | `app/chunk.py` | Partición en chunks con overlap (`chunk_text`) |
| [rag/google_ai.py](rag/google_ai.py) | `app/embed.py` + `app/generate.py` | Cliente de Google AI: embeddings y generación |
| [rag/chroma_store.py](rag/chroma_store.py) | `app/store.py` | ChromaDB: alta y consulta top-k |
| [rag/data.py](rag/data.py) | — | Lectura de `.pdf`, `.md` y `.txt` |
| [app.py](app.py) | `ui/streamlit_app.py` | UI: carga, pregunta, citas y scores |
| [docs/](docs/) | `data/` | Corpus de ejemplo (6 PDF) |
| `chroma/` | `chroma/` | Índice persistente (en `.gitignore`) |
| [config.yaml](config.yaml) | — | Chunking, `top_k`, umbral de abstención, carpetas |
| [.env.example](.env.example) | `.env.example` | Plantilla de variables de entorno |

## 1. Instalación

Requiere Python 3.14 (probado con 3.14.7 en Windows 11).

```powershell
cd "RAG-proyecto\Admin Page RAG"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Clave de Google AI

1. Entra a [Google AI Studio](https://aistudio.google.com/apikey) con tu
   cuenta de Google y pulsa **Create API key**.
2. Copia la plantilla y pega la clave en `GOOGLE_API_KEY=`:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

`.env` está en `.gitignore`: nunca subas la clave al repositorio. También
puedes definir `GOOGLE_API_KEY` como variable de entorno, que tiene prioridad
sobre el `.env`.

## 3. Levantar la API y la UI

En dos terminales, ambas con el venv activado y dentro de la carpeta del
proyecto:

```powershell
# Terminal 1: API (documentación interactiva en http://localhost:8000/docs)
uvicorn api:app --reload --port 8000

# Terminal 2: UI (se abre en http://localhost:8501)
streamlit run app.py
```

## 4. Probar una pregunta

1. **Indexa el corpus.** En la barra lateral de Streamlit sube los PDF de
   `docs/` y pulsa **Ingestar**. Resultado esperado: *6 documento(s), 168
   chunks indexados*.
2. **Pregunta algo del dominio**, por ejemplo *¿Cómo configuro un ambiente de
   admin page?*, y pulsa **Preguntar**. La respuesta sale en español con citas
   `[n]`; en **Citas** se ven el origen, el chunk y el score de cada fragmento.
3. **Pregunta algo fuera del dominio**, por ejemplo *¿Cuál es la receta de la
   paella valenciana?*: el sistema se abstiene con *"No tengo evidencia
   suficiente…"*.

Lo mismo contra la API, sin la UI (PowerShell):

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/ingest -Form @{ paths = "docs" }

$body = @{ question = "¿Cómo configuro un ambiente de admin page?"; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post http://localhost:8000/query -ContentType "application/json; charset=utf-8" -Body $body
```

## API

| Endpoint | Entrada | Salida |
|---|---|---|
| `GET /health` | — | `status`, `embed_model`, `google_api_key` (si hay clave), `chroma` (`ok` y número de chunks) |
| `POST /ingest` | multipart: `files` (se guardan en `docs/`) y/o `paths` (archivos o carpetas del servidor) | `documents`, `chunks`, `skipped` (archivos sin texto) |
| `POST /query` | JSON: `question` (obligatoria, no vacía), `top_k` opcional (1–20, por defecto 3) | `answer`, `citations` (`id`, `index`, `source`, `title`, `text`, `score`), `abstained` |

- Reingestar un archivo reemplaza sus chunks anteriores (no duplica).
- Cada chunk guarda en Chroma los metadatos `source`, `title` e `index`
  (posición del chunk en su documento).
- Códigos de error: `400` entrada inválida, `422` pregunta vacía, `409` el
  índice se creó con otro modelo de embeddings, `503` Google AI no disponible
  (falta la clave, cuota agotada o saturación, tras reintentar). Una pregunta
  fuera de dominio **nunca** da error: responde `200` con `abstained: true`.

## Regla de abstención

`/query` responde `abstained: true` y `answer` = *"No tengo evidencia
suficiente en los documentos para responder esa pregunta."* cuando:

1. **No hay documentos indexados** en Chroma.
2. **El chunk más parecido tiene similitud coseno menor que `api_min_score`**
   (0.5 en [config.yaml](config.yaml)). En este caso no se llama a Gemini.
3. **Gemini indica que el contexto no basta.** El *system prompt* le exige
   responder en español, usar solo los fragmentos numerados `[1]`, `[2]`… y,
   si no alcanzan, responder exactamente `NO_EVIDENCE`.

Cuando se abstiene no devuelve citas ni completa con conocimiento propio del
modelo.

## Configuración

| Dónde | Clave | Por defecto | Para qué |
|---|---|---|---|
| `config.yaml` | `chunk_words` / `overlap` | 80 / 15 | Tamaño y solapamiento de los chunks (en palabras) |
| `config.yaml` | `top_k` | 3 | Chunks recuperados por pregunta |
| `config.yaml` | `api_min_score` | 0.5 | Umbral de similitud para abstenerse |
| `config.yaml` | `chroma_dir` / `docs_dir` | `chroma` / `docs` | Carpetas del índice y del corpus |
| `.env` | `GOOGLE_API_KEY` | — | Clave de Google AI (obligatoria) |
| `.env` | `GOOGLE_EMBED_MODEL` | `gemini-embedding-001` | Modelo de embeddings. Si lo cambias, borra `chroma/` y reingesta |
| `.env` | `GOOGLE_GEN_MODEL` | `gemini-3.8-flash` | Modelo que redacta las respuestas |
| `.env` | `API_URL` | `http://localhost:8000` | Dirección de la API que usa Streamlit |

## Solución de problemas

| Síntoma | Causa y solución |
|---|---|
| Streamlit: *No se pudo conectar con la API* | La API no está corriendo: levántala en la terminal 1. |
| Streamlit: *La API no tiene GOOGLE_API_KEY* | Falta la clave en `.env`: agrégala y reinicia uvicorn. |
| `429 RESOURCE_EXHAUSTED` | Se agotó la cuota gratuita de Google AI: espera a que se reinicie o activa la facturación en AI Studio. |
| `503 high demand` | Google AI está saturado: espera un minuto y repite. |
| `409 el índice se creó con…` | Cambiaste `GOOGLE_EMBED_MODEL`: borra `chroma/` y vuelve a ingestar. |

## Anexo: pipeline didáctico (no es el backend)

`ask.py`, `01_embed.py` … `04_answer.py` y los módulos `rag/embed.py`,
`rag/store.py`, `rag/retrieve.py`, `rag/generate.py`, `rag/vectors.py`,
`rag/tokenize.py`, `rag/pipeline.py`, `rag/format.py` y `rag/cli.py` son la
versión original del ejercicio: embeddings bag-of-words en memoria y respuesta
extractiva, sin API externa. Se conservan solo como referencia para comparar.
**No los usan ni la API ni la UI**, y el índice del sistema se construye
exclusivamente con embeddings de Google AI en ChromaDB.

```powershell
python ask.py    # modo interactivo de la versión didáctica
```
