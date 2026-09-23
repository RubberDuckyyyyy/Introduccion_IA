# Admin Page RAG

Asistente de preguntas y respuestas sobre "cómo configurar el ambiente de
admin page", construido sobre el pipeline de [RAG/project](../RAG/project):
embeddings bag-of-words + k-NN exacto, sin API externa ni LLM. La respuesta es
extractiva (cita y pega las oraciones más relevantes del contexto recuperado);
si no hay evidencia suficiente, el sistema se abstiene en vez de inventar.

## Setup

```powershell
cd "Admin Page RAG"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 1. Agrega tu documentación

Pon tus archivos `.md`, `.txt` o `.pdf` (variables de entorno, permisos, pasos
de despliegue, etc.) dentro de `docs/`. Cada archivo se indexa automáticamente
como un documento; no hace falta tocar ningún YAML. Borra `docs/ejemplo.md`
cuando agregues los tuyos.

Los PDF se leen con `pypdf`, que extrae el texto que el PDF ya trae embebido.
Si es un PDF escaneado (solo imágenes, sin capa de texto), la extracción da
vacío y el archivo se ignora en silencio — en ese caso pásalo primero por OCR
o convierte el texto a `.md`/`.txt` a mano.

## 2. Ajusta la configuración (opcional)

`config.yaml` controla el tamaño de chunk, el solapamiento, cuántos chunks se
recuperan por pregunta (`top_k`) y el umbral mínimo de similitud
(`min_score`) antes de abstenerse. Los valores por defecto (`chunk_words: 80`,
`overlap: 15`) están pensados para documentación técnica en prosa; ajústalos
si tus documentos son muy cortos o muy largos.

## 3. Pregunta

Modo interactivo (recomendado, indexa una sola vez):

```bash
python ask.py
```

O el pipeline paso a paso, igual que en RAG/project:

```bash
python 01_embed.py              # ver cómo se vectoriza el texto
python 02_index.py              # chunkear + indexar docs/
python 03_retrieve.py --query "¿cómo configuro las variables de entorno?"
python 04_answer.py --query "¿cómo configuro las variables de entorno?"
```

## Cómo funciona

1. **Index** (`02_index.py`) — lee cada archivo de `docs/`, lo divide en
   chunks solapados y guarda un vector (conteo de palabras) por chunk.
2. **Retrieve** (`03_retrieve.py`) — vectoriza la pregunta y compara contra
   todos los chunks con similitud coseno.
3. **Answer** (`04_answer.py` / `ask.py`) — arma el prompt con los chunks
   top-k y devuelve las oraciones que comparten más palabras con la
   pregunta, citando `[n]` la fuente. Por debajo de `min_score`, responde
   "no tengo evidencia suficiente" en vez de inventar.

## Siguiente paso natural

Este método es extractivo a propósito (sin dependencias, sin costo). Si más
adelante quieres respuestas redactadas en lenguaje natural en vez de
oraciones pegadas, el punto de extensión es `rag/generate.py`: reemplaza
`answer()` por una llamada a un LLM (p. ej. la API de Claude) usando el mismo
`prompt` que ya se construye con `build_prompt()` — la recuperación
(`retrieve.py`, `store.py`) no cambia.
