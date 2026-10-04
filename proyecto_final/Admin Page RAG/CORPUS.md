# Corpus de ejemplo

El corpus son los PDF de la carpeta [`docs/`](docs/) y ya viene incluido en el
repositorio: no hay que descargar nada.

| Archivo | Contenido | Páginas |
|---|---|---:|
| `settup eclipse 1.pdf` | Configurar Eclipse para Java Batch (workspace, repos `banner_product` / `banner_general`). | 4 |
| `settup eclipse 2.pdf` | Eclipse para Banner Admin Pages: prerrequisitos (Maven, Tomcat 7.0.47, JDK 1.8), clonado e importación del proyecto. | 13 |
| `settup eclipse 3.pdf` | Eclipse Setup Quick Guide (enero 2023): `settings.xml` de Maven, versiones de Eclipse y JDK, plugins de Morphis Foundation, *sizing* y caché. | 27 |
| `settup eclipse 4.pdf` | Instalar Banner Admin en IntelliJ IDEA 2024.1: carpeta de repositorios, clonado de la rama `dev` / `dev_saas`, apertura y build. | 8 |
| `settup eclipse 5.pdf` | Banner Admin para PostgreSQL y Oracle: repositorios (`base`, `build`, `common`, `general`…), rama `dev` (Oracle) y `dev_saas` (PG), JREs y pasos para ambas bases. | 20 |
| `settup eclipse 6.pdf` | Configurar Banner Admin Pages 9.x ("transformed pages") en la máquina local: software requerido (JDK 1.7/1.8, Tomcat 7.0.47, Eclipse Oxygen/Neon, Maven 3.2.3), `settings.xml`, clonado e importación. | 18 |

**Total:** 6 documentos, 90 páginas, 10 805 palabras y 168 chunks (80 palabras,
15 de solapamiento). Los seis son PDF con capa de texto (no escaneados), en
inglés.

## Cómo indexarlo

Con la API corriendo (ver el [README](README.md)):

- **Desde Streamlit:** sube los seis PDF en la barra lateral y pulsa **Ingestar**.
- **Desde la API:** `POST /ingest` con el campo `paths = docs`, por ejemplo:

  ```powershell
  curl.exe -X POST http://localhost:8000/ingest -F "paths=docs"
  ```

Resultado esperado: `{"documents": 6, "chunks": 168, "skipped": []}`.

## Usar tu propio corpus

Copia tus `.pdf`, `.md` o `.txt` en `docs/` (o súbelos desde Streamlit) y
vuelve a ingestar. Volver a ingestar un archivo reemplaza sus chunks
anteriores. Todo archivo de texto dentro de `docs/` se indexa, así que no
pongas ahí notas que no sean parte del corpus.
