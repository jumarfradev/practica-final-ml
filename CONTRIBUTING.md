# Guía de contribución

Esta guía describe cómo trabajar en este repositorio. Léela antes de hacer tu primer commit.

---

## Primera vez: configuración inicial

Cuando clones el repo por primera vez, sigue estos pasos para tener tu entorno listo:

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/practica-final-ml.git
cd practica-final-ml

# 2. Crear el entorno virtual con uv (Python 3.11)
uv venv --python 3.11 env-pontia-ml

# 3. Activar el entorno
source env-pontia-ml/Scripts/activate    # Windows (Git Bash)
# source env-pontia-ml/bin/activate      # Mac / Linux

# 4. Instalar dependencias
uv pip install -r requirements.txt

# 5. Instalar pre-commit hooks
pre-commit install

# 6. Verificar instalación
python -c "import pandas, sklearn, tensorflow, xgboost, lightgbm, keras, mlflow; print('OK')"
```

> Pre-commit **solo se enlaza una vez por máquina**. Si no lo instalas, tus commits no
> pasarán por los hooks locales (aunque CI seguirá ejecutándose en GitHub).

---

## Flujo de ramas

Trabajamos con **GitHub Flow simplificado**: una rama por tarea, mergeada a `main` vía Pull Request.

### Rama principal

- **`main`**: rama estable. Siempre debe estar funcional y pasar CI.
  No se commitea directamente, solo se mergea desde otras ramas vía PR.

### Ramas de trabajo

Una rama por tarea. Convención de nombres:

```
<tipo>/<descripción-corta-en-kebab-case>
```

Tipos válidos:

| Tipo | Cuándo usarlo | Ejemplo |
|---|---|---|
| `feature/` | Nueva funcionalidad | `feature/data-loader` |
| `fix/` | Corregir un bug | `fix/null-handling-country` |
| `docs/` | Solo documentación | `docs/update-readme` |
| `refactor/` | Reorganizar sin cambiar lógica | `refactor/split-trainer-module` |
| `chore/` | Tareas de mantenimiento | `chore/update-dependencies` |
| `experiment/` | Pruebas exploratorias en notebooks | `experiment/xgboost-tuning` |

### Cómo crear una rama y trabajar en ella

```bash
# 1. Asegúrate de estar en main al día
git checkout main
git pull

# 2. Crea la rama nueva
git checkout -b feature/nombre-de-la-tarea

# 3. Trabaja: edita, añade, commitea
git add archivos_modificados
git commit -m "feat: descripción del cambio"

# 4. Sube la rama a GitHub
git push -u origin feature/nombre-de-la-tarea

# 5. Abre un Pull Request en GitHub apuntando a main
# 6. Espera revisión + CI verde
# 7. Mergea desde la UI de GitHub (Squash and merge)

# 8. Después del merge, limpia tu rama local
git checkout main
git pull
git branch -d feature/nombre-de-la-tarea
```

---

## Convención de commits

Usamos **Conventional Commits** con mensajes en español.

### Formato

```
<tipo>: <descripción corta en imperativo>

[cuerpo opcional con más detalle]
```

### Tipos

| Prefijo | Uso |
|---|---|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de bug |
| `docs` | Cambios solo en documentación |
| `style` | Formato (espacios, comas...) sin cambios funcionales |
| `refactor` | Reorganización sin cambiar comportamiento |
| `test` | Añadir o modificar tests |
| `chore` | Tareas auxiliares (dependencias, configs) |
| `ci` | Cambios en CI/CD o workflows |
| `perf` | Mejora de rendimiento |

### Ejemplos buenos

```
feat: añadir data loader con imputación de nulos en country
fix: corregir error de KeyError en evaluator con etiquetas binarias
docs: actualizar README con instrucciones de uso del trainer
chore: añadir lightgbm a requirements.txt
ci: añadir step de Black al workflow de calidad
```

### Ejemplos a evitar

```
❌ "cambios varios"
❌ "Fix"
❌ "Actualización"
❌ "WIP"
❌ "asdf"
```

### Regla de oro

Escribe el commit como si completara la frase: **"Si aplico este commit, va a... [tu mensaje]"**.

> ✅ "feat: añadir data loader" → "Si aplico, va a añadir data loader". Tiene sentido.
> ❌ "feat: data loader añadido" → "Si aplico, va a data loader añadido". No tiene sentido.

---

## Pull Requests

### Antes de abrir el PR

1. Asegúrate de que `pre-commit run --all-files` pasa en local.
2. Asegúrate de que tu rama tiene los últimos cambios de `main`:
```bash
   git checkout main
   git pull
   git checkout tu-rama
   git merge main    # o: git rebase main
```
3. Empuja tu rama: `git push`.

### Al abrir el PR

- Asígnate como autor.
- Si trabajas en pareja, asigna a la otra persona como revisor.
- Rellena **todas** las secciones de la plantilla. No dejes secciones vacías.
- Si el PR está en progreso, márcalo como **Draft**.

### Estrategia de merge

Usamos **"Squash and merge"** por defecto:

- Combina todos los commits de la rama en uno solo al mergear a `main`.
- Mantiene el historial de `main` limpio (1 commit = 1 feature).
- El mensaje del commit squashed sigue la convención (`feat: ...`).

---

## Antes de cada commit (checklist)

- [ ] El entorno virtual está activo (`(env-pontia-ml)` en el prompt).
- [ ] `pre-commit` está instalado (`pre-commit install` se ejecutó alguna vez).
- [ ] El código corre en local (no commitees código a medias sin probar).
- [ ] No has dejado `print()` de debug, claves o datos sensibles.

---

## Estructura del proyecto

```
practica-final-ml/
├── .github/                 # Workflows de CI y plantilla de PR
├── data/
│   ├── raw/                 # Dataset original (NO MODIFICAR)
│   └── processed/           # Datos procesados (regenerable)
├── docs/                    # Documentación adicional (informe final)
├── models/                  # Modelos serializados (ignorados en git)
├── notebooks/
│   ├── exploracion/         # Notebooks de pruebas / EDA inicial
│   └── finales/             # Notebooks limpios para presentación
├── outputs/                 # Gráficas y resultados
├── src/                     # Código fuente del pipeline
└── trainer.py               # Punto de entrada CLI
```

### Reglas:

- **No subir** modelos `.pkl` (regenerables con `trainer.py`).
- **No subir** datos derivados de `data/processed/`.
- **Sí subir** las gráficas finales `.png` de `outputs/`.
- **No tocar** `data/raw/` (es el origen inmutable).

---

## Resolución de problemas comunes

### `pre-commit` modifica archivos al hacer commit

Es normal. Pre-commit corrige formato automáticamente. Vuelve a añadir los cambios y commitea de nuevo:

```bash
git add .
git commit -m "tu mensaje"
```

### CI falla por imports

Probablemente añadiste una librería localmente con `uv pip install` pero no la añadiste a `requirements.txt`. Revisa.

### Conflictos al mergear con `main`

```bash
git checkout main
git pull
git checkout tu-rama
git merge main
# resuelve los conflictos en VS Code
git add .
git commit -m "chore: resolver conflictos con main"
git push
```

---

## Dudas

Si algo no está claro o falta documentar, abre un issue o pregunta al equipo.
