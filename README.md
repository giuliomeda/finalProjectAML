# Stroke Prediction Project

Applicazione web per la predizione del rischio di ictus 

## Requisiti

- Python 3.12
- [uv](https://github.com/astral-sh/uv) (per la gestione delle dipendenze, alternativa a pip)

## Installazione

1. **Clona la repository**  
   ```sh
   git clone <URL_DEL_REPO>
   cd finalProjectAML
   ```
2. **Installa le dipendenze**  
   Se stai usando `uv`:
   ```sh
   uv sync
   ```
   Installa tutte le dipendenze presenti in pyproject.toml
3. **Esegui l'applicazione**  
   ```sh
   uv run streamlit run app.py
   ```
    Oppure 
    ```sh
    ./run.sh
    ```
Visita `http://localhost:8501` nel tuo browser per vedere l'app in azione.

## Struttura del Progetto

- `app.py`: Il file principale dell'applicazione Streamlit.
- `model/`: Directory contenente il modello XGBoost e i file di pipeline.
