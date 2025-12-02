# Terminal Bitcoinmoon: La App pa' Cachar el Bitcoin

**Autor:** [@gmolate](https://github.com/gmolate)

## ¿De Qué Se Trata la Cuestión?

Esta es una maquinita pa' analizar el Bitcoin, con un estilo medio "Bloomberg" pero hecho en casa, ¿cachai? La idea es que tengai gráficos que se mueven, indicadores y hasta análisis con IA pa' que no te pille la máquina y sepas pa' dónde va la micro.

Este chiche lo estamos armando con un equipo de tres cracks:
1.  **El Informático:** Un capo de Python que se preocupa de que el código no se caiga y todo funcione tiqui-taca.
2.  **El Trader:** Un zorro que sabe de lucas y mercados, y dice qué análisis son los que realmente importan.
3.  **El Diseñador:** El que le pone el toque visual pa' que la cuestión se vea bonita y no sea un cacho de usar.

## ¿Qué Gracias Trae?

### Fuentes de Datos (Pa' todos los gustos)

La aplicación puede sacar datos de varias partes. Unas son gratis y otras más pro:

-   **Gratis y sin dramas:**
    -   **Yahoo Finance & CoinGecko:** Vienen listas pa' usar. Conectan y muestran los datos al toque, sin registrarse en ninguna parte.
    -   **Archivo CSV:** Podís cargar tu propio archivo con datos históricos si lo tenís a mano.
-   **Opcionales (Requieren configuración):**
    -   **Kaggle:** Si querís bajar el historial completo, podís usar la API de Kaggle. Necesitai tu archivo `kaggle.json`.
    -   **Polygon:** Pa' datos más específicos, también se puede conectar a Polygon, pero te pide una API key.

### Chiches del Analizador

-   **Gráficos que se Mueven:** Los gráficos no son una foto, podís hacerles zoom y moverte en el tiempo como querai.
-   **Indicadores Técnicos:** Trae el CMF y el RSI listos pa' usar, que son los que usan los que saben.
-   **Análisis de Ciclos (Halvings):** Te muestra los ciclos de Halving del Bitcoin pa' que veai el patrón histórico.
-   **El Toque Astral:** Te pone las fases de la luna en el gráfico, pa' ver si influye en el precio o es puro cuento.
-   **Análisis con IA (¡Sin Claves API!):** Esta es la papa misma. La aplicación genera un informe completo y un **prompt** basado en todos los datos del gráfico. El sistema es simple:
    1.  Vas a la pestaña **"Análisis (Asistido por IA)"**.
    2.  Elegís el modelo de IA que más te guste de la lista (Grok, Gemini, ChatGPT, Kimi, Deepseek, Qwen).
    3.  Copiái el prompt que te generó el programa con el botón **"Copiar Prompt"**.
    4.  Presionas **"Acceder al Servicio"** y se abrirá el sitio web del LLM en tu navegador.
    5.  Pegas el prompt y la IA te hace un análisis profesional al instante. ¡Fácil, rápido y sin meter ninguna clave!
-   **No se Queda Pegado:** La carga de datos es piola y no te congela la aplicación. Puedes seguir haciendo click mientras carga.

## Cómo Echar a Andar Este Cacharrito

1.  **Requisitos:** Tenís que tener Python 3.9 o más nuevo instalado. Si no, estás sonado.
2.  **Clona el Repositorio:** Lo típico, poh: `git clone <repository-url>`
3.  **Instala las Dependencias:** Ándate a la carpeta del proyecto en la terminal y tira este comando. Con esto se instalan todas las librerías pa' que la magia funcione.
    ```bash
    pip install -r requirements.txt
    ```
4.  **(Opcional) Configurar APIs de Datos:** Si querís usar Kaggle o Polygon, tenís que tener tus credenciales (el `kaggle.json` o la API key de Polygon) listas pa' que el programa las pueda usar. Si no, usa las opciones gratis y listo.

## Pa' Prender la Máquina

Una vez que está todo instalado, dale nomás con esto en la terminal:
```bash
python bitcoinmoon.py
```

Con eso, la app queda lista y funcionando con todas sus gracias.

## ¿Querís Probar si Funciona? (Testing)

El proyecto usa `pytest` pa' cachar si la lógica de los cálculos está buena.

-   Las pruebas usan datos de un archivo (`test_data.csv`), así que no necesitai internet.
-   Pa' correr las pruebas, abre la terminal en la carpeta del proyecto y escribe:
    ```bash
    pytest
    ```
CI:
- Se incluye un workflow de GitHub Actions en .github/workflows/python-app.yml que ejecuta flake8.
