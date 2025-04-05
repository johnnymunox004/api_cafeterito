


# pip install fastapi uvicorn python-multipart pydantic matplotlib tensorflow numpy opencv-python openai

# python app.py

# pip install pyngrok

# ngrok http 8000



from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import sys
import time
import uuid
import base64
from io import BytesIO
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Dict, Any, Optional

# verficar si la librerias estan diponibles   de esta line a la 57 es para revisaar la libreriaas
MATPLOTLIB_AVAILABLE = False
CV2_AVAILABLE = False
TENSORFLOW_AVAILABLE = False

# Intentar importar matplotlib por separado
try:
    import matplotlib
    matplotlib.use('Agg')  # Configuración para usar matplotlib sin interfaz gráfica
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    print("Matplotlib importado correctamente")
except ImportError as e:
    print(f"Advertencia: No se pudo importar matplotlib: {e}")

# Intentar importar TensorFlow y NumPy
try:
    import tensorflow as tf
    import numpy as np
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow importado correctamente")
except ImportError as e:
    print(f"Error importando TensorFlow: {e}")

# Intentar importar OpenCV (primero intentar headless si está disponible)
try:
    try:
        import cv2
        CV2_AVAILABLE = True
        print("OpenCV importado correctamente")
    except ImportError:
        # Si hay un error, intentar importar módulos esenciales sin OpenCV
        print("OpenCV no disponible, se usará simulación para funciones de procesamiento de imágenes")
except Exception as e:
    print(f"Error general al configurar OpenCV: {e}")
    print("Funcionando en modo simulación")

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="la poderosa e increible api de de cafeterito ia ",
    description="API para análisis del nivel de tostión de café y consultas a expertos",
    version="1.0.0"
)

# Configuración de CORS para poder usar la api desde cafeterito frontend 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# cargamos el modelo de la carpeta final_coffee_tl_model.keras paraa analizar la imagen 
model = None
model_error = None
if TENSORFLOW_AVAILABLE:
    try:
        model_path = 'final_coffee_tl_model.keras'
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            print(f"Modelo cargado exitosamente desde {model_path}")
        else:
            model_error = f"Archivo del modelo no encontrado: {model_path}"
            print(f"Error: {model_error}")
    except Exception as e:
        model_error = str(e)
        print(f"Error al cargar el modelo: {e}")

# Configuración de OpenAI/OpenRouter
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#  cambiar credenciales por si se acaba la capa gratuita 
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Definición de modelos de datos
class CoffeeQuery(BaseModel):
    pregunta: str

# Función para generar una imagen simple simulada
def generate_simple_chart_base64():
    """Genera una imagen simple cuando matplotlib no está disponible"""
    # Un patrón básico de una imagen en base64 (una barra gris simple)
    return "iVBORw0KGgoAAAANSUhEUgAAAfQAAAGQCAIAAADq4XyxAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH5gQBAw8fT5zxCAAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAACNUlEQVR42u3VQREAAAjDMMC/5+ECjiYK+NnNzR4AwF99AQAA5AYAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3AAC5AQDIDQBAbgAAcgMAkBsAgNwAAOQGACA3ALK1Ay1QAZixNZQpAAAAAElFTkSuQmCC"

# aqui se genera el grafico de barrasa y devuelbe un base64 
def generate_bar_chart(probabilities):
    """Genera un gráfico de barras y lo devuelve como imagen base64"""
    # Si matplotlib no está disponible, devolver una imagen simple
    if not MATPLOTLIB_AVAILABLE:
        return generate_simple_chart_base64()
    
    plt.figure(figsize=(10, 6))
    
    # Datos para el grafico
    roast_levels = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Crear el grafico de barras
    bars = plt.bar(roast_levels, values, color=['#8B4513', '#D2B48C', '#A0522D'])
    
    # se añade las etiquetas del graafico 
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}', ha='center', va='bottom')
    
    # Estilo y etiquetas
    plt.title('Análisis de Nivel de Tostión de Café', fontsize=16)
    plt.xlabel('Nivel de Tostión', fontsize=14)
    plt.ylabel('Probabilidad', fontsize=14)
    plt.ylim(0, 1.1)  # Limitar eje Y entre 0 y 1.1 para dejar espacio a las etiquetas
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Guardar en un buffer en memoria en lugar de archivo
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    
    # Convertir a base64 para incluir en JSON
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_str

# Función para detectar granos de café en una imagen - versión simulada
def detect_coffee_beans(image_path):
    """
    Versión simulada para cuando OpenCV no está disponible.
    Devuelve una imagen base64 simulada y datos de simulación.
    """
    # Si OpenCV está disponible, usarlo
    if CV2_AVAILABLE:
        try:
            # Leer la imagen original
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"No se pudo cargar la imagen desde {image_path}")
            
            # Crear una copia para marcar los granos detectados
            marked_image = image.copy()
            
            # Convertir a escala de grises para detección (pero conservar original a color)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Aplicar desenfoque gaussiano para reducir ruido
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Aplicar umbralización adaptativa para segmentar los granos
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Operaciones morfológicas para mejorar la segmentación
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Dilatación para conectar áreas cercanas
            dilated = cv2.dilate(opening, kernel, iterations=1)
            
            # Encontrar contornos de los granos
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por tamaño para eliminar ruido
            min_area = 100  # Ajustar según el tamaño esperado de los granos
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            # Lista para almacenar imágenes de granos individuales
            individual_beans = []
            bean_positions = []
            
            # Procesar cada contorno válido
            for i, contour in enumerate(valid_contours):
                # Obtener rectángulo delimitador con rotación mínima
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Dibujar contorno en la imagen marcada
                cv2.drawContours(marked_image, [box], 0, (0, 255, 0), 2)
                
                # Añadir número de identificación
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(marked_image, f"#{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Extraer el grano individual como una región de interés (ROI)
                # Usar el rectángulo delimitador normal para facilitar el recorte
                roi = image[y:y+h, x:x+w]
                
                # Asegurarse de que la ROI no está vacía
                if roi.size > 0:
                    # Redimensionar la imagen del grano al tamaño que espera el modelo
                    resized_bean = cv2.resize(roi, (224, 224))
                    individual_beans.append(resized_bean)
                    bean_positions.append((x, y, w, h))
            
            # Convertir la imagen marcada a formato base64
            _, img_encoded = cv2.imencode('.png', marked_image)
            marked_image_base64 = base64.b64encode(img_encoded).decode('utf-8')
            
            return marked_image_base64, individual_beans, bean_positions
        
        except Exception as e:
            print(f"Error en la detección de granos con OpenCV: {str(e)}")
            # Si hay error, caer en el modo simulación
    
    # Si OpenCV no está disponible o hubo error, simular
    print("Usando simulación para detección de granos")
    
    # Imagen base64 simulada con cuadrados marcados como si fueran granos
    marked_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgAQMAAAAPH06nAAAABlBMVEX///8AAABVwtN+AAAACXBIWXMAAA7EAAAOxAGVKw4bAAADHUlEQVR4nO2aTY7bMAxGpUFWWWaVZY7gI/gIPYKP0CP0CB7NbmYzm8HAsgiJlPgnBfKWrzyAIP74JFtq5ji4XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6Xy+VyuVwu18eUNX/iZ/o7/JL5mmbz4JeJkwvx5eIlJPnuHNOGfMgnl4RkDLFpTfZQpjSeHIiBTwT88mNe/wIFOGKk6JrGEzuYqM9S6+8gRtNZYlmTIVL9kxN7Elv9/WAeIsLPqsGfXHsE4GIWsZwHMY9vTk7T2Z/CnUQT8JDMyR9kdMJ4lU6f3Px4wSHGFOB2+qlEkhUj2mY7mtFGFzS5kdiwPQQTtEE7xgqjxEiGYIpvFEtFdncfLrFu2qYpkLW6J6wPJ3L0YuwTwFEg5qPzEH02YgSh0UXA2RfdUDHrfpAYNT+SZrtCxDwWjGEwM1oGWnTTlG2E9Yr9uERXCTDsVhNOATABdhtDLYuYAQjZF4B5vn1i9S9Jlk93zeDIQkQw8VgNSsUMQMy+NaIJxhU1gUWs66QGHAZzRiC+qc+miBhrMRo33yCm0ZoICeP6Odn34yIg5LpnvcjkpMpDqVF1hLjXrYNJQCIU8ZLLRkzjl3JOQSrWuhZP3Qxm3cdSI1ZsFtCVIqa19zDEuOCO2ogBaxzlZnhA0JbUE/MlD6t/7jxwFaLkYZ7a1DwoRyVmuQ+gvuYhiS+pPHC7t7giPTEvvX5TEevfpdSbfJDJrPVJ14qcwQyPF3nQz9G45pMxIxFXIeblAk7RLDBhKRCl/gEY2UfiAUEOLPWfrXqcPQUwQM76nJKXhxQqMu75eeSKuCfTBGtFkf3zFESqv3xrDmKeCTiWF2s+pCNP1RP32pL7sEQcB/iD9W0eEAwrMsXlglTEZI+qmWYxQ4p3ZOvPgQbXO5IDJF0TnVxRTpBmseB0r+/qIK7TjXJwsZ8OMtwHk6Mos3jH/kHiqe//8XK5XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6X629R/wEmT5kw6y0H4AAAAABJRU5ErkJggg=="
    
    # Simular 5 granos detectados con sus posiciones
    individual_beans = []  # En modo simulación, no hay imágenes reales
    bean_positions = [
        (50, 50, 30, 30),
        (100, 50, 30, 30),
        (150, 50, 30, 30),
        (50, 100, 30, 30),
        (100, 100, 30, 30)
    ]
    
    return marked_image_base64, individual_beans, bean_positions

# Función para procesar una imagen individual de grano de café
def process_bean_image(img=None, simulated_level=None):
    """
    Procesa una imagen de grano de café o simula un resultado.
    En modo simulación, el parámetro img se ignora y se usa simulated_level.
    """
    roast_levels = ["Dark", "Green", "Light", "Medium"]
    if not TENSORFLOW_AVAILABLE or img is None:
        # Si TensorFlow no está disponible o no hay imagen, simular
        return simulate_prediction(simulated_level)
    
    # Si TensorFlow está disponible y tenemos una imagen
    try:
        # Asegurarse de que la imagen está en RGB (el modelo espera RGB, no BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalizar
        img = img / 255.0
        
        # Añadir dimensión de batch
        img = np.expand_dims(img, axis=0)
        
        # Realizar predicción
        predictions = model.predict(img)
        
        # Niveles de tostión de café
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Crear diccionario de probabilidades
        probabilities = {
            roast_levels[i]: float(predictions[0][i]) 
            for i in range(len(roast_levels))
        }
        
        return roast_levels[predicted_class], confidence, probabilities
    except Exception as e:
        print(f"Error procesando imagen de grano: {str(e)}")
        # Si hay error, volver a simulación
        return simulate_prediction(simulated_level)

# Función para procesar la imagen completa identificando granos individuales
# Reemplaza esta función en tu código
def process_image_with_beans(image_path):
    """
    Analiza una imagen de café, detecta granos individuales y los evalúa.
    Si no es posible el análisis real, usa simulación.
    """
    roast_levels = ["Dark", "Green", "Light", "Medium"]
    try:
        # Primero verificar si tenemos TensorFlow y el modelo cargado
        if not TENSORFLOW_AVAILABLE or model is None:
            print("TensorFlow o modelo no disponible, usando simulación completa")
            return simulate_prediction_with_beans(image_path)
        
        # Si tenemos TensorFlow pero no OpenCV, analizamos la imagen completa con el modelo
        if not CV2_AVAILABLE:
            print("OpenCV no disponible, analizando imagen completa con el modelo")
            # Cargar imagen usando TensorFlow directamente
            img_raw = tf.io.read_file(image_path)
            img = tf.image.decode_image(img_raw, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = img / 255.0
            img = tf.expand_dims(img, 0)
            
            # Usar el modelo para predecir
            predictions = model.predict(img)
            
            # Niveles de tostión de café
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Crear diccionario de probabilidades
            probabilities = {
                roast_levels[i]: float(predictions[0][i]) 
                for i in range(len(roast_levels))
            }
            
            # Generar gráfico de barras
            chart_base64 = generate_bar_chart(probabilities)
            
            # Usar imagen simulada para detección de granos
            marked_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgAQMAAAAPH06nAAAABlBMVEX///8AAABVwtN+AAAACXBIWXMAAA7EAAAOxAGVKw4bAAADHUlEQVR4nO2aTY7bMAxGpUFWWWaVZY7gI/gIPYKP0CP0CB7NbmYzm8HAsgiJlPgnBfKWrzyAIP74JFtq5ji4XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6Xy+VyuVwu18eUNX/iZ/o7/JL5mmbz4JeJkwvx5eIlJPnuHNOGfMgnl4RkDLFpTfZQpjSeHIiBTwT88mNe/wIFOGKk6JrGEzuYqM9S6+8gRtNZYlmTIVL9kxN7Elv9/WAeIsLPqsGfXHsE4GIWsZwHMY9vTk7T2Z/CnUQT8JDMyR9kdMJ4lU6f3Px4wSHGFOB2+qlEkhUj2mY7mtFGFzS5kdiwPQQTtEE7xgqjxEiGYIpvFEtFdncfLrFu2qYpkLW6J6wPJ3L0YuwTwFEg5qPzEH02YgSh0UXA2RfdUDHrfpAYNT+SZrtCxDwWjGEwM1oGWnTTlG2E9Yr9uERXCTDsVhNOATABdhtDLYuYAQjZF4B5vn1i9S9Jlk93zeDIQkQw8VgNSsUMQMy+NaIJxhU1gUWs66QGHAZzRiC+qc+miBhrMRo33yCm0ZoICeP6Odn34yIg5LpnvcjkpMpDqVF1hLjXrYNJQCIU8ZLLRkzjl3JOQSrWuhZP3Qxm3cdSI1ZsFtCVIqa19zDEuOCO2ogBaxzlZnhA0JbUE/MlD6t/7jxwFaLkYZ7a1DwoRyVmuQ+gvuYhiS+pPHC7t7giPTEvvX5TEevfpdSbfJDJrPVJ14qcwQyPF3nQz9G45pMxIxFXIeblAk7RLDBhKRCl/gEY2UfiAUEOLPWfrXqcPQUwQM76nJKXhxQqMu75eeSKuCfTBGtFkf3zFESqv3xrDmKeCTiWF2s+pCNP1RP32pL7sEQcB/iD9W0eEAwrMsXlglTEZI+qmWYxQ4p3ZOvPgQbXO5IDJF0TnVxRTpBmseB0r+/qIK7TjXJwsZ8OMtwHk6Mos3jH/kHiqe//8XK5XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6X629R/wEmT5kw6y0H4AAAAABJRU5ErkJggg=="
            
            analysis_table = {
                "roast_level": roast_levels[predicted_class],
                "confidence_score": confidence,
                "class_probabilities": probabilities,
                "recommended_brewing": get_brewing_recommendation(roast_levels[predicted_class]),
                "chart_image": chart_base64,
                "marked_image": marked_image_base64,
                "beans_detected": 0,
                "individual_analysis": [],
                "analysis_method": "Modelo real con TensorFlow (sin detección de granos)"
            }
            
            print(f"Análisis completado usando modelo real: {roast_levels[predicted_class]}")
            return roast_levels[predicted_class], confidence, analysis_table
        
        # Si llegamos aquí, tenemos tanto TensorFlow como OpenCV disponibles
        # Detectar granos individuales 
        marked_image_base64, individual_beans, bean_positions = detect_coffee_beans(image_path)
        
        # Si no se detectaron granos, analizar la imagen completa con el modelo
        if len(individual_beans) == 0:
            print("No se detectaron granos individuales, analizando imagen completa")
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Realizar predicción con el modelo
            predictions = model.predict(img)
            
            # Niveles de tostión de café
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Crear diccionario de probabilidades
            probabilities = {
                roast_levels[i]: float(predictions[0][i]) 
                for i in range(len(roast_levels))
            }
            
            chart_base64 = generate_bar_chart(probabilities)
            
            analysis_table = {
                "roast_level": roast_levels[predicted_class],
                "confidence_score": confidence,
                "class_probabilities": probabilities,
                "recommended_brewing": get_brewing_recommendation(roast_levels[predicted_class]),
                "chart_image": chart_base64,
                "marked_image": marked_image_base64,
                "beans_detected": 0,
                "individual_analysis": [],
                "analysis_method": "Modelo real con TensorFlow (imagen completa)"
            }
            
            print(f"Análisis completado usando modelo real para imagen completa: {roast_levels[predicted_class]}")
            return roast_levels[predicted_class], confidence, analysis_table
        
        # Analizar cada grano individualmente con el modelo
        individual_results = []
        all_probabilities = {"Dark": 0.0, "Green": 0.0, "Light": 0.0, "Medium": 0.0}
        
        for i, bean_img in enumerate(individual_beans):
            prediction, confidence, probabilities = process_bean_image(bean_img)
            
            # Sumar probabilidades para calcular el promedio después
            for key in all_probabilities:
                all_probabilities[key] += probabilities[key]
            
            # Guardar resultados individuales
            individual_results.append({
                "bean_id": i + 1,
                "position": bean_positions[i],
                "roast_level": prediction,
                "confidence": confidence,
                "probabilities": probabilities
            })
        
        # Calcular promedio de probabilidades
        for key in all_probabilities:
            all_probabilities[key] /= len(individual_beans)
        
        # Determinar el nivel de tostión predominante
        predominant_level = max(all_probabilities, key=all_probabilities.get)
        avg_confidence = all_probabilities[predominant_level]
        
        # Generar gráfico
        chart_base64 = generate_bar_chart(all_probabilities)
        
        analysis_table = {
            "roast_level": predominant_level,
            "confidence_score": avg_confidence,
            "class_probabilities": all_probabilities,
            "recommended_brewing": get_brewing_recommendation(predominant_level),
            "chart_image": chart_base64,
            "marked_image": marked_image_base64,
            "beans_detected": len(individual_beans),
            "individual_analysis": individual_results,
            "analysis_method": "Modelo real con detección de granos"
        }
        
        print(f"Análisis completado usando modelo real para {len(individual_beans)} granos: {predominant_level}")
        return predominant_level, avg_confidence, analysis_table
        
    except Exception as e:
        error_msg = f"Error en el procesamiento de imagen: {str(e)}"
        print(error_msg)
        
        # Si hay un error pero TensorFlow está disponible, intentar con imagen completa
        if TENSORFLOW_AVAILABLE and model is not None:
            try:
                img_raw = tf.io.read_file(image_path)
                img = tf.image.decode_image(img_raw, channels=3)
                img = tf.image.resize(img, [224, 224])
                img = img / 255.0
                img = tf.expand_dims(img, 0)
                
                predictions = model.predict(img)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                
                probabilities = {
                    roast_levels[i]: float(predictions[0][i]) 
                    for i in range(len(roast_levels))
                }
                
                chart_base64 = generate_bar_chart(probabilities)
                
                analysis_table = {
                    "roast_level": roast_levels[predicted_class],
                    "confidence_score": confidence,
                    "class_probabilities": probabilities,
                    "recommended_brewing": get_brewing_recommendation(roast_levels[predicted_class]),
                    "chart_image": chart_base64,
                    "error_recovery": True,
                    "analysis_method": "Modelo real (recuperación de error)"
                }
                
                print(f"Análisis recuperado usando modelo real: {roast_levels[predicted_class]}")
                return roast_levels[predicted_class], confidence, analysis_table
            except Exception as e2:
                print(f"Error en la recuperación: {str(e2)}")
        
        # Si todo lo demás falla, usar simulación
        return simulate_prediction_with_beans(image_path)

# Modificar la función simulate_prediction
def simulate_prediction(level=None):
    """
    Simulación básica del análisis de una imagen.
    Permite especificar un nivel (Dark, Light, Medium, Green) o lo genera aleatoriamente.
    """
    import random
    
    roast_levels = ["Dark", "Green", "Light", "Medium"]
    if level is None:
        # Usar Medium como default con mayor probabilidad
        simulated_class = random.choices(roast_levels, weights=[0.2, 0.1, 0.2, 0.5])[0]
    else:
        simulated_class = level if level in roast_levels else "Medium"
    
    simulated_confidence = 0.75
    
    # Probabilidades según el nivel simulado
    if simulated_class == "Dark":
        probabilities = {"Dark": 0.75, "Green": 0.05, "Light": 0.05, "Medium": 0.15}
    elif simulated_class == "Light":
        probabilities = {"Dark": 0.05, "Green": 0.05, "Light": 0.75, "Medium": 0.15}
    elif simulated_class == "Green":
        probabilities = {"Dark": 0.05, "Green": 0.75, "Light": 0.10, "Medium": 0.10}
    else:  # Medium
        probabilities = {"Dark": 0.15, "Green": 0.05, "Light": 0.10, "Medium": 0.70}
    
    # Generar gráfico simulado
    chart_base64 = generate_bar_chart(probabilities)
    
    analysis_table = {
        "roast_level": simulated_class,
        "confidence_score": simulated_confidence,
        "class_probabilities": probabilities,
        "recommended_brewing": get_brewing_recommendation(simulated_class),
        "chart_image": chart_base64,
        "simulation_note": "RESULTADO SIMULADO - No se ha usado análisis real con IA",
        "is_simulated": True  # Nuevo campo para indicar explícitamente que es simulado
    }
    
    return simulated_class, simulated_confidence, analysis_table

# Modificar la función simulate_prediction_with_beans
def simulate_prediction_with_beans(image_path=None):
    """
    Simulación del análisis de una imagen con múltiples granos.
    """
    # Usar una imagen base64 fija para simulación
    marked_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgAQMAAAAPH06nAAAABlBMVEX///8AAABVwtN+AAAACXBIWXMAAA7EAAAOxAGVKw4bAAADHUlEQVR4nO2aTY7bMAxGpUFWWWaVZY7gI/gIPYKP0CP0CB7NbmYzm8HAsgiJlPgnBfKWrzyAIP74JFtq5ji4XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6Xy+VyuVwu18eUNX/iZ/o7/JL5mmbz4JeJkwvx5eIlJPnuHNOGfMgnl4RkDLFpTfZQpjSeHIiBTwT88mNe/wIFOGKk6JrGEzuYqM9S6+8gRtNZYlmTIVL9kxN7Elv9/WAeIsLPqsGfXHsE4GIWsZwHMY9vTk7T2Z/CnUQT8JDMyR9kdMJ4lU6f3Px4wSHGFOB2+qlEkhUj2mY7mtFGFzS5kdiwPQQTtEE7xgqjxEiGYIpvFEtFdncfLrFu2qYpkLW6J6wPJ3L0YuwTwFEg5qPzEH02YgSh0UXA2RfdUDHrfpAYNT+SZrtCxDwWjGEwM1oGWnTTlG2E9Yr9uERXCTDsVhNOATABdhtDLYuYAQjZF4B5vn1i9S9Jlk93zeDIQkQw8VgNSsUMQMy+NaIJxhU1gUWs66QGHAZzRiC+qc+miBhrMRo33yCm0ZoICeP6Odn34yIg5LpnvcjkpMpDqVF1hLjXrYNJQCIU8ZLLRkzjl3JOQSrWuhZP3Qxm3cdSI1ZsFtCVIqa19zDEuOCO2ogBaxzlZnhA0JbUE/MlD6t/7jxwFaLkYZ7a1DwoRyVmuQ+gvuYhiS+pPHC7t7giPTEvvX5TEevfpdSbfJDJrPVJ14qcwQyPF3nQz9G45pMxIxFXIeblAk7RLDBhKRCl/gEY2UfiAUEOLPWfrXqcPQUwQM76nJKXhxQqMu75eeSKuCfTBGtFkf3zFESqv3xrDmKeCTiWF2s+pCNP1RP32pL7sEQcB/iD9W0eEAwrMsXlglTEZI+qmWYxQ4p3ZOvPgQbXO5IDJF0TnVxRTpBmseB0r+/qIK7TjXJwsZ8OMtwHk6Mos3jH/kHiqe//8XK5XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6Xy+VyuVwul8vlcrlcLpfL5XK5XC6X629R/wEmT5kw6y0H4AAAAABJRU5ErkJggg=="
    
    # Valores simulados para los niveles de tostión
    roast_levels = ["Dark", "Green", "Light", "Medium"]
    simulated_class = "Medium"  # Valor simulado por defecto
    simulated_confidence = 0.75
    
    # Valores simulados para las probabilidades
    probabilities = {
        "Dark": 0.15,
        "Green": 0.05,
        "Light": 0.10,
        "Medium": 0.70
    }
    
    # Generar gráfico simulado
    chart_base64 = generate_bar_chart(probabilities)
    
    # Simular resultados de granos individuales
    individual_results = []
    for i in range(5):  # Simular 5 granos
        # Alternar entre Medium y Dark para simular variedad
        bean_level = "Medium" if i % 2 == 0 else "Dark"
        
        # Ligeras variaciones en la confianza
        bean_confidence = 0.7 + (i * 0.05)
        
        # Ligeras variaciones en las probabilidades
        bean_probabilities = {
            "Dark": 0.15 + (i * 0.02),
            "Green": 0.05,
            "Light": 0.10 - (i * 0.01),
            "Medium": 0.70 - (i * 0.01)
        }
        
        individual_results.append({
            "bean_id": i + 1,
            "position": (50 + i*30, 50 + i*20, 25, 25),  # Posiciones simuladas
            "roast_level": bean_level,
            "confidence": bean_confidence,
            "probabilities": bean_probabilities
        })
    
    analysis_table = {
        "roast_level": simulated_class,
        "confidence_score": simulated_confidence,
        "class_probabilities": probabilities,
        "recommended_brewing": get_brewing_recommendation(simulated_class),
        "chart_image": chart_base64,
        "marked_image": marked_image_base64,
        "beans_detected": 5,
        "individual_analysis": individual_results,
        "simulation_note": "///////////// RESULTADO SIMULADO - No se ha usado análisis real con IA /////////////",
        "is_simulated": True  # Nuevo campo para indicar explícitamente que es simulado
    }
    
    return simulated_class, simulated_confidence, analysis_table

# Función para procesar una imagen completa (versión simplificada)
def process_image(image_path):
    """
    Versión simplificada para procesar una imagen completa sin detección de granos.
    """
    roast_levels = ["Dark", "Green", "Light", "Medium"]
    if not TENSORFLOW_AVAILABLE or not CV2_AVAILABLE:
        # Si TensorFlow o OpenCV no está disponible, simular
        return simulate_prediction()
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen desde {image_path}")
            
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Realizar predicción
        predictions = model.predict(img)
        
        # Niveles de tostión de café
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Crear diccionario de probabilidades
        probabilities = {
            roast_levels[i]: float(predictions[0][i]) 
            for i in range(len(roast_levels))
        }
        
        # Generar gráfico de barras
        chart_base64 = generate_bar_chart(probabilities)
        
        analysis_table = {
            "roast_level": roast_levels[predicted_class],
            "confidence_score": confidence,
            "class_probabilities": probabilities,
            "recommended_brewing": get_brewing_recommendation(roast_levels[predicted_class]),
            "chart_image": chart_base64
        }
        
        return roast_levels[predicted_class], confidence, analysis_table
    except Exception as e:
        print(f"Error en process_image: {str(e)}")
        return simulate_prediction()

def get_brewing_recommendation(roast_level):
    """
    Proporciona recomendaciones de preparación según el nivel de tostión.
    """
    recommendations = {
        "Dark": "Ideal para espresso o café con leche. Preparar con agua a 90-93°C durante 25-30 segundos.",
        "Light": "Perfecto para métodos de filtrado como V60 o Chemex. Usar agua a 94-96°C con una proporción 1:16 (café:agua).",
        "Medium": "Versátil, funciona bien en French Press o AeroPress. Preparar con agua a 93-95°C durante 2-4 minutos.",
        "Green": "Café crudo, no apto para consumo directo. Debe ser tostado antes de su preparación."
    }
    return recommendations.get(roast_level, "Consultar con un barista especializado")

# Rutas de la API
@app.get("/")
def read_root():
    """
    Endpoint principal con información sobre la API.
    """
    return {
        "message": "Bienvenido a la API de Análisis de Tostión de Café", 
        "status": "online",
        "model_status": "Disponible" if model else "No disponible",
        "model_error": model_error if model_error else None,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "opencv_available": CV2_AVAILABLE,
        "matplotlib_available": MATPLOTLIB_AVAILABLE,
        "endpoints": {
            "/docs": "Documentación interactiva de la API",
            "/coffee-expert/": "Consultas a experto en café",
            "/analyze-roast/": "Análisis del nivel de tostión del café con detección de granos",
            "/system-status/": "Estado detallado del sistema",
            "/visualize-beans/": "Interfaz visual para análisis de granos",
            "/check-tensorflow/": "Verificar funcionamiento de TensorFlow"
        },
        "current_time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "user": "johnnymunox004",
        "mode": "Simulación" if not CV2_AVAILABLE else "Normal"
    }

@app.get("/system-status/")
def system_status():
    """
    Devuelve información detallada sobre el estado del sistema.
    """
    # Información sobre TensorFlow, si está disponible
    tensorflow_info = {}
    if TENSORFLOW_AVAILABLE:
        tensorflow_info = {
            "version": tf.__version__,
            "cuda_available": tf.test.is_built_with_cuda() if hasattr(tf.test, 'is_built_with_cuda') else False,
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0
        }
    
    # Información sobre OpenCV, si está disponible
    opencv_info = {}
    if CV2_AVAILABLE:
        try:
            opencv_info = {
                "version": cv2.__version__,
                "build_info": cv2.getBuildInformation() if hasattr(cv2, 'getBuildInformation') else "No disponible"
            }
        except:
            opencv_info = {"version": "Error obteniendo información"}
    
    # Información sobre dependencias
    dependencies = {
        "tensorflow": str(tf.__version__) if TENSORFLOW_AVAILABLE else "No disponible",
        "opencv": str(cv2.__version__) if CV2_AVAILABLE else "No disponible",
        "numpy": str(np.__version__) if 'np' in globals() else "No disponible",
        "matplotlib": str(matplotlib.__version__) if MATPLOTLIB_AVAILABLE else "No disponible"
    }
    
    # Información sobre el modelo
    model_info = {
        "loaded": model is not None,
        "error": model_error,
        "path": os.path.abspath('final_coffee_tl_model.keras') if os.path.exists('final_coffee_tl_model.keras') else "No encontrado"
    }
    
    return {
        "api_version": "1.0.0",
        "system_time": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "python_version": sys.version,
        "dependencies": dependencies,
        "tensorflow_details": tensorflow_info,
        "opencv_details": opencv_info,
        "model": model_info,
        "openai_api_configured": bool(client.api_key),
        "operating_mode": "Simulación (sin OpenCV)" if not CV2_AVAILABLE else "Normal"
    }

@app.post("/analyze-roast/")
async def analyze_roast(file: UploadFile = File(...)):
    """
    Analiza una imagen de café y determina el nivel de tostión (Dark, Light, Medium).
    Usa el modelo TensorFlow incluso si OpenCV no está disponible.
    """
    # Guardar imagen temporalmente
    temp_file_path = f"temp_{uuid.uuid4()}.jpg"
    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        print(f"Analizando imagen: {temp_file_path}")
        print(f"TensorFlow disponible: {TENSORFLOW_AVAILABLE}, OpenCV disponible: {CV2_AVAILABLE}")
        
        # Procesar imagen con el modelo o simulación
        prediction, confidence, analysis_table = process_image_with_beans(temp_file_path)
        
        # Indicar si es simulado o real
        is_simulated = not (TENSORFLOW_AVAILABLE and model is not None) or analysis_table.get("is_simulated", False)
        
        if is_simulated:
            note = "///////////// RESULTADO SIMULADO - No se ha usado análisis real con IA /////////////"
        else:
            note = "Análisis basado en modelo real"
            if not CV2_AVAILABLE:
                note += " (sin detección de granos individuales)"
        
        return {
            "roast_level": prediction,
            "confidence": confidence,
            "analysis": analysis_table,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "note": note,
            "using_real_model": TENSORFLOW_AVAILABLE and model is not None,
            "is_simulated": is_simulated
        }
    except Exception as e:
        # Proporcionar mensaje de error detallado
        error_detail = f"Error procesando imagen: {str(e)}"
        print(error_detail)  # Imprimir en el registro del servidor
        
        # En caso de error grave, intentar con simulación
        prediction, confidence, analysis_table = simulate_prediction()
        
        return {
            "roast_level": prediction,
            "confidence": confidence,
            "analysis": analysis_table,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "note": "///////////// RESULTADO SIMULADO (debido a error) /////////////",
            "error": error_detail,
            "using_real_model": False,
            "is_simulated": True
        }
    finally:
        # Limpiar archivos temporales
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/coffee-expert/")
async def coffee_expert(query: CoffeeQuery):
    """
    Consulta a un experto en café sobre cualquier tema relacionado.
    """
    try:
        # Verificar que hay una pregunta
        if not query.pregunta:
            return JSONResponse(
                status_code=400,
                content={"error": "Debes proporcionar una pregunta"}
            )
            
        # Llamar a la API de OpenRouter
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://coffee-analysis-app.azurewebsites.net",  
                "X-Title": "Coffee Analysis App", 
                'ngrok-skip-browser-warning': 'true',
                'User-Agent': 'cafeterito-app-client'
            },
            model="mistralai/mistral-small-24b-instruct-2501:free",
            messages=[
                {
                    "role": "system", 
                    "content": 'Eres un experto en café, responde de manera clara y concisa sobre cultivo, procesamiento, tostión y preparación de café.'
                },
                {
                    "role": "user", 
                    "content": query.pregunta
                }
            ],
            max_tokens=200  
        )
        
        respuesta = completion.choices[0].message.content
        
        return {
            "respuesta": respuesta,
            "query": query.pregunta,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo respuesta: {str(e)}")

@app.get("/test-roast-analysis/")
def test_roast_analysis():
    """
    Endpoint de prueba que simula el análisis de tostión sin necesidad de subir un archivo.
    """
    prediction, confidence, analysis_table = simulate_prediction_with_beans(None)
    
    return {
        "message": "Este es un análisis de tostión simulado para pruebas",
        "roast_level": prediction,
        "confidence": confidence,
        "analysis": analysis_table,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "note": "Esta es una respuesta simulada para fines de prueba."
    }

@app.get("/check-tensorflow/")
def check_tensorflow():
    """
    Endpoint para verificar que TensorFlow está funcionando correctamente.
    """
    if not TENSORFLOW_AVAILABLE:
        return {
            "status": "Error",
            "tensorflow_working": False,
            "error_message": "TensorFlow no está disponible",
            "message": "Error al verificar TensorFlow"
        }
        
    try:
        # Crear un tensor simple para verificar que TensorFlow funciona
        sample_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
        result = tf.reduce_sum(sample_tensor).numpy()
        
        # Verificar que el modelo está cargado
        model_info = {
            "model_loaded": model is not None,
            "model_path": 'final_coffee_tl_model.keras',
            "model_exists": os.path.exists('final_coffee_tl_model.keras'),
            "model_size_mb": round(os.path.getsize('final_coffee_tl_model.keras') / (1024 * 1024), 2) if os.path.exists('final_coffee_tl_model.keras') else 0
        }
        
        # Información detallada sobre versiones de dependencias
        versions = {
            "tensorflow": tf.__version__,
            "numpy": np.__version__,
            "opencv": str(cv2.__version__) if CV2_AVAILABLE else "No disponible",
            "python": sys.version
        }
        
        return {
            "status": "OK",
            "tensorflow_working": True,
            "test_calculation": int(result),
            "model_info": model_info,
            "versions": versions,
            "message": "TensorFlow está funcionando correctamente"
        }
    except Exception as e:
        return {
            "status": "Error",
            "tensorflow_working": False,
            "error_message": str(e),
            "error_type": type(e).__name__,
            "message": "Error al verificar TensorFlow"
        }


print(f"Model disponible: {model is not None}")
print(f"Ruta del modelo: {os.path.abspath('final_coffee_tl_model.keras')}")
print(f"Existe archivo del modelo: {os.path.exists('final_coffee_tl_model.keras')}")


@app.get("/visualize-beans/", response_class=HTMLResponse)
def visualize_beans():
    """
    Devuelve una página HTML para visualizar los resultados del análisis de granos.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Visualizador de Análisis de Granos de Café</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                max-width: 900px; 
                margin: 0 auto; 
                padding: 20px;
            }
            .container { margin-bottom: 20px; }
            h1 { color: #5D4037; }
            h2 { color: #795548; }
            .image-container { margin: 20px 0; }
            .bean-details { 
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .bean-card {
                border: 1px solid #BCAAA4;
                border-radius: 5px;
                padding: 10px;
                background-color: #EFEBE9;
                width: calc(33% - 20px);
                box-sizing: border-box;
            }
            .instructions {
                background-color: #FFF8E1;
                padding: 15px;
                border-left: 5px solid #FFB300;
                margin-bottom: 20px;
            }
            #resultDisplay {
                border: 1px solid #BCAAA4;
                padding: 20px;
                border-radius: 5px;
                background-color: #EFEBE9;
                margin-top: 20px;
            }
            button {
                background-color: #795548;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #5D4037;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            .mode-indicator {
                background-color: #F44336;
                color: white;
                padding: 5px 10px;
                border-radius: 3px;
                display: inline-block;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <h1>Visualizador de Análisis de Granos de Café</h1>
        
        <div id="modeIndicator" class="mode-indicator" style="display:none;">
            MODO SIMULACIÓN - OpenCV no disponible
        </div>
        
        <div class="instructions">
            <h3>Instrucciones:</h3>
            <p>1. Sube una imagen de granos de café usando el botón de abajo</p>
            <p>2. La API detectará los granos individuales y analizará su nivel de tostión</p>
            <p>3. Los resultados se mostrarán con imágenes y gráficos</p>
            <p><small><strong>Nota:</strong> Si accedes a través de ngrok y ves una advertencia, 
               añade el encabezado 'ngrok-skip-browser-warning' a tus peticiones o utiliza 
               esta interfaz que ya lo incluye automáticamente.</small></p>
        </div>
        
        <div class="container">
            <h2>Subir imagen para análisis</h2>
            <input type="file" id="coffeeImage" accept="image/*">
            <button onclick="analyzeImage()">Analizar granos</button>
        </div>
        
        <div id="resultDisplay">
            <p>Los resultados del análisis aparecerán aquí...</p>
        </div>
        
        <script>
            // Comprobar estado del sistema al cargar
            window.onload = async function() {
                try {
                    const response = await fetch('/system-status/', {
                        headers: {
                            'ngrok-skip-browser-warning': 'true',  // Añadir encabezado para evitar la advertencia de ngrok
                            'User-Agent': 'cafeterito-app-client'
                        }
                    });
                    const data = await response.json();
                    
                    if (!data.opencv_details || Object.keys(data.opencv_details).length === 0) {
                        document.getElementById('modeIndicator').style.display = 'block';
                    }
                } catch (error) {
                    console.error("Error verificando estado del sistema:", error);
                }
            }
            
            async function analyzeImage() {
                const fileInput = document.getElementById('coffeeImage');
                const resultDisplay = document.getElementById('resultDisplay');
                
                if (!fileInput.files || fileInput.files.length === 0) {
                    resultDisplay.innerHTML = '<p style="color: red;">Por favor selecciona una imagen primero</p>';
                    return;
                }
                
                resultDisplay.innerHTML = '<p>Analizando imagen, por favor espera...</p>';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/analyze-roast/', {
                        method: 'POST',
                        headers: {
                            'ngrok-skip-browser-warning': 'true',  // Añadir encabezado para evitar la advertencia de ngrok
                            'User-Agent': 'cafeterito-app-client'
                        },
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error(`Error: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    displayResults(data);
                } catch (error) {
                    resultDisplay.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
            
            function displayResults(data) {
                const resultDisplay = document.getElementById('resultDisplay');
                const analysis = data.analysis;
                
                let html = `
                    <h2>Resultados del Análisis</h2>
                `;
                
                // Mostrar un banner prominente si es simulado
                if (data.is_simulated) {
                    html += `
                        <div style="background-color: #FFC107; color: #212121; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-weight: bold; border: 2px solid #FF9800;">
                            ///////////// RESULTADO SIMULADO - No se ha usado análisis real con IA /////////////
                        </div>
                    `;
                }
                
                html += `
                    <p><strong>Nivel de tostión predominante:</strong> ${data.roast_level}</p>
                    <p><strong>Confianza:</strong> ${(data.confidence * 100).toFixed(2)}%</p>
                    <p><strong>Recomendación:</strong> ${analysis.recommended_brewing}</p>
                    <p><strong>Nota:</strong> ${data.note || 'Análisis completado'}</p>
                    
                    <div class="image-container">
                        <h3>Imagen con granos detectados:</h3>
                        <img src="data:image/png;base64,${analysis.marked_image}" alt="Granos detectados" style="max-width: 100%;">
                    </div>
                    
                    <div class="image-container">
                        <h3>Gráfico de niveles de tostión:</h3>
                        <img src="data:image/png;base64,${analysis.chart_image}" alt="Gráfico de niveles de tostión" style="max-width: 100%;">
                    </div>
                `;
                
                if (analysis.beans_detected && analysis.beans_detected > 0) {
                    html += `
                        <h3>Análisis individual de granos (${analysis.beans_detected} detectados):</h3>
                        <div class="bean-details">
                    `;
                    
                    analysis.individual_analysis.forEach(bean => {
                        html += `
                            <div class="bean-card">
                                <h4>Grano #${bean.bean_id}</h4>
                                <p>Nivel: ${bean.roast_level}</p>
                                <p>Confianza: ${(bean.confidence * 100).toFixed(2)}%</p>
                                <p>Dark: ${(bean.probabilities.Dark * 100).toFixed(2)}%</p>
                                <p>Green: ${(bean.probabilities.Green * 100).toFixed(2)}%</p>
                                <p>Medium: ${(bean.probabilities.Medium * 100).toFixed(2)}%</p>
                                <p>Light: ${(bean.probabilities.Light * 100).toFixed(2)}%</p>
                            </div>
                        `;
                    });
                    
                    html += `</div>`;
                }
                
                resultDisplay.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    # Este bloque solo se ejecuta cuando se ejecuta directamente este archivo
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)