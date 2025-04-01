from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import time
from openai import OpenAI
import uuid
from pydantic import BaseModel

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Coffee Analysis API",
    description="API para análisis de imágenes de café y consultas a expertos",
    version="1.0.0"
)

# Configurar CORS para permitir solicitudes desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo una sola vez al inicio
try:
    model = tf.keras.models.load_model('final_coffee_tl_model.keras')
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

# Configuración de OpenAI/OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),  # Usa variable de entorno
)

# Definición de modelos de datos
class CoffeeQuery(BaseModel):
    pregunta: str

# Función para procesar imágenes
def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize para el modelo
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir a RGB
    img = img / 255.0  # Normalización
    img = np.expand_dims(img, axis=0)  # Añadir dimensión de batch
    
    # Realizar predicción
    predictions = model.predict(img)
    
    # Suponiendo que es un modelo de clasificación de café
    class_names = ["Healthy", "Rust", "Red Spider Mite", "Cercospora"]  # Ajusta según tu modelo
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    # Crear tabla de análisis como dict
    analysis_table = {
        "diagnosis": class_names[predicted_class],
        "confidence_score": confidence,
        "class_probabilities": {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        },
        "recommended_action": get_recommendation(class_names[predicted_class])
    }
    
    return class_names[predicted_class], confidence, analysis_table

def get_recommendation(diagnosis):
    recommendations = {
        "Healthy": "Continuar con las prácticas de cultivo actuales",
        "Rust": "Aplicar fungicida específico y aislar plantas afectadas",
        "Red Spider Mite": "Aplicar acaricida y aumentar la humedad ambiental",
        "Cercospora": "Aplicar fungicida de cobre y mejorar el drenaje"
    }
    return recommendations.get(diagnosis, "Consultar con un especialista")

# Efecto de escritura para respuestas
def escribir_texto(texto):
    result = ""
    for char in texto:
        result += char
    return result

# Rutas de la API
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Análisis de Café", "status": "online"}

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analiza una imagen de café y devuelve un diagnóstico
    """
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no disponible")
    
    # Guardar imagen temporalmente
    temp_file_path = f"temp_{uuid.uuid4()}.jpg"
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Procesar imagen
        prediction, confidence, analysis_table = process_image(temp_file_path)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "analysis": analysis_table,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")
    finally:
        # Limpiar archivos temporales
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/coffee-expert/")
async def coffee_expert(query: CoffeeQuery):
    """
    Consulta a un experto en café sobre cualquier tema relacionado
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
            },
            model="mistralai/mistral-small-24b-instruct-2501:free",
            messages=[
                {
                    "role": "system", 
                    "content": 'Eres un experto en café, responde de manera clara y concisa sobre cultivo, procesamiento y preparación de café.'
                },
                {
                    "role": "user", 
                    "content": query.pregunta
                }
            ],
            max_tokens=200  
        )
        
        respuesta = completion.choices[0].message.content
        # Aplicar efecto de escritura (para simular el efecto deseado)
        respuesta_formateada = escribir_texto(respuesta)
        
        return {
            "respuesta": respuesta_formateada,
            "query": query.pregunta,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo respuesta: {str(e)}")

if __name__ == "__main__":
    # Este bloque solo se ejecuta cuando se ejecuta directamente este archivo
    # (no cuando se importa como módulo)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)