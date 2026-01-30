<div align="center">

# 📊 ANALIZADOR DE SENTIMIENTOS
### Proyecto de IA – Segundo Bimestre  

</div>

---

## 📚 Información Académica

**Institución:** Escuela Politécnica Nacional    
**Carrera:** Escuela de Formación de Tecnólogos    
**Asignatura:** Fundamentos de Inteligencia Artificial    
**Docente:** Ing. Vanessa Guevara    
**Período Académico:** 2025-B  

---

## 👩‍💻 Integrantes

- Nayely del Rocio Ayol Guanoluisa  
- Jhosselin Britani Naula Charco  
- Alessia de los Ángeles Pérez Palacios  

---

## 🎯 Descripción del Proyecto

**Analizador de Sentimientos** es un sistema interactivo basado en Inteligencia Artificial que analiza reseñas de productos para clasificar los comentarios como **positivos, negativos o neutros**. 

El sistema utiliza un modelo de Machine Learning (**Regresión Logística**) entrenado localmente para la clasificación técnica y se integra con la API de **Google Gemini** para generar una respuesta empática y humana que valide la emoción del usuario.

---

## 🚀 Funcionalidades

- 🤖 **Análisis Híbrido:** Clasificación mediante ML y respuesta creativa vía IA Generativa (Gemini 2.0 Flash).  
- 🧠 **Clasificación Precisa:** Diferenciación entre opiniones positivas y negativas con umbrales de confianza calibrados.  
- 🗂️ **Procesamiento en Español:** Limpieza de texto (Regex) y manejo de caracteres especiales.  
- 📈 **Métricas de Evaluación:** Modelo validado con matrices de confusión y reportes de clasificación.  
- 🌐 **Interfaz Web:** Experiencia de usuario moderna desarrollada con React y Vite.

---

## 🛠️ Stack Tecnológico

| Componente | Tecnologías |
| :--- | :--- |
| **Frontend** | React (Vite), Axios, Tailwind CSS |
| **Backend** | Python, FastAPI, Uvicorn |
| **IA & ML** | Scikit-learn, TF-IDF, Google GenAI SDK |
| **Despliegue** | Vercel (Full Stack) |

---

## 🔗 Accesos Rápidos

<a href="https://proyecto-inteligencia-a-jt23.vercel.app" target="_blank">
  <img src="https://img.shields.io/badge/Demo-Ver%20Sitio%20En%20Vivo-brightgreen?style=for-the-badge&logo=vercel">
</a>

<a href="https://github.com/alessia-23/Proyecto_Inteligencia_Artificial" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-Ver%20Repositorio-black?style=for-the-badge&logo=github">
</a>

<a href="https://drive.google.com/file/d/1Go2qRoce8dZhqms6s3pf5MYLyS2mfAz0/view?usp=sharing" target="_blank">
  <img src="https://img.shields.io/badge/Video-Ver%20Demo-pink?style=for-the-badge&logo=vimeo">
</a>

<a href="https://gamma.app/docs/Sistema-de-Analisis-de-Sentimientos-con-IA-oi7fytolio4lfhl" target="_blank">
  <img src="https://img.shields.io/badge/Gamma-Presentación-blue?style=for-the-badge&logo=gamma&logoColor=white">
</a>

---

## 📊 Arquitectura del Sistema

El flujo de datos del proyecto sigue estos pasos:  

1. **Entrada:** El usuario ingresa un comentario en la interfaz de React.  
2. **Procesamiento:** FastAPI recibe el texto y lo limpia.  
3. **Predicción ML:** El vectorizador TF-IDF transforma el texto y el modelo `sentiment_model.pkl` predice la polaridad.  
4. **Generación IA:** Gemini recibe la predicción y redacta un mensaje corto y empático.  
5. **Salida:** El usuario visualiza el sentimiento detectado y el mensaje de la IA.

---

## ⚙️ Instalación y Configuración Local

### 1. Clonar el repositorio
```bash
git clone https://github.com/alessia-23/Proyecto_Inteligencia_Artificial.git
cd Proyecto_Inteligencia_Artificial
