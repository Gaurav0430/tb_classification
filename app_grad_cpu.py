import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from PIL import Image
import uvicorn
import io
import cv2
import tempfile
from fastapi.responses import JSONResponse

app = FastAPI()

# Model paths
MODEL_1_PATH = r"C:\Users\MSI\TuberculosisNet\working\model_train_deep\tbnet_stable_model.h5"
MODEL_2_PATH = r"C:\Users\MSI\TuberculosisNet\models\tbnet_model.h5"
OUTPUT_DIR = r"C:\Users\MSI\TuberculosisNet\working\output_images"


os.makedirs(OUTPUT_DIR, exist_ok=True)


graph1 = tf.get_default_graph()
session1 = tf.Session()
with session1.as_default():
    with graph1.as_default():
        model1 = load_model(MODEL_1_PATH)


graph2 = tf.get_default_graph()
session2 = tf.Session()
with session2.as_default():
    with graph2.as_default():
        model2 = load_model(MODEL_2_PATH)


MODEL_LAYERS = {
    "model1": "conv2d_7",  
    "model2": "conv2d_2"  
}

def grad_cam(model, img_array, layer_name, session, graph):
    """Generate Grad-CAM heatmap for the given model and image"""
    
    with session.as_default():
        with graph.as_default():
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[model.get_layer(layer_name).output, model.outputs[0]]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                predicted_class = tf.argmax(predictions[0])
                class_channel = predictions[:, predicted_class]
                
            grads = tape.gradient(class_channel, conv_outputs)
            
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            pooled_grads_value = session.run(pooled_grads)
            conv_outputs_value = session.run(conv_outputs)[0]
            predicted_class_value = session.run(predicted_class)
            
            for i in range(pooled_grads_value.shape[0]):
                conv_outputs_value[:, :, i] *= pooled_grads_value[i]
                
            heatmap = np.mean(conv_outputs_value, axis=-1)
            
            heatmap = np.maximum(heatmap, 0)
            
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
                
            return heatmap, predicted_class_value

def create_gradcam_visualization(original_img, heatmap, size=(224, 224)):
    """Create and return Grad-CAM visualization"""
    heatmap = cv2.resize(heatmap, size)
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    original_img = cv2.resize(original_img, size)
    
    superimposed_img = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)
    
    return superimposed_img

@app.post("/predict-with-gradcam/")
async def predict_with_gradcam(file: UploadFile = File(...)):
    global graph1, session1, graph2, session2, model1, model2
    
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        img_for_cv = np.array(img)
        img_for_cv = cv2.cvtColor(img_for_cv, cv2.COLOR_RGB2BGR)  
        
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with session1.as_default():
            with graph1.as_default():
                pred1 = model1.predict(img_array)
        
        with session2.as_default():
            with graph2.as_default():
                pred2 = model2.predict(img_array)
        
        avg_prediction = (pred1 + pred2) / 2
        
        predicted_class = np.argmax(avg_prediction)
        confidence = float(np.max(avg_prediction))
        
        model1_confidence = float(pred1[0][predicted_class])
        model2_confidence = float(pred2[0][predicted_class])
        
        if model1_confidence > model2_confidence:
            best_model = model1
            best_model_name = "model1"
            best_session = session1
            best_graph = graph1
        else:
            best_model = model2
            best_model_name = "model2"
            best_session = session2
            best_graph = graph2
        
        best_layer = MODEL_LAYERS[best_model_name]
        
        heatmap, _ = grad_cam(
            model=best_model, 
            img_array=img_array, 
            layer_name=best_layer,
            session=best_session,
            graph=best_graph
        )
        
        gradcam_image = create_gradcam_visualization(img_for_cv, heatmap)
        
        temp_file_path = os.path.join(OUTPUT_DIR, f"gradcam_{file.filename}")
        cv2.imwrite(temp_file_path, gradcam_image)
        
        _, buffer = cv2.imencode('.png', gradcam_image)
        gradcam_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "filename": file.filename,
            "prediction": "TB" if predicted_class == 1 else "Non-TB",
            "confidence": confidence,
            "model1_confidence": model1_confidence,
            "model2_confidence": model2_confidence,
            "best_model": "Model 1" if best_model_name == "model1" else "Model 2",
            "best_layer_used": best_layer,
            "gradcam_image_path": temp_file_path,
            "gradcam_image_base64": gradcam_base64
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global graph1, session1, graph2, session2
    
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        with session1.as_default():
            with graph1.as_default():
                pred1 = model1.predict(img_array)
        
        with session2.as_default():
            with graph2.as_default():
                pred2 = model2.predict(img_array)
        
        avg_prediction = (pred1 + pred2) / 2
        
        predicted_class = np.argmax(avg_prediction)
        confidence = float(np.max(avg_prediction))
        
        return {
            "filename": file.filename,
            "prediction": "TB" if predicted_class == 1 else "Non-TB",
            "confidence": confidence,
            "model1_confidence": float(pred1[0][predicted_class]),
            "model2_confidence": float(pred2[0][predicted_class])
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)