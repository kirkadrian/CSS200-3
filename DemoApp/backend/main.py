import io
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# 1. Define the custom layer exactly as it was during training
@tf.keras.utils.register_keras_serializable()
class AdvancedColorAugmentation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvancedColorAugmentation, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        # Data augmentation is usually inactive during inference
        return inputs
        
    def get_config(self):
        config = super(AdvancedColorAugmentation, self).get_config()
        return config

app = FastAPI(title="E-Waste Classification API")

# 2. Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Hardcode the exact class order to prevent label-shuffle bugs
CLASS_NAMES = ["Motherboard", "Battery", "RAM", "WiFi Card", "CPU Fan"]

# 4. Define paths to models inside the 'models' and 'rerun' folders
MODEL_PATHS_V13 = {
    "Baseline DenseNet": "models/Baseline_DenseNet-121_best.keras",
    "Baseline MobileNet": "models/Baseline_MobileNet-V2_best.keras",
    "Optimized DenseNet (Step A)": "models/Optimized_DenseNet-121_StepA_v2_best.keras",
    "Optimized DenseNet (Step B)": "models/Optimized_DenseNet-121_StepB_v2_best.keras"
}

MODEL_PATHS_V14 = {}
for s in [1, 2, 3]:
    MODEL_PATHS_V14[f"Baseline DenseNet S{s}"] = f"../rerun/Baseline_DenseNet-121_seed{s}_best.keras"
    MODEL_PATHS_V14[f"Baseline MobileNet S{s}"] = f"../rerun/Baseline_MobileNet-V2_seed{s}_best.keras"
    MODEL_PATHS_V14[f"Frozen Control S{s}"] = f"../rerun/Optimized_DenseNet-121_frozen_control_seed{s}_best.keras"
    MODEL_PATHS_V14[f"Our Model S{s}"] = f"../rerun/Optimized_DenseNet-121_unfreeze_bnfrozen_seed{s}_best.keras"

models_v13 = {}
models_v14 = {}

print("Loading V13 models globally...")
for name, path in MODEL_PATHS_V13.items():
    print(f"Loading {name} from {path}...")
    try:
        model = tf.keras.models.load_model(
            path,
            custom_objects={'AdvancedColorAugmentation': AdvancedColorAugmentation},
            compile=False
        )
        models_v13[name] = model
        print(f"[{name}] loaded successfully!")
    except Exception as e:
        print(f"Error loading [{name}]: {e}")

print("Loading V14 models globally...")
for name, path in MODEL_PATHS_V14.items():
    print(f"Loading {name} from {path}...")
    try:
        model = tf.keras.models.load_model(
            path,
            custom_objects={'AdvancedColorAugmentation': AdvancedColorAugmentation},
            compile=False
        )
        models_v14[name] = model
        print(f"[{name}] loaded successfully!")
    except Exception as e:
        print(f"Error loading [{name}]: {e}")

# Warm-up prediction for all models
print("Running warm-up predictions...")
dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
for name, model in models_v13.items():
    try:
        model.predict(dummy_input, verbose=0)
    except Exception as e:
        pass
for name, model in models_v14.items():
    try:
        model.predict(dummy_input, verbose=0)
    except Exception as e:
        pass
print("Warm-up complete.")


def preprocess_image_raw(image: Image.Image) -> np.ndarray:
    """
    Exactly matches the notebook:
    img = image.load_img(..., target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    """
    # Resize to 224x224
    img = image.resize((224, 224))
    # Convert to float32 [0.0, 255.0]
    img_array = np.array(img, dtype=np.float32)
    # Expand dims
    return np.expand_dims(img_array, axis=0)

def make_gradcam_heatmap(img_array, model, pred_index=None):
    # Find last conv layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        try:
            shape = None
            if hasattr(layer, 'output'):
                shape = layer.output.shape
            elif hasattr(layer, 'output_shape'):
                shape = layer.output_shape
                
            if shape is not None:
                if isinstance(shape, list):
                    shape = shape[0]
                if len(shape) == 4:
                    last_conv_layer_name = layer.name
                    break
        except Exception:
            pass
            
    if not last_conv_layer_name:
        raise ValueError("Could not find a convolutional layer.")

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_jet_colormap():
    x = np.linspace(0, 1, 256)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    return np.column_stack((r, g, b))

def apply_heatmap(image: Image.Image, heatmap: np.ndarray, alpha=0.4):
    # Resize heatmap to match image size
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size, Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img)
    
    jet = generate_jet_colormap()
    heatmap_colored = jet[heatmap_resized] * 255
    heatmap_colored = heatmap_colored.astype(np.uint8)
    
    heatmap_pil = Image.fromarray(heatmap_colored).convert("RGBA")
    
    # Adjust opacity
    heatmap_pil.putalpha(int(255 * alpha))
    
    # Overlay on original image
    original_rgba = image.convert("RGBA")
    superimposed_img = Image.alpha_composite(original_rgba, heatmap_pil)
    return superimposed_img.convert("RGB")

async def process_prediction(file: UploadFile, loaded_models: dict):
    if not loaded_models:
        return {"error": "No models were successfully loaded for this version."}
        
    try:
        image_bytes = await file.read()
        
        # Open raw image
        raw_img = Image.open(io.BytesIO(image_bytes))
        if raw_img.mode != "RGB":
            raw_img = raw_img.convert("RGB")
            
        processed_batch = preprocess_image_raw(raw_img)
        
        results = {}
        for name, model in loaded_models.items():
            predictions = model.predict(processed_batch, verbose=0)
            
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index])
            predicted_class = CLASS_NAMES[predicted_index]
            
            results[name] = {
                "prediction": predicted_class,
                "confidence": confidence,
                "all_scores": {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
            }
            
        return results
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict")
@app.post("/predict/v13")
async def predict_v13(file: UploadFile = File(...)):
    return await process_prediction(file, models_v13)

@app.post("/predict/v14")
async def predict_v14(file: UploadFile = File(...)):
    return await process_prediction(file, models_v14)

@app.post("/heatmap/v14")
async def get_heatmap_v14(file: UploadFile = File(...), model_key: str = Form(...)):
    if model_key not in models_v14:
        return {"error": f"Model {model_key} not found in V14."}
    
    model = models_v14[model_key]
    try:
        image_bytes = await file.read()
        raw_img = Image.open(io.BytesIO(image_bytes))
        if raw_img.mode != "RGB":
            raw_img = raw_img.convert("RGB")
            
        processed_batch = preprocess_image_raw(raw_img)
        
        heatmap = make_gradcam_heatmap(processed_batch, model)
        superimposed = apply_heatmap(raw_img, heatmap)
        
        # Convert to Base64
        buffered = io.BytesIO()
        superimposed.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"heatmap_base64": img_str}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "API is running. Send a POST request with an image to /predict"}
