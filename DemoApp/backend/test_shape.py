import sys
import tensorflow as tf
from main import models_v14, AdvancedColorAugmentation

print("Loaded models:", list(models_v14.keys()))

model = models_v14.get("Our Model S1") or list(models_v14.values())[0]

for layer in reversed(model.layers):
    try:
        shape = layer.output_shape
        print(f"Layer: {layer.name}, shape: {shape}")
        if isinstance(shape, tuple) and len(shape) == 4:
            print("FOUND:", layer.name)
            break
    except Exception as e:
        print(f"Layer: {layer.name}, error: {e}")

