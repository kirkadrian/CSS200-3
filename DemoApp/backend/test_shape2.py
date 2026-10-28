import sys
import tensorflow as tf
from main import models_v14, AdvancedColorAugmentation

model = list(models_v14.values())[0]

for layer in reversed(model.layers):
    try:
        if hasattr(layer, 'output'):
            shape = layer.output.shape
            print(f"Layer: {layer.name}, shape: {shape}")
            if len(shape) == 4:
                print("FOUND:", layer.name)
                break
    except Exception as e:
        print(f"Layer: {layer.name}, error: {e}")

