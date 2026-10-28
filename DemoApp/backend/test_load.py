import tensorflow as tf
@tf.keras.utils.register_keras_serializable()
class AdvancedColorAugmentation(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AdvancedColorAugmentation, self).__init__(**kwargs)
    def call(self, inputs, training=False):
        return inputs
    def get_config(self):
        return super(AdvancedColorAugmentation, self).get_config()

try:
    tf.keras.models.load_model("models/Optimized_DenseNet-121_StepB_v2_best.keras", custom_objects={'AdvancedColorAugmentation': AdvancedColorAugmentation}, compile=False)
    print("Success loading Step B with compile=False")
except Exception as e:
    print(f"Error: {e}")
