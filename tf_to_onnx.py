import onnx
from onnx_tf.backend import prepare

# Replace 'path/to/your/tf_model' with the actual path to your TensorFlow model
tf_model_path = 'path/to/your/tf_model'

try:
    # Load the TensorFlow model
    tf_rep = prepare(tf_model_path)

    # Convert the TensorFlow model to ONNX format
    onnx_model_path = 'output_model.onnx'
    onnx_model = tf_rep.export_graph(onnx_model_path)

    print("ONNX model successfully saved to:", onnx_model_path)

except Exception as e:
    print("An error occurred:", e)
