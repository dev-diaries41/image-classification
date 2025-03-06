import torch
import ai_edge_torch
from model import ImageClassifier, ImageClassifierMobile

MODEL_PATH = "checkpoint/screenshot_mobile_best.pt"
TFLITE_MODEL_PATH = "models/screenshot_mobile.tflite"

print("Loading PyTorch model...")
model = ImageClassifierMobile(num_classes=3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))

# Convert PyTorch model to TFLite
print("Converting PyTorch model to TFLite...")
sample_inputs = (torch.randn(1, 3, 224, 224),)

# Convert and serialize PyTorch model to a tflite flatbuffer. Note that we
# are setting the model to evaluation mode prior to conversion.
edge_model = ai_edge_torch.convert(model.eval(), sample_inputs)
edge_model.export(TFLITE_MODEL_PATH)
print(f"Model successfully converted to TFLite: {TFLITE_MODEL_PATH}")
