from transformers import AutoModelForCausalLM, AutoTokenizer
#load the model and tokenizer from huggingface
#load the quantization library from optimum
from optimum.quanto import QuantizedModelForCausalLM, qint8

# The model we want and the folder where we will save the small version.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SAVE_FOLDER = "./qwen-int8"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
small_model = QuantizedModelForCausalLM.quantize(model, weights=qint8)
print("Saving the small model to:", SAVE_FOLDER)
small_model.save_pretrained(SAVE_FOLDER)
tokenizer.save_pretrained(SAVE_FOLDER)

print("Done! Now run:  python run_quantized.py")