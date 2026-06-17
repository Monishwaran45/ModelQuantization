from transformers import AutoTokenizer
from optimum.quanto import QuantizedModelForCausalLM 
# The folder where quantize_qwen.py saved the small model.
SAVE_FOLDER = "./qwen-int8"
#load the small model and tokenizer from the local folder 
tokenizer = AutoTokenizer.from_pretrained(SAVE_FOLDER) #load the tokenizer
model = QuantizedModelForCausalLM.from_pretrained(SAVE_FOLDER)  #load the quantized model, this will be the small 8-bit version, we will use it to answer a question in the next step

#  write your question (the INPUT) 
question = input("Please enter your question: ").strip()

# Qwen is a chat model, so we put our question in chat format.
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Give accurate and concise answers. "
            "For technical topics, explain step-by-step with examples. "
            "Use bullet points when appropriate."
        )
    },
    {
        "role": "user",
        "content": question
    }
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
#let the model generate an answer (the OUTPUT)
output = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)
# Keep only the new words the model added (remove our question).
answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("\nINPUT :", question)
print("OUTPUT:", answer)