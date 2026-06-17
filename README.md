# Model Quantisation (Qwen2.5-0.5B, CPU Only)
This Technique based on the PQT Post Quantization Training 
A  model quantisation using Qwen2.5-0.5B-Instruct. The project shows how to quantise a model with Optimum Quanto and run it locally on a CPU.
where a 32 bit Modal into 8 bit

## Files

| File                                | Description                                              |
| ----------------------------------- | -------------------------------------------------------- |
| `01_model_quantisation_guide.ipynb` | Notebook explaining quantisation concepts and examples   |
| `python quant_qwen.py`                  | Downloads and quantises the model, then saves it locally |
| `run_quantized.py`                  | Loads the quantised model and generates responses        |
| `requirements.txt`                  | Project dependencies                                     |

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Quantise the model

```bash
HuggingFace Token Need
$env:HF_TOKEN = "Your TOKEN"; python quant_qwen.py 
python quant_qwen.py 
```

This creates:

```text
qwen-int8/
```

### 2. Run the quantised model

```bash
 python .\v1_quant_qwen.py      
```

Please enter your question: hi 

INPUT : hi
OUTPUT: Hello! How can I assist you today? Please let me know if there's anything specific you'd like to talk about or any questions you have. I'm here to help answer your queries.


### 3. Open the notebook

```bash
jupyter notebook 01_model_quantisation_guide.ipynb
```

## What is Quantisation?

Quantisation reduces the precision of model weights (for example, FP32 → INT8).

Benefits:

* Smaller model size
* Lower memory usage
* Faster inference
* CPU-friendly deployment

## Requirements

* Python 3.10+
* CPU-only machine supported
* No NVIDIA GPU required
