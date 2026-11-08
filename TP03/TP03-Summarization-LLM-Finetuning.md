# TP: LLM MODEL & FINE-TUNING

### **Fine-Tuning Overview**

- **Definition:** Fine-tunining is a technique that adapts a pretrained model to a specific task by continuing its training on a smaller, task-specific dataset. It leverages knowledge learned from large-scale data to optimize performance on target applications.

- **Flow:** Large corpus of data $\rightarrow$ Pretrained Model $\rightarrow$ Fine-Tuning with a Task-Specific Dataset $\rightarrow$ Final Model.

### **Model & Preprocessor Setup**

**1. Model Selection**

When choosing a model, we evalutate on 5 core factors:

- **Task Alignment:** Matching model architecture and pretraining to specific use case.

- **Model Architecture:** Evaluating architecture suitability and modularity.

- **Model Size and Complexity:** Balancing model size with available hardware resources.

- **Performance Benchmarking:** Assessing transfer learning capabilities such as F1-score, accuracy, precision, and recall score.

- **Community Support and Documentation:** Ensure documentation and framework support from other developers.

**2. Model Loaders (Hugging Face Transformers)**

Loaders pull the model architecture, pretrained weights, and configs into memory:

- **Model-Specific Loaders:** Tied to exact architectures when you know the model type (e.g., `BertForSequenceClassification.from_pretrained('bert-base-uncased')`).

- **AutoModel Loaders:** Dynamically select the correct architecture based on the provided identifier (e.g., `AutoModelForCausalLM.from_pretrained('gpt2')`, `AutoModelForSequenceClassification`).

| AutoModel Loader | Description |
| :--- | :--- |
| `AutoModel` | General-purpose model, no specific task or head attached.[cite: 1] |
| `AutoModelForSequenceClassification` | For models trained for sequence classification tasks (e.g., text classification).[cite: 1] |
| `AutoModelForTokenClassification` | For models used in token classification tasks (e.g., NER).[cite: 1] |
| `AutoModelForQuestionAnswering` | For models designed for question answering tasks.[cite: 1] |
| `AutoModelForCausalLM` | For models trained for autoregressive text generation (e.g., GPT).[cite: 1] |
| `AutoModelForMaskedLM` | For models trained for masked language modeling (e.g., BERT).[cite: 1] |
| `AutoModelForSeq2SeqLM` | For sequence-to-sequence tasks (e.g., T5, BART for translation or summarization).[cite: 1] |
| `AutoModelForSpeechSeq2Seq` | For speech-to-text models (e.g., ASR).[cite: 1] |
| `AutoModelForVision` | For models trained on visual tasks like image classification.[cite: 1] |
| `AutoModelForImageSegmentation` | For image segmentation tasks.[cite: 1] |

**3. Processor Loader**

- Combines tokenizers, image processors, audio processors, and feature extractors depending on the modality.

- **AutoProcessor:** Automatically selects the appropriate processor for multimodal models (e.g., `AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", use_fast=True)`).

**4. Quantization & BitsAndBytesConfig**

- **Quantization** reduces weight precision from 32-bit floating-point (`FP32`) down to lower-bit formats (like `INT8` or `INT4`) to make models smaller, faster, and more memory-efficient.

- Key BitsAndBytesConfig settings include:

    - `load_in_8bit=True` or `load_in_4bit=True`: Specifies weight precision.

    - `bnb_8bit_use_double_quant=True`: Applies secondary quantization for extra efficiency.

    - `bnb_8bit_quant_type="nf4"`: Sets the quantization data type.

    - `bnb_8bit_compute_dtype=torch.float16`: Sets precision for forward pass computations.

**5. Model Setup Example**

Models are initialized with parameters like `quantization_config`, `device_map="auto"`, and custom attention implementations (`attn_implementation="flash_attention_2"`)