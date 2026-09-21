# LoRA Fine-Tuning with TinyLlama

This project demonstrates parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation) for a small instruction-tuned language model. The notebook fine-tunes TinyLlama on the Databricks Dolly 15k dataset and evaluates the model before and after training.

## Objective

The goal is to:

- load the TinyLlama base model
- prepare and format the Dolly instruction dataset
- apply LoRA adapters to the model
- train with supervised fine-tuning (SFT)
- save the trained adapter
- compare the base model output with the fine-tuned model output

## Model and dataset

- Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Dataset: databricks/databricks-dolly-15k
- Training method: SFT + LoRA
- Target modules: q_proj, v_proj

## Main notebook workflow

The notebook is organized into the following sections:

1. Install dependencies
2. Verify CUDA and PyTorch setup
3. Define hyperparameters
4. Load tokenizer and base model
5. Generate sample responses from the base model
6. Load and format the Dolly dataset into chat messages
7. Configure LoRA adapters
8. Train the model with SFTTrainer
9. Save the final adapter
10. Compare base vs. fine-tuned outputs

## Key configuration

The notebook uses the following main settings:

- NUM_TRAIN_SAMPLES = 1000
- MAX_SEQ_LENGTH = 512
- LORA_R = 8
- LORA_ALPHA = 16
- LORA_DROPOUT = 0.05
- LEARNING_RATE = 2e-4
- NUM_EPOCHS = 1
- BATCH_SIZE = 4
- GRAD_ACCUM_STEPS = 4
- OUTPUT_DIR = ./sft-lora-qwen

## Environment requirements

A CUDA-enabled GPU is strongly recommended for training. The notebook installs and uses:

- transformers
- trl
- peft
- datasets
- accelerate
- torch
- torchvision

## Setup

Run the notebook in a Python environment with GPU support. Install dependencies first:

```bash
pip install -U --ignore-installed transformers trl peft datasets accelerate torch -q
```

If you are using a specific CUDA build, install the matching PyTorch wheel before running training.

## Training notes

This approach keeps the base model frozen and trains only a small set of LoRA weights. This reduces memory usage and training cost while still improving instruction-following behavior on the target task.

## Outputs

After training, the notebook saves:

- the LoRA adapter in the output directory
- the tokenizer alongside the adapter
- generated responses for evaluation

The adapter is saved under:

```text
./sft-lora-qwen/final-adapter
```

## Expected result

The notebook compares the original model with the fine-tuned adapter and shows how instruction-following quality changes on example prompts.

## Notes

- Training runtime depends on GPU memory and dataset size.
- This notebook is intended as a practical learning example for LoRA-based fine-tuning.
- Use smaller datasets or fewer training samples when testing in constrained environments.

## References

- Hugging Face Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- TRL (Transformer Reinforcement Learning)
- Databricks Dolly dataset

