# TAMA: Rethinking Table Instruction Tuning

This repository provides the source code and resources for the paper [*Rethinking Table Instruction Tuning*](www.google.com), published in ACL 2025 Findings.

---

## Repository Structure

### `datasets`
Contains processed data derived from existing table understanding benchmarks and the combined datasets used for our hyperparameter exploration.

### `src`
Includes scripts for data processing, evaluation, and visualization. Follow the commands below to replicate experiments and plots presented in the paper:

- **Plot Generation**:
```
cd src/
python -m src.plots.llama31_feverous
```

- **Evaluation**:
```
cd src/
python -m src.evaluation.hitab
```

- **Data Processing**:
```
cd src/
python -m src.process_data.select_train_data
```


Additional scripts for other datasets and plots can be found within respective directories.

---

## Model Training and Inference

We utilize the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) library for model training and inference. Example YAML configuration files are provided under the `yamls` directory.

Modify the YAML configuration files to match your local environment:
- Adjust paths such as `deepspeed`, `dataset_dir`, and `output_dir`.
- Tune hyperparameters like `learning_rate`, `lr_scheduler_type`, and `per_device_train_batch_size` based on your computational resources.

### Example Training Command:
```
cd yamls/
llamafactory-cli train train.yaml
```


For further guidance, detailed examples, or troubleshooting, please consult the original [LLaMA Factory documentation](https://github.com/hiyouga/LLaMA-Factory). You may also open issues directly in this repository or search existing solutions on the [LLaMA Factory GitHub](https://github.com/hiyouga/LLaMA-Factory/issues).

---

## Model Checkpoints

Two of our trained TAMA models are publicly available on Hugging Face:

- [TAMA (lr=5e-7)](https://huggingface.co/MichiganNLP/tama-5e-7)
- [TAMA (lr=1e-6)](https://huggingface.co/MichiganNLP/tama-1e-6)
