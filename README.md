# TAMA

Source code for the paper [Rethinking Table Instruction Tuning](www.google.com) at ACL 2025 Findings.


### Content of This Repo
`datasets`: Processed data from existing table understanding benchmarks, our combined datasets for hyperparameter exploration.

`src`: 
- under which you may run the script to replicate our plots in the paper: 
```
cd src/
python -m src.plots.llama31_feverous
```
and other plots.

- evaluation script 
```
cd src/
python -m src.evaluation.hitab
```
and other datasets.

- process data
```
cd src/
python -m src.process_data.select_train_data
```

### Model Training and Inference

We leverage the [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) library for our training and inference, here we provide two examples of the training and inference yaml files.

You may modify the value in the yaml file such as `deepspeed`, `dataset_dir`, `output_dir` to point to the correct path, and adjust the values for `learning_rate`, `lr_scheduler_type`, `per_device_train_batch_size` based on your server setup and choices.

For training,
```
cd yamls/
llamafactory-cli train train.yaml
```

You may refer to the original [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) for detailed examples.
You may open up issues here or search in the original [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) repo for issues of running the script.

### Model Checkpoints

We open source two of our trained TAMA models on huggingface:

- [TAMA @ lr=5e-7](https://huggingface.co/MichiganNLP/tama-5e-7)
- [TAMA @ lr=1e-6](https://huggingface.co/MichiganNLP/tama-1e-6)

