# Notebook Copilot

## Overview

This project is a tool for creating a dataset of Jupyter notebooks with FIM (Fill In The Middle) examples, and how fine-tune a model to generate FIM code specific to `jupyter` notebooks.

## Create Data

To create a dataset of Jupyter notebooks with FIM (Fill In The Middle) examples, run the following command:

```bash
python jupyter_fim.py
```

Within that file, you can specify the path containing the notebooks you want to use to create the dataset, and the path to save the dataset.

## Fine-Tune Model

To fine-tune a model to generate FIM code specific to `jupyter` notebooks, run the following command:

```bash
python jupyter_fine_tune.py
```

Inside there are a ton of parameters you'll need to set. Please refer to the tutorials at the bottom which got this working for me.

## Deploy Model

Once you have a fine-tuned model, the LoRA adapter weights should be pushed to the Hugging Face Hub. From there what I did was use this [space](https://huggingface.co/spaces/noahksman/notebook-copilot) to convert it to a GGUF file, at which point you can turn it into an ollama model and host it locally to be used by the Continue.dev extension.
