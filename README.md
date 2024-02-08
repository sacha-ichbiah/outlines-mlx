
# outlinesmlx

outlinesmlx is a minimalistic library aims at adapting the Outlines library within the MLX framework. `pip install outlinesmlx`
[Outlines](https://github.com/outlines-dev/outlines/) provides ways to control the generation of language models to make their output more predictable.
Combined with [MLX](https://github.com/ml-explore/mlx), it allows to perform guided-generation with large language models while leveraging Apple Silicon hardware. 

<img src="https://raw.githubusercontent.com/sacha-ichbiah/outlines-mlx/main/logo.png" alt="Outlines-MLX" width=300></img>

## Design principles

We design it as an adapter that replaces the Pytorch parts of the original Outlines library, to replace it with MLX compatible parts.
We will continue to update it actively as Outlines evolves with time. 

### Versioning: 
outlinesmlx-x is the mlx adapter to outlines-x. It can easily be checked with a `pip list`
``` bash
outlines                             0.0.27
outlinesmlx                          0.0.27 
```

## Why Outlines MLX ?

We are convinced that guided generation is an important technology that will define the future of AI applications beyond chatbots. As the Apple Silicon ML accelerators  become increasingly powerful, we want to extend guided-generation capabilities to this family of devices. The original [Outlines](https://github.com/outlines-dev/outlines/) library relies on Pytorch, and adapting it to MLX requires to change many keys components.

## Installation

outlinesmlx can be installed directly from the pipy repository:

``` bash
pip install outlinesmlx
```


## Supported models

The models are imported using the library [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/).

In this way, you can also import seemlessly quantized models. 

You can import any model from the HuggingFace hub using this library. 


### Load model with a MLX backend

Check the Examples folder and the original [Outlines](https://github.com/outlines-dev/outlines/) library for more use cases.

``` python
import outlinesmlx as outlines

model = outlines.models.mlx("mlx-community/Mistral-7B-Instruct-v0.1-4bit-mlx")

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""
answer = outlines.generate.choice(model, ["Positive", "Negative"])(prompt)
```


### Disclaimer

This library is maintained on a monthly basis. Due to the rapid evolution of the MLX framework and the original Outlines library, it may not be up-to-date with their latest advancements. outlinesmlx is designed to perform experiments with guided generation on Apple Silicon. Please check the original [Outlines](https://github.com/outlines-dev/outlines/) library for an up-to-date implementation. 

outlinesmlx is only compatible with mlx models. If you want to do guided generation using transformers or other architectures, please use the original [Outlines](https://github.com/outlines-dev/outlines/) library.

### Contributions

We are welcoming external contributions !


### Citation

Do not forget to cite the original paper !

``` bash
@article{willard2023efficient,
  title={Efficient Guided Generation for LLMs},
  author={Willard, Brandon T and Louf, R{\'e}mi},
  journal={arXiv preprint arXiv:2307.09702},
  year={2023}
}
```
