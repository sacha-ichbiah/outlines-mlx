# outlinesmlx

`outlinesmlx` is a minimalistic library aimed at adapting the Outlines library for use with the MLX framework.

To install, use: `pip install outlinesmlx`.

[Outlines](https://github.com/outlines-dev/outlines/) provides methods to control the generation of language models to make their output more predictable. Combined with [MLX](https://github.com/ml-explore/mlx), it enables guided generation with large language models while leveraging Apple Silicon hardware.

<img src="https://raw.githubusercontent.com/sacha-ichbiah/outlines-mlx/main/logo.png" alt="Outlines-MLX" width=300></img>

## Design Principles

We designed it as an adapter that replaces the PyTorch parts of the original Outlines library with MLX compatible components. We will continue to update it actively as Outlines evolves over time.

### Versioning

`outlinesmlx-x` is the MLX adapter for `outlines-x`. Versions can easily be checked with :

```bash
pip list | grep outlines
```

## Why Outlines MLX?

We believe that guided generation is an important technology that will define the future of AI applications beyond chatbots. As Apple Silicon chips become increasingly powerful, we aim to extend guided-generation capabilities to a whole new family of devices. The original [Outlines](https://github.com/outlines-dev/outlines/) library relies on PyTorch, and adapting it to MLX requires changing many key components.

## Installation

`outlinesmlx` can be installed directly from the PyPI repository:

```bash
pip install outlinesmlx
```

## Supported Models

The models are imported using the library [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/).

This allows for seamless importation of quantized models. You can import any model from the HuggingFace hub using this library.

### Load Model with an MLX Backend

Refer to the examples folder and the original [Outlines](https://github.com/outlines-dev/outlines/) library for more use cases.

```python
import outlinesmlx as outlines

model = outlines.models.mlx("mlx-community/Mistral-7B-Instruct-v0.1-4bit-mlx")

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""
answer = outlines.generate.choice(model, ["Positive", "Negative"])(prompt)
```

### Disclaimer

This library is maintained on a monthly basis. Due to the rapid evolution of the MLX framework and the original Outlines library, it may not be up-to-date with their latest advancements. `outlinesmlx` is designed for experiments with guided generation on Apple Silicon. Please refer to the original [Outlines](https://github.com/outlines-dev/outlines/) library for an up-to-date implementation.

`outlinesmlx` is only compatible with MLX models. If you wish to perform guided generation using transformers or other architectures, please use the original [Outlines](https://github.com/outlines-dev/outlines/) library.

### Contributions

We welcome external contributions!

### Citation

Please do not forget to cite the original paper:

```bibtex
@article{willard2023efficient,
  title={Efficient Guided Generation for LLMs},
  author={Willard, Brandon T and Louf, R{\'e}mi},
  journal={arXiv preprint arXiv:2307.09702},
  year={2023}
}
```