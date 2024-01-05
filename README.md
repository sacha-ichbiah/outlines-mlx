

# Outlines-mlx
    
Outlines MLX is a minimalistic implementation of the Outlines library within the MLX framework.
[Outlines](https://github.com/outlines-dev/outlines/) provides ways to control the generation of language models to make their output more predictable.
Combined with [MLX](https://github.com/ml-explore/mlx), it allows to perform guided-generation with large language models while leveraging Apple Silicon hardware. 

<img src="logo.png" alt="Outlines-MLX" width=300></img>



## Installation
``` bash
git clone sachaichbiah/outlines-mlx
cd outlines-mlx
pip install -e . 
```


## Features

Check the original repository to see the available features.


## Supported models

The supported models are:

| Models                             | 
|------------------------------------|
| TinyLlama/TinyLlama-1.1B-Chat-v0.6 |
| microsoft/phi-2                    |
| mistralai/Mistral-7B               |



### Load model with a MLX backend

Check the original [Outlines](https://github.com/outlines-dev/outlines/) library for more use cases.

``` python
import outlines

model = outlines.models.mlx("TinyLlama/TinyLlama-1.1B-Chat-v0.6")

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome!
"""
answer = outlines.generate.choice(model, ["Positive", "Negative"])(prompt)
```



### Model quantization


``` python
import outlines

#model = outlines.models.mlx("microsoft/phi-2",model_kwargs={'trust_remote_code':True, 'quantize':True, 'q_group_size':64,"q_bits":4, "force_conversion":True}, tokenizer_kwargs= {'trust_remote_code':True})
model = outlines.models.mlx("mistralai/Mistral-7B-Instruct-v0.2",model_kwargs={'trust_remote_code':True, 'quantize':True, 'q_group_size':64,"q_bits":4, "test_loading_instruct":True,"force_conversion":True},tokenizer_kwargs= {'trust_remote_code':True})

prompt = "What is the IP address of the Google DNS servers? "

guided = outlines.generate.regex(model, r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)", max_tokens=30)(prompt)

print(guided)
# What is the IP address of the Google DNS servers?
# 2.2.6.1
```

### Disclaimer

This library is not up to date. It was designed to perform experiments with guided generation on Apple Silicon M1/M2. Please check the original [Outlines](https://github.com/outlines-dev/outlines/) library for an up-to-date implementation. 

