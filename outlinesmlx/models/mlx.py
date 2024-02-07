from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer
import mlx.core as mx
import mlx_lm


from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Any


from outlines.models.tokenizer import Tokenizer
if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np 
from transformers import AutoTokenizer

__all__ = ["transformers"]

ModelType = Any



def get_llama_tokenizer_types():
    """Get all the Llama tokenizer types/classes that need work-arounds.

    When they can't be imported, a dummy class is created.

    """
    try:
        from transformers.models.llama import LlamaTokenizer
    except ImportError:

        class LlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.llama import LlamaTokenizerFast
    except ImportError:

        class LlamaTokenizerFast:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizer
    except ImportError:

        class CodeLlamaTokenizer:  # type: ignore
            pass

    try:
        from transformers.models.code_llama import CodeLlamaTokenizerFast
    except ImportError:

        class CodeLlamaTokenizerFast:  # type: ignore
            pass

    return (
        LlamaTokenizer,
        LlamaTokenizerFast,
        CodeLlamaTokenizer,
        CodeLlamaTokenizerFast,
    )

class MLX:
    """Represents a MLX model."""

    def __init__(
        self,
        model: ModelType,
        tokenizer: "PreTrainedTokenizer",
    ):
        self.model = model
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        past_key_values: mx.array,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """Compute a forward pass through the transformer model.

        Parameters
        ----------
        input_ids
            The input token ids.  Must be one or two dimensional.
        attention_mask
            The attention mask.  Must be one or two dimensional.
        past_key_values
            A tuple of tuples containing the cached key and value tensors for each
            attention head.

        Returns
        -------
        The computed logits and the new cached key and value tensors.

        """
        assert 0 < input_ids.ndim < 3

        if past_key_values:
            input_ids = input_ids[..., -1][...,None]

        #print("Shape input ids", input_ids.shape)

        logits, kv_cache = self.model(
            input_ids,
            cache=past_key_values,
        )

        return logits, kv_cache

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        past_key_values: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        logits, kv_cache = self.forward(input_ids,None, past_key_values)
        next_token_logits = logits[..., -1, :]

        return next_token_logits, kv_cache



class TransformerTokenizer(Tokenizer):
    """Represents a tokenizer for models in the `transformers` library."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.eos_token

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.pad_token = self.tokenizer.pad_token

        self.special_tokens = set(self.tokenizer.all_special_tokens)

        self.vocabulary = self.tokenizer.get_vocab()
        self.is_llama = isinstance(self.tokenizer, get_llama_tokenizer_types())

    def encode(
        self, prompt: Union[str, List[str]], **kwargs
    ) -> Tuple[mx.array, mx.array]:
        kwargs["padding"] = True 
        kwargs["return_tensors"] = "np"
        output = self.tokenizer(prompt, **kwargs)
        return mx.array(output["input_ids"]), mx.array(output["attention_mask"])

    def decode(self, token_ids: mx.array) -> List[str]:
        text = self.tokenizer.batch_decode(np.array(token_ids), skip_special_tokens=True)
        return text

    def convert_token_to_string(self, token: str) -> str:
        from transformers.file_utils import SPIECE_UNDERLINE

        string = self.tokenizer.convert_tokens_to_string([token])

        if self.is_llama:
            # A hack to handle missing spaces to HF's Llama tokenizers
            if token.startswith(SPIECE_UNDERLINE) or token == "<0x20>":
                return " " + string

        return string

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return other.model_name == self.model_name and other.kwargs == self.kwargs
        return NotImplemented

    def __hash__(self):
        from datasets.fingerprint import Hasher

        return hash(Hasher.hash(self.tokenizer))

    
def mlx(
    model_name: str,
    tokenizer_kwargs: dict = {},
    adapter_file: Optional[str] = None
):
    
    
    """Instantiate a model from a predefined set of models and its tokenizer from 'transformers'.

    Parameters
    ----------
    model_name
        The name of the model as listed on Hugging Face's model page.
    tokenizer_kwargs
        A dictionary that contains the keyword arguments to pass to the
        `from_pretrained` method when loading the tokenizer.
    adapter_file : Optional[str], optional
        Path to the adapter file. If provided, applies LoRA layers to the model. Defaults to None.
        

    Returns
    -------
    A `MLXModel` model instance.

    """

    model, hf_tokenizer =  mlx_lm.load(model_name, tokenizer_kwargs, adapter_file)
    tokenizer_kwargs['trust_remote_code']=True
    tokenizer = TransformerTokenizer(hf_tokenizer)

    return MLX(model, tokenizer)

