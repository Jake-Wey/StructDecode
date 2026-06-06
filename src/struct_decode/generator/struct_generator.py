"""StructGenerator - Main entry point for structured generation."""

from typing import Optional, Union, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from struct_decode.processors.mask_processor import TokenMaskProcessor
from struct_decode.processors.base import BaseLogitsProcessor, LogitsProcessorList
from struct_decode.samplers import BaseSampler, GreedySampler, NucleusSampler

class StructGenerator:
    """
    Main class for structured text generation with LLMs.

    This class wraps a HuggingFace model and provides constrained decoding
    capabilities through logits processors and custom samplers.

    Example:
        >>> generator = StructGenerator(model_name="Qwen/Qwen2.5-1.5B-Instruct")
        >>> output = generator.generate(
        ...     prompt="What is the answer?",
        ...     allowed_tokens={"Yes", "No"}
        ... )
    """

    def __init__(
        self,
        model_name: str = "QWen/QWen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float16,
        **model_kwargs
    ):
        """
        Initialize the generator with a HuggingFace model.

        Args:
            model_name: Name or path of the HuggingFace model.
            device: Device to load the model on. If None, auto-detects.
            torch_dtype: Data type for model weights.
            **model_kwargs: Additional arguments for model loading.
        """

        self.model_name = model_name
        self.device = device or ("cude" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            trust_remote_code=True,
            **model_kwargs
        )
        self.model.eval()

        self.vocab_size = len(self.tokenizer)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id or self.eos_token_id

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        allowed_tokens: Optional[Union[set, List[str]]] = None,
        allowed_token_ids: Optional[set] = None,
        logits_processors: Optional[List[BaseLogitsProcessor]] = None,
        sampler: Optional[BaseSampler] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
        stop_strings: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text with optional constrained decoding.

        Args:
            prompt: The input prompt to continue.
            max_new_tokens: Maximum number of tokens to generate.
            allowed_tokens: Set of allowed tokens (as strings). Simple constraint.
            allowed_token_ids: Set of allowed token IDs. Alternative to allowed_tokens.
            logits_processors: Custom logits processors to apply.
            sampler: Custom sampler to use. If None, uses greedy or nucleus.
            temperature: Sampling temperature (used if do_sample=True).
            top_p: Nucleus sampling threshold (used if do_sample=True).
            do_sample: Whether to use sampling. If False, uses greedy.
            stop_strings: Strings that stop generation when encountered.
            **kwargs: Additional generation arguments.

        Returns:
            The generated text (without the prompt).
        """

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_length = input_ids.shape[1]

        processors = []

        if allowed_tokens is not None or allowed_token_ids is not None:
            if allowed_tokens is not None:
                allowed_token_ids = set()
                for token in allowed_tokens:
                    token_id = self.tokenizer.encode(token, add_special_tokens=False)
                    if len(token_id) == 1:
                        allowed_token_ids.add(token_id[0])
                    else:
                        allowed_token_ids.update(token_id)

            mask_processor = TokenMaskProcessor(
                allowed_token_ids=allowed_token_ids,
                vocab_size=self.vocab_size
            )
            processors.append(mask_processor)

        if logits_processors:
            processors.extend(logits_processors)

        processor_list = LogitsProcessorList(processors)

        if sampler is None:
            if do_sample:
                sampler = NucleusSampler(top_p=top_p, temperature=temperature)
            else:
                sampler = GreedySampler()

        processor_list.reset()

        with torch.no_grad():
            output_ids = self._generate_loop(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                processors=processor_list,
                sampler=sampler,
                stop_strings=stop_strings
            )

        generated_text = self.tokenizer.decode(
            output_ids[0, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return generated_text
    
    def _generate_loop(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        processors: LogitsProcessorList,
        sampler: BaseSampler,
        stop_strings: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Main generation loop with constrained decoding.

        Args:
            input_ids: Starting token IDs. Shape: (1, seq_len)
            max_new_tokens: Maximum tokens to generate.
            processors: Logits processors to apply.
            sampler: Sampler for token selection.
            stop_strings: Stop generation when these strings appear.

        Returns:
            Complete sequence including input.
        """

        output_ids = input_ids.clone()
        past_key_values = None

        for _ in range(max_new_tokens):
            if past_key_values is None:
                outputs = self.model(output_ids, use_cache=True)
            else:
                outputs = self.model(
                    output_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            logits = processors(output_ids, logits)

            next_token = sampler.sample(logits)

            output_ids = torch.cat([output_ids, next_token.unsqueeze(-1)], dim=-1)

            if next_token.item() == self.eos_token_id:
                break

            if stop_strings:
                current_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                for stop_str in stop_strings:
                    if stop_str in current_text:
                        break

        return output_ids
    
    def generate_with_regex(
        self,
        prompt: str,
        regex_pattern: str,
        max_new_tokens: int = 100,
        sampler: Optional[BaseSampler] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text constrained to match a regex pattern.

        Args:
            prompt: Input prompt.
            regex_pattern: Regex pattern to match.
            max_new_tokens: Maximum tokens to generate.
            sampler: Custom sampler. If None, uses greedy or nucleus based on do_sample.
            do_sample: Whether to use sampling.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text matching the regex pattern.

        Raises:
            RuntimeError: If the C++ engine is not available.
            ValueError: If the regex pattern is not supported.
        """

        # TODO
        return ""