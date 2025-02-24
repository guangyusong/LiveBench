from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from livebench.conversation import Conversation, get_conv_template
from livebench.model.model_adapter import BaseModelAdapter

class Rwkv7Adapter(BaseModelAdapter):
    """Model adapter for RWKV-7"""

    def match(self, model_path: str):
        return "rwkv7" in model_path.lower()

    def load_model(self, model_path: str, from_pretrained_kwargs: dict) -> Tuple[torch.nn.Module, AutoTokenizer]:
        revision = from_pretrained_kwargs.get("revision", "main")
        
        # Load tokenizer with proper settings
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=revision,
            padding_side="left",  # Better for causal LM
            model_max_length=2048  # From config
        )
        
        # Ensure special tokens are set
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with optimized settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Better for modern GPUs
            device_map="auto",
            attn_implementation="flash_attention_2",
            **from_pretrained_kwargs
        )
        
        # Set generation config
        model.generation_config = GenerationConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
            do_sample=True,  # Enable sampling
            temperature=0.8,  # Good balance for code
            top_k=50,        # Standard top-k
            top_p=0.95,      # Standard top-p
            repetition_penalty=1.1,  # Avoid repetition
            max_new_tokens=1024,  # Reasonable for code generation
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Optimize with torch.compile
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"Warning: torch.compile failed, falling back to default: {e}")
            
        return model, tokenizer

    def get_default_conv_template(self, model_path: str) -> Conversation:
        return get_conv_template("rwkv")
