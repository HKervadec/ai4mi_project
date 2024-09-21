from .base import (
    SegVol,
    SegVolConfig,
    SegVolModel,
    SegVolProcessor,
)
from transformers import AutoModel, AutoTokenizer

from peft import LoraConfig, get_peft_model, PeftModel


class SegVolLoRA(SegVolModel):
    """
    Baseline model for SegVol.
    """

    def __init__(
        self,
        config: SegVolConfig,
        target_modules: list[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        train_only_vit: bool = False,
    ):
        super().__init__(config)

        self.model: SegVol = AutoModel.from_pretrained(
            "BAAI/SegVol", trust_remote_code=True, test_mode=config.test_mode
        ).model

        clip_tokenizer = AutoTokenizer.from_pretrained("BAAI/SegVol")
        self.model.text_encoder.tokenizer = clip_tokenizer

        self.processor = SegVolProcessor(spatial_size=self.config.spatial_size)

        if target_modules is None:
            target_modules = [
                "mlp.linear1",
                "mlp.linear2",
                "attn.qkv",
                "self_attn.q_proj",
                "self_attn.v_proj",
                "cross_attn_image_to_token.q_proj",
                "cross_attn_image_to_token.v_proj",
                "final_attn_token_to_image.q_proj",
                "final_attn_token_to_image.v_proj",
            ]
        peft_config = LoraConfig(
            target_modules=target_modules,
            inference_mode=config.test_mode,
            r=lora_r,
            use_rslora=True,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # FIXME: Not sure but this may break MoE. Why do we make 2 peft models?
        self.model.image_encoder = get_peft_model(self.model.image_encoder, peft_config)

        if train_only_vit:
            self.model.mask_decoder.requires_grad_(False)
            self.model.prompt_encoder.requires_grad_(False)
        else:
            self.model.mask_decoder = get_peft_model(
                self.model.mask_decoder, peft_config
            )

    def save_pretrained(self, path: str):
        self.model.image_encoder.save_pretrained(path + "/image_encoder")
        if isinstance(self.model.mask_decoder, PeftModel):
            self.model.mask_decoder.save_pretrained(path + "/mask_decoder")
        super().save_pretrained(path)

    def train(self, mode=True):
        self.model.train(mode)
        self.model.test_mode = not mode
        self.config.test_mode = not mode
        return self
