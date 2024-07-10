from sentence_transformers.models import Transformer
from transformers import AutoModel, T5Config, MT5Config, M2M100Config
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Encoder


class CustomTransformer(Transformer):
    """
    Add support for M2M100Encoder. 
    """
    def _load_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the transformer model"""
        if isinstance(config, T5Config):
            self._load_t5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, MT5Config):
            self._load_mt5_model(model_name_or_path, config, cache_dir, **model_args)
        elif isinstance(config, M2M100Config):
            self._load_m2m_100_model(model_name_or_path, config, cache_dir, **model_args)
        else:
            self.auto_model = AutoModel.from_pretrained(
                model_name_or_path, config=config, cache_dir=cache_dir, **model_args
            )

    def _load_m2m_100_model(self, model_name_or_path, config, cache_dir, **model_args):
        """Loads the encoder model from M2M100Model"""
        M2M100Encoder._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.auto_model = M2M100Encoder.from_pretrained(
            model_name_or_path, config=config, cache_dir=cache_dir, **model_args
        )  
  