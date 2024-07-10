from src.ModularSentenceTransformer import ModularSentenceTransformer
import adapters

def load_adapter_model(model_path, adapter_path=None, max_seq_length=512):
    model = ModularSentenceTransformer(model_name_or_path=model_path)
    model.max_seq_length = max_seq_length

    if adapter_path:
        adapters.init(model[0].auto_model)
        model[0].auto_model.load_adapter(adapter_path)

    return model
