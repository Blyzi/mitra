from src.utils.gemma3 import Gemma3Model
from src.utils.qwen3 import Qwen3Model
from src.utils.models import Model
from src.utils.llama3 import Llama3Model


def get_model(name: str) -> Model:
    if "google/gemma-3" in name:
        return Gemma3Model(name)
    if "Qwen/Qwen3" in name:
        return Qwen3Model(name)
    if "meta-llama/Llama-3" in name:
        return Llama3Model(name)

    raise NotImplementedError(f"Model {name} not implemented")
