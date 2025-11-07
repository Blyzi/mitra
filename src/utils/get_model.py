from src.utils.gemma3 import Gemma3Model
from src.utils.models import Model


def get_model(name: str) -> Model:
    if "google/gemma-3" in name:
        return Gemma3Model(name)

    raise NotImplementedError(f"Model {name} not implemented")
