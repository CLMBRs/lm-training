from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

from .electra import CustomElectraConfig, CustomElectraModel

__all__ = ["CustomElectraConfig", "CustomElectraModel"]

AutoConfig.register("custom_electra", CustomElectraConfig)
AutoModel.register(CustomElectraConfig, CustomElectraModel)
AutoModelForMaskedLM.register(CustomElectraConfig, CustomElectraModel)
