from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .rnn import RNNConfig, RNNForLanguageModeling

__all__ = ["RNNConfig", "RNNForLanguageModeling"]

AutoConfig.register("rnn", RNNConfig)
AutoModel.register(RNNConfig, RNNForLanguageModeling)
AutoModelForCausalLM.register(RNNConfig, RNNForLanguageModeling)
