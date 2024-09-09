from typing import Optional
import torch
import transformers
from transformers import ElectraForMaskedLM, ElectraForPreTraining 
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel, AutoModelForMaskedLM
#from transformers.tokenization_utils import PretrainedTokenizerFast
from transformers.models.electra.modeling_electra import ElectraForPreTrainingOutput



class CustomElectraConfig(PretrainedConfig):
    model_type= "custom_electra"

    def __init__(
        self,
        generator_model: transformers.DebertaV2Config,
        discriminator_model: transformers.DebertaV2Config,
        vocab_size: int,
        embedding_sharing: bool= True,
        discriminator_lambda_loss: float= 50.0, 
        pretrained_model_name_or_path= None,
        **kwargs,
    ):  
        print("===== kwargs =====")
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        print(f'pretraineds_model_name_or_path: {pretrained_model_name_or_path}')
        #super().__init__(pretrained_model_name_or_path)
        super().__init__(**kwargs)
        self.vocab_size= vocab_size
        self.generator_config= generator_model
        self.discriminator_config= discriminator_model
        self.embedding_sharing= embedding_sharing
        self.discriminator_lambda_loss= discriminator_lambda_loss
        self._name_or_path= pretrained_model_name_or_path
        
        #self._name_or_path= CustomElectraConfig 
# suggest to subclass PreTrainedModel, this has several advantages to using nn.Module
# you can take advantage of existing functions and ensure it will work with Trainer
# https://discuss.huggingface.co/t/resources-for-using-custom-models-with-trainer/4151
class CustomElectraModel(PreTrainedModel):

    # loss weights are default from paper
    #def __init__(self, config: AutoConfig, tokenizer: PretrainedTokenizerFast=None, lambda_loss=50.0):
    
    config_class = CustomElectraConfig

    def __init__(self, config: CustomElectraConfig):

        super().__init__(config)

        self.lambda_loss = self.config.discriminator_lambda_loss
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)

        # can change this to pass the configs,
        # or pass the generator/discriminators themselves
        #base_gen_config = ElectraConfig.from_pretrained('google/electra-base-generator')
        base_gen_config= AutoConfig.from_pretrained(config.generator_config)
        self.generator: ElectraForMaskedLM = AutoModelForMaskedLM.from_config(base_gen_config)

        #base_disc_config = ElectraConfig.from_pretrained('google/electra-base-discriminator')
        base_disc_config = AutoConfig.from_pretrained(config.discriminator_config)
        self.discriminator: ElectraForPreTraining = AutoModelForMaskedLM.from_config(base_disc_config)

        # we can modify src/train_tokenizer.py to take different tokenizer classes
        # and train it there, pass it here
        #self.tokenizer = tokenizer if tokenizer else PretrainedTokenizerFast.from_pretrained("google/electra-base-generator")

    
    def forward(self, 
                input_ids: Optional[torch.Tensor] = None, 
                attention_mask: Optional[torch.Tensor] = None, 
                token_type_ids: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> ElectraForPreTrainingOutput:
        '''
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Labels indicating whether or not the token was replaced (used for computing loss).
            Values should be in `[0, 1]`: (hf convention, check ElectraForPreTraining.forward())

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.
        '''

        # get generator output
        generator_output = self.generator(input_ids, attention_mask, token_type_ids, labels=labels)
        generator_logits = generator_output.logits[labels, :] # only for selected indices
        
        with torch.no_grad():
            # sample substitution tokens from generator
            generator_tokens = self.sample(generator_logits)

            # substitute new sampled token ids in the place of selected indices
            discriminator_input = input_ids.clone()
            discriminator_input[labels] = generator_tokens
        
        # pass to discriminator
        discriminator_output = self.discriminator(discriminator_input, attention_mask, token_type_ids, labels=labels)

        # loss
        loss = generator_output.loss + discriminator_output.loss * self.lambda_loss #might have to rewrite as loss= generator_output.loss.loss + discriminator_output.loss.loss * self.lambda_loss or it might be that discriminator_output.loss.loss is the only one that needs to be .loss twice
        # must return some output.loss in order to work with trainer
        return ElectraForPreTrainingOutput(
            loss=loss,
            logits=discriminator_output.logits,
            hidden_states=discriminator_output.hidden_states,
            attentions=discriminator_output.attentions,
        )
    

    def sample(self, logits):
        gumbel_output = self.gumbel_dist.sample(logits.shape)
        return (logits.float() + gumbel_output).argmax(dim=-1)


