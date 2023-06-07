import math
from typing import Literal, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput
from tqdm import tqdm


class RNNForLanguageModeling(nn.Module):
    """RNN with optional dropout for use with a language modeling objective."""

    RNN_Type = Literal["RNN", "LSTM", "GRU"]

    def __init__(
        self,
        rnn_type: RNN_Type,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_p: float = 0.0,
        tie_weights: bool = False,
        # output_hidden_states: bool = False  # TODO - maybe?
        use_return_dict: bool = False,
        emb_init_range: float = 0.1,
        recur_init_range: Optional[Union[int, float]] = None,
        lin_init_range: Optional[Union[int, float]] = None,
        embedding_kwargs: dict = {},
        rnn_kwargs: dict = {},
    ):
        """Constructor.

        Args:
            rnn_type:
                Choose from:
                    * RNN
                    * LSTM
                    * GRU
            vocab_size:
                Size of the vocabulary; determines dimensions of embedding and linear
                (output) layers.
            embedding_dim:
                Embedding dimension; determines dimensions of embedding and recurrent
                layers.
            hidden_dim:
                Hidden dimension; determines dimensions of recurrent and linear (output)
                layers.
            num_layers:
                Number of recurrent layers.
            dropout_p:
                Percentage dropout in the recurrent layers and from the recurrent layer
                to the final linear layer.
            tie_weights:
                True if the embedding weights and the final linear layer weights should
                be tied together. Requires that embedding_dim == hidden_dim.
            emb_init_range:
                The range within which to uniformly initialize the embedding layer
                weights, i.e., they will be initialized within the range:
                    [-emb_init_range, emb_init_range]

                Default: 0.1.
            recur_init_range:
                The range within which to uniformly initialize the recurrent layer
                weights, i.e., they will be initialized within the range:
                    [-recur_init_range, recur_init_range]

                Can be int, float, or None. If None, defaults to 1/sqrt(hidden_dim).

                Default: None.
            lin_init_range:
                The range within which to uniformly initialize the final linear layer
                weights, i.e., they will be initialized within the range:
                    [-lin_init_range, lin_init_range]

                Can be int, float, or None. If None, is set equal to recur_init_range.

                Default: None.
            embedding_kwargs:
                Optional keyword arguments to pass to the constructor of the Embedding
                layer.
            rnn_kwargs:
                Optional keyword arguments to pass to the constructor of the recurrent
                layer(s).
        """
        super().__init__()

        assert (
            not tie_weights or embedding_dim == hidden_dim
        ), "Embedding and hidden dimensions are not equal, cannot tie weights."

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.tie_weights = tie_weights
        # self.output_hidden_states = output_hidden_states  # TODO - maybe?
        self.use_return_dict = use_return_dict
        self.emb_init_range = emb_init_range

        if recur_init_range is None:
            self.recur_init_range = 1 / math.sqrt(self.hidden_dim)
        elif isinstance(recur_init_range, float) or isinstance(recur_init_range, int):
            self.recur_init_range = float(recur_init_range)
        else:
            raise ValueError(
                "Invalid value for recur_init_range provided, expected int, float, or "
                f"None but got: {recur_init_range} of type {type(recur_init_range)}"
            )

        if lin_init_range is None:
            self.lin_init_range = self.recur_init_range
        elif isinstance(lin_init_range, float) or isinstance(lin_init_range, int):
            self.lin_init_range = float(lin_init_range)
        else:
            raise ValueError(
                "Invalid value for lin_init_range provided, expected int, float, or "
                f"None but got: {lin_init_range} of type {type(lin_init_range)}"
            )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, **embedding_kwargs
        )

        self.recurrent: nn.RNNBase = getattr(nn, rnn_type)(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout_p,
            batch_first=True,
            **rnn_kwargs,
        )

        # self.recurrent.all_weights

        self.dropout = nn.Dropout(p=self.dropout_p)

        self.lm_head = nn.Linear(
            in_features=self.hidden_dim, out_features=self.vocab_size
        )

        if self.tie_weights:
            self.embedding.weight = self.lm_head.weight

        self.init_weights()

    def init_weights(self):
        # self.emb_init_range = 0.1
        self.embedding.weight.data.uniform_(-self.emb_init_range, self.emb_init_range)

        self.lm_head.weight.data.uniform_(-self.lin_init_range, self.lin_init_range)
        self.lm_head.bias.data.zero_()

        for i in range(self.num_layers):
            self.recurrent.all_weights[i][0] = torch.FloatTensor(
                self.embedding_dim, self.hidden_dim
            ).uniform_(-self.recur_init_range, self.recur_init_range)
            self.recurrent.all_weights[i][1] = torch.FloatTensor(
                self.hidden_dim, self.hidden_dim
            ).uniform_(-self.recur_init_range, self.recur_init_range)

    # def forward(self, inp, h_in):
    # def forward(self, inp):
    # def forward(self, *args, **kwargs):
    #     print("forward args:", args)
    #     print("forward kwargs:", kwargs)
    #     # output, h_out = self.recurrent(self.dropout(self.embedding(inp)))  # , h_in)
    #     # return self.lm_head(self.dropout(output))  # , h_out

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,  # TODO - maybe?
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """

        tqdm.write(f"input_ids: {input_ids}")
        tqdm.write(f"labels: {labels}")
        tqdm.write(f"attention_mask: {attention_mask}")
        if labels is None:
            raise ValueError(
                "Labels need to be provided for autoregressive language modeling"
            )
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        # )

        return_dict = return_dict if return_dict is not None else self.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time."
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        inputs_embeds = self.dropout(inputs_embeds)

        batch_size, seq_length = input_shape
        # required mask seq length can be calculated via length of past
        # mask_seq_length = past_key_values_length + seq_length

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_length, device=inputs_embeds.device
            )
        elif not ((attention_mask == 0) + (attention_mask == 1)).all():
            raise ValueError(
                "Attention mask in an RNN represents paddings, so all values should be "
                "either 0 or 1"
            )
        # embed positions
        # if attention_mask is None:
        #     attention_mask = torch.ones(
        #         batch_size, seq_length, device=inputs_embeds.device
        #     )

        # hidden_states = inputs_embeds  # TODO?

        rnn_out, h_out = self.recurrent(inputs_embeds)  # , h_in)
        # logits = self.lm_head(self.dropout(rnn_out), h_out).contiguous()
        logits = self.lm_head(self.dropout(rnn_out)).contiguous()

        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        tqdm.write(f"Labels: {labels}")
        tqdm.write(f"Labels shape: {labels.shape}")
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # shift_attention_mask = attention_mask[..., 1:].contiguous()
        tqdm.write(f"shift_logits shape: {shift_logits.shape}")
        tqdm.write(f"shift_labels: {shift_labels.shape}")
        tqdm.write(f"shift_labels shape: {shift_labels}")

        loss_fct = nn.CrossEntropyLoss()
        # Flatten the tokens and compute loss
        loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        # loss_fct_masked = nn.CrossEntropyLoss(reduction="none")
        # loss = loss_fct_masked(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
        # masked_loss = loss * shift_attention_mask.view(-1)
        # mean_loss = (1 / shift_attention_mask.sum()) * masked_loss.sum()

        # tqdm.write(f"Mean masked loss: {mean_loss}")
        # tqdm.write(f"Loss: {loss}")
        # assert mean_loss == loss, "Losses different"

        if not return_dict:
            return (loss, logits)
            # return (mean_loss, logits)
            # output = (logits,) + outputs[1:]
            # return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,  # mean_loss,
            logits=logits,
            # hidden_states=outputs.hidden_states,
        )

        # if return_dict:
        #     outputs = BaseModelOutputWithPast(
        #         last_hidden_state=hidden_states,
        #         hidden_states=all_hidden_states,
        #     )
        # else:
        #     outputs = tuple(v for v in [hidden_states, next_cache, all_hidden_states] if v is not None)

        # # decoder layers
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attns = () if output_attentions else None

        # for idx, decoder_layer in enumerate(self.layers):
        #     if self.gradient_checkpointing and self.training:

        #         def create_custom_forward(module):
        #             def custom_forward(*inputs):
        #                 # None for past_key_value
        #                 return module(*inputs, output_attentions, None)

        #             return custom_forward

        #         layer_outputs = torch.utils.checkpoint.checkpoint(
        #             create_custom_forward(decoder_layer),
        #             hidden_states,
        #             head_mask[idx] if head_mask is not None else None,
        #             None,
        #         )
        #     else:
        #         layer_outputs = decoder_layer(
        #             hidden_states,
        #             layer_head_mask=(head_mask[idx] if head_mask is not None else None),
        #             past_key_value=past_key_value,
        #             output_attentions=output_attentions,
        #         )

        #     hidden_states = layer_outputs[0]

        # if self.final_layer_norm is not None:
        #     hidden_states = self.final_layer_norm(hidden_states)

        # if self.project_out is not None:
        #     hidden_states = self.project_out(hidden_states)

        # # add hidden states from the last decoder layer
        # if output_hidden_states:
        #     all_hidden_states += (hidden_states,)
