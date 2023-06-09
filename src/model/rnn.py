"""Implementation of a RNN with a language modeling head.

See class docstring for more info.
"""

import math
from typing import Any, Literal, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutput


class RNNForLanguageModeling(nn.Module):
    """RNN with optional dropout for use with a language modeling objective.

    Consists of three components:
        * Embedding layer
        * Recurrent layer- Simple Elman RNN, LSTM, or GRU
        * Language modeling head- linear layer with bias

    Primarily intended for use in conjunction with a HuggingFace `Trainer` instance, or
    at least with a `transformers.tokenization_utils.PreTrainedTokenizer` instance.
    See the method documentation of `forward` for more information on how to use this
    without a `PreTrainedTokenizer` or `Trainer`.
    """

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
        # output_hidden_states: bool = False  # TODO?
        use_return_dict: bool = False,
        emb_init_range: float = 0.1,
        recur_init_range: Optional[Union[int, float]] = None,
        lin_init_range: Optional[Union[int, float]] = None,
        embedding_kwargs: dict[str, Any] = {},
        rnn_kwargs: dict[str, Any] = {},
    ):
        """Constructor.

        Args:
            rnn_type ("RNN", "LSTM", or "GRU"):
                Determines the model architecture to use in the recurrent component of
                the model.
                Choose from:
                    * RNN - recurrent layers are a `nn.RNN` instance
                        Simple Elmann RNN with a tanh non-linearity by default. To use a
                        RELU non-linearity instead, include `"nonlinearity": "relu"` as
                        part of the `rnn_kwargs` argument to this constructor.
                    * LSTM - recurrent layers are a `nn.LSTM` instance.
                    * GRU - recurrent layers are a `nn.GRU` instance.

                To customize these further, see `rnn_kwargs`.
            vocab_size (`int`):
                Size of the vocabulary; determines dimensions of embedding and linear
                (output) layers.
            embedding_dim (`int`):
                Embedding dimension; determines dimensions of embedding and recurrent
                layers.
            hidden_dim (`int`):
                Hidden dimension; determines dimensions of recurrent and linear (output)
                layers.
            num_layers (`int`):
                Number of recurrent layers.
            dropout_p (`int`):
                Percentage dropout in the recurrent layers and from the recurrent layer
                to the final linear layer.

                default: 0.0
            tie_weights (`bool`, *optional*):
                `True` if the embedding weights and the final linear layer weights
                should be tied together. Requires that `embedding_dim == hidden_dim`.

                default: `False`
            use_return_dict (`bool`, *optional*):
                Whether or not `forward` should return a
                `transformers.modeling_outputs.ModelOutput` instance instead of a plain
                tuple. Can be overriden by the `return_dict` parameter of the `forward`
                function call itself, if provided.

                Default: `False`.
            emb_init_range (`float`, *optional*):
                The range within which to uniformly initialize the embedding layer
                weights, i.e., they will be initialized within the range:
                    [-emb_init_range, emb_init_range]

                If `tie_weights` is `True`, this value is effectively ignored and the
                embedding layer's weights are instead initialized to the same as the
                final linear layer's weights -- see `lin_init_range`.

                Default: 0.1.
            recur_init_range (`float`, *optional*):
                The range within which to uniformly initialize the recurrent layer
                weights, i.e., they will be initialized within the range:
                    [-recur_init_range, recur_init_range]

                Can be int, float, or None. If None, defaults to `1/sqrt(hidden_dim)`.

                Default: None.
            lin_init_range (`float`, *optional*):
                The range within which to uniformly initialize the final linear layer
                weights, i.e., they will be initialized within the range:
                    [-lin_init_range, lin_init_range]

                Can be int, float, or None. If None, is set equal to recur_init_range.

                Default: None.
            embedding_kwargs (`dict`, *optional*):
                Optional keyword arguments to pass to the constructor of the Embedding
                layer. May contain any keyword arguments accepted by the constructor of
                `torch.nn.Embedding`, except for these duplicates of other arguments to
                this constructor:
                    * num_embeddings(see: vocab_size)
                    * embedding_dim (see: embedding_dim)
                Use this to further customize the embedding layer. For example, to
                freeze the embedding associated with a padding token with index 0 to the
                value 0, include the following key-value pair in this argument:
                    "padding_idx": 0

                Default: {}
            rnn_kwargs (`dict`, *optional*):
                Optional keyword arguments to pass to the constructor of the recurrent
                layer(s). May contain any keyword arguments accepted by the constructor
                the `torch.nn` module associated with the `rnn_type` constructor, except
                for these duplicates of other arguments to this constructor:
                    * input_size (see: vocab_size)
                    * hidden_size (see: hidden_size)
                    * num_layers (see: num_layers)
                    * dropout (see: dropout_p)
                    * batch_first (see: batch_first)
                Use this to further customize the RNN layers. For example, to make the RNN
                bidirectional, include the following key-value pair in this argument:
                    "bidirectional": True

                Default: {}
        Raises:
            ValueError: If `tie_weights` is True but `embedding_dim` and `hidden_dim`
                are not equal or if `recur_init_range`/`lin_init_range` are of an
                invalid type.
        """
        super().__init__()

        if tie_weights and embedding_dim != hidden_dim:
            raise ValueError(
                "Embedding and hidden dimensions are not equal, cannot tie weights."
            )

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.tie_weights = tie_weights
        # self.output_hidden_states = output_hidden_states  # TODO?
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

        self.dropout = nn.Dropout(p=self.dropout_p)

        self.lm_head = nn.Linear(
            in_features=self.hidden_dim, out_features=self.vocab_size
        )

        if self.tie_weights:
            self.embedding.weight = self.lm_head.weight

        self.init_weights()

    def init_weights(self):
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # output_hidden_states: Optional[bool] = None,  # TODO?
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[tuple, CausalLMOutput]:
        """Forward pass from input to output.

        Primarily intended to be used in conjunction with the HuggingFace ecosystem. If
        used without `PreTrainedTokenizer` or `Trainer`, care should be taken when
        constructing the arguments `input_ids`, `labels`, and `return_dict`. See the
        documentation for those parameters below for more information.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding tokens will
                be ignored in the loss computation by default should you provide it.

                Indices can be obtained using `AutoTokenizer`/`PreTrainedTokenizer` from
                the `transformers` package. See the HuggingFace docs on
                `PreTrainedTokenizer.encode`and `PreTrainedTokenizer.__call__` for
                details.

                They can also be generated manually or any other way, if not using
                `PreTrainedTokenizer`. In this case, ensure that all elements of
                `input_ids` are in the range `[0, ..., vocab_size]`.

                Either `input_ids` OR `inputs_embeds` must be provided, but not both.

                [What are input
                IDs?](https://huggingface.co/docs/transformers/v4.18.0/en/glossary#input-ids)
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly
                pass an embedded representation.  This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.

                If provided, inputs_embeds is still subject to dropout.

                Either `input_ids` OR `inputs_embeds` must be provided, but not both.

            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling (cross-entropy) loss.
                Indices should either be in `[0, ..., config.vocab_size]` or -100
                (masked) (see `input_ids` docstring).

                The loss is only computed for the tokens with labels in the range
                `[0, ..., config.vocab_size]`. Tokens with indices set to `-100` are
                ignored (they generally correspond to the padding token).

                If using this class in conjunction with the `Trainer` class from the
                `transformers` library, padding tokens will automatically be replaced
                wih -100. If not using Trainer, care should be taken to do this
                replacement manually.

                Dev Notes:
                    Parameter is currently optional in the function signature, but its
                    absence results in a ValueError being thrown. We may want to support
                    the use case where the user only wants logits, and thus no labels
                    argument is provided.
            return_dict (`bool`, *optional*):
                Whether or not to return a `transformers.modeling_outputs.ModelOutput`
                instead of a plain tuple. If not provided, defaults to the value of the
                model's `use_return_dict` field. If provided, overrides the value of the
                model's `use_return_dict` field.

            Possible TODO:
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states for all time steps.
                Dev Notes:
                    HF provides this option for its own transformer models (for each
                    hidden layer rather than for each time step). It may be nice to have
                    this option for certain use cases e.g. probes?
        Returns:
            A `transformers.modeling_outputs.CausalLMOutput` instance if `return_dict`
            is True, containing the keys `loss` and `logits` with the corresponding
            values. If `return_dict` is False, returns a tuple containing the loss and
            the logits, in that order. Types:
                loss: `float`
                logits: `torch.Tensor` of shape (batch_size, seq_length, vocab_size)
        Raises:
            ValueError:
                * If `labels` is not provided.
                * If both `input_ids` and `inputs_embeds` are provided, or if neither
                are provided.
        """
        # TODO: Do we want this check? Maybe someone just wants the logits
        if labels is None:
            raise ValueError(
                "Labels need to be provided for autoregressive language modeling."
            )
        # output_hidden_states = (  # TODO?
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

        # hidden_states = inputs_embeds  # TODO?

        rnn_out, h_out = self.recurrent(inputs_embeds)
        logits = self.lm_head(self.dropout(rnn_out)).contiguous()

        # move labels to correct device to enable model parallelism
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        # Flatten the tokens and compute loss
        loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        if return_dict:
            return CausalLMOutput(
                loss=loss,
                logits=logits,
                # hidden_states=outputs.hidden_states,
            )
        else:
            return (loss, logits)
