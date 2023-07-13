"""Implementation of a RNN with a language modeling head + related classes.

See class and function docstrings for more info.
"""

from enum import auto
from dataclasses import dataclass
import math
from typing import Any, Optional, Union

from strenum import UppercaseStrEnum
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput


class RNNType(UppercaseStrEnum):
    """Enum to store possible architectures for the recurrent layers of a
    `RNNForLanguageModeling` instance. Each member's value corresponds to a class in
    the `torch.nn` package that subclasses from `torch.nn.RNNBase`.
    """

    RNN = auto()
    LSTM = auto()
    GRU = auto()

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid RNN architecture type, please select one of: "
            f"{list(cls._value2member_map_.keys())}"
        )


@dataclass
class CausalRNNModelOutput(ModelOutput):
    """Base class for recurrent causal language model (or autoregressive) outputs.

    Args:
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary
            token before SoftMax).
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss (for next-token prediction).

            Returned when `labels` is provided.
        recurrent_outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, D*hidden_size)` where D=2 if bidirectional, otherwise D=1, *optional*):
            Final recurrent layer outputs of the model at each time step.

            Returned when `output_recurrent_outputs=True` is passed or when the model
            attribute `output_recurrent_outputs=True`.
        last_state (Tuple of one or two `torch.FloatTensor`'s of shape `(D*num_layers, batch_size, hidden_size)` where D=2 if bidirectional, otherwise D=1, *optional*):
            For vanilla RNN's/GRU's, a tuple containing one tensor, the final
            hidden-state of the model for each input sequence in the batch. For LSTM's,
            a tuple of two tensors, the first containing the final hidden-state and the
            second the final cell-state for each input sequence in the batch.

            Returned when `output_last_state=True` is passed or when the model
            attribute `output_last_state=True`.
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    recurrent_outputs: Optional[torch.Tensor] = None
    last_state: Optional[tuple[torch.Tensor]] = None


class RNNForLanguageModeling(nn.Module):
    """RNN with optional dropout for use with a language modeling objective.

    Consists of three components:
        * Embedding layer
        * Recurrent layer- choice of vanilla (Elman) RNN, LSTM, or GRU
        * Language modeling head- linear layer with bias

    Primarily intended for use in conjunction with a HuggingFace `Trainer` instance, or
    at least with a `transformers.tokenization_utils.PreTrainedTokenizer` instance.
    See the method documentation of `forward` for more information on how to use this
    without a `PreTrainedTokenizer` or `Trainer`.
    """

    def __init__(
        self,
        rnn_type: RNNType,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout_p: float = 0.0,
        tie_weights: bool = False,
        bidirectional: bool = False,
        emb_init_range: float = 0.1,
        recur_init_range: Optional[float] = None,
        lin_init_range: Optional[float] = None,
        output_recurrent_outputs: bool = False,
        output_last_state: bool = False,
        embedding_kwargs: dict[str, Any] = {},
        rnn_kwargs: dict[str, Any] = {},
    ):
        """Constructor.

        Args:
            rnn_type ("RNN", "LSTM", or "GRU"):
                Determines the model architecture to use in the recurrent component of
                the model. Choose from:
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
                layers. For bidirectional RNN's, the linear layer's input feature
                dimension is twice the `hidden_dim`; otherwise, they are equal.
            num_layers (`int`):
                Number of recurrent layers.
            dropout_p (`int`, *optional*):
                Percentage dropout in the recurrent layers and from the recurrent layer
                to the final linear layer.

                Default: 0.0.
            tie_weights (`bool`, *optional*):
                `True` if the embedding weights and the final linear layer weights
                should be tied together. Requires that `embedding_dim=hidden_dim` for
                unidirectional RNNs or that `embedding_dim=2*hidden_dim` for
                bidirectional RNNs.

                Default: `False`.
            bidirectional (`bool`, *optional*):
                `True` if recurrent layers should be bidirectional.

                Default: `False`.
            emb_init_range (`float`, *optional*):
                The range within which to uniformly initialize the embedding layer
                weights, i.e., they will be initialized within the range:
                    `[-emb_init_range, emb_init_range]`

                If `tie_weights=True`, this value is effectively ignored and the
                embedding layer's weights are instead initialized to the same as the
                final linear layer's weights -- see `lin_init_range`.

                Default: 0.1.
            recur_init_range (`float`, *optional*):
                The range within which to uniformly initialize the recurrent layer
                weights, i.e., they will be initialized within the range:
                    `[-recur_init_range, recur_init_range]`

                Can be a real number or None. If None, defaults to `1/sqrt(hidden_dim)`.

                Default: None.
            lin_init_range (`float`, *optional*):
                The range within which to uniformly initialize the final linear layer
                weights, i.e., they will be initialized within the range:
                    `[-lin_init_range, lin_init_range]`

                Can be a real number or `None`. If None, is set equal to
                `recur_init_range`.

                Default: None.
            output_recurrent_outputs (`bool`, *optional*):
                Whether or not to return the final recurrent layer outputs of the model
                at each time step. See the return section of the `forward` function
                docstring for more detail.

                Default: `False`.
            output_last_state (`bool`, *optional*):
                Whether or not to return the final hidden-state (+ final cell-state for
                LSTMs) of the recurrent layers for each input sequence in the input
                batch. See the return section of the `forward` function docstring for
                more detail.

                Default: `False`.
            embedding_kwargs (`dict`, *optional*):
                Optional keyword arguments to pass to the constructor of the Embedding
                layer. May contain any keyword arguments accepted by the constructor of
                `torch.nn.Embedding`, except for these duplicates of other arguments to
                this constructor:
                    * num_embeddings (see: vocab_size)
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
                    * `input_size` (see: `vocab_size`)
                    * `hidden_size` (see: `hidden_size`)
                    * `num_layers` (see: `num_layers`)
                    * `batch_first` (this is set to `True` for implementation reasons)
                        Use this to further customize the RNN layers. For example, to
                        construct a vanilla RNN that uses a RELU non-linearity, include
                        the following key-value pair in this argument:
                            "nonlinearity": "relu"
                    * `dropout` (see: `dropout_p`)
                    * `bidirectional` (see: `bidirectional`)

                Default: {}
        Raises:
            TypeError:
                If `embedding_kwargs` or `rnn_kwargs` contain duplicates of other
                arguments to this constructor, as explained above.
            ValueError:
                * If `tie_weights=True` but `embedding_dim!=hidden_dim` for
                unidirectional RNNs or `embedding_dim!=2*hidden_dim` for bidirectional
                RNNs
                * If `recur_init_range`/`lin_init_range` are of an
                invalid type.
                * If `rnn_type` is not a valid `RNNType` or string that can be converted
                to a valid `RNNType`: "RNN", "LSTM," or "GRU".
        """
        super().__init__()

        # double linear layer input dimensions if bidirectional
        lm_in_features = hidden_dim * (1 + bidirectional)
        if tie_weights and (embedding_dim != lm_in_features) :
            raise ValueError(
                f"embedding_dim is not {'double' if bidirectional else 'equal to'} "
                "hidden_dim, cannot tie weights."
            )
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lm_in_features = lm_in_features
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.tie_weights = tie_weights
        self.bidirectional = bidirectional
        self.emb_init_range = emb_init_range

        if recur_init_range is None:
            self.recur_init_range = 1 / math.sqrt(self.hidden_dim)
        elif isinstance(recur_init_range, float):
            self.recur_init_range = float(recur_init_range)
        else:
            raise ValueError(
                "Invalid value for recur_init_range provided, expected float or None "
                f"but got: {recur_init_range} of type {type(recur_init_range)}"
            )

        if lin_init_range is None:
            self.lin_init_range = self.recur_init_range
        elif isinstance(lin_init_range, float):
            self.lin_init_range = float(lin_init_range)
        else:
            raise ValueError(
                "Invalid value for lin_init_range provided, expected float or None "
                f"but got: {lin_init_range} of type {type(lin_init_range)}"
            )

        self.output_recurrent_outputs = output_recurrent_outputs
        self.output_last_state = output_last_state

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, **embedding_kwargs
        )

        self.recurrent: nn.RNNBase = getattr(nn, RNNType(rnn_type.upper()))(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout_p,
            bidirectional=self.bidirectional,
            **rnn_kwargs,
        )

        self.dropout = nn.Dropout(p=self.dropout_p)

        self.lm_head = nn.Linear(
            in_features=self.lm_in_features,
            out_features=self.vocab_size,
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
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_recurrent_outputs: Optional[bool] = None,
        output_last_state: Optional[bool] = None,
        **kwargs,
    ) -> CausalRNNModelOutput:
        """Forward pass from input to output.

        Primarily intended to be used in conjunction with the HuggingFace ecosystem. If
        used without `PreTrainedTokenizer` or `Trainer`, care should be taken when
        constructing the arguments `input_ids`, and `labels`. See the documentation for
        those parameters below for more information.

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
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_dim)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly
                pass an embedded representation.  This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.

                If provided, `inputs_embeds` is still subject to dropout.

                Either `input_ids` OR `inputs_embeds` must be provided, but not both.

            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling (cross-entropy) loss.
                Indices should either be in `[0, ..., vocab_size)` or -100
                (masked) (see `input_ids` docstring).

                The loss is only computed for the tokens with labels in the range
                `[0, ..., vocab_size)`. Tokens with indices set to `-100` are
                ignored (they generally correspond to the padding token).

                If using this class in conjunction with the `Trainer` class from the
                `transformers` library, padding tokens will automatically be replaced
                wih -100. If not using Trainer, care should be taken to do this
                replacement manually.
            output_recurrent_outputs (`bool`, *optional*):
                Whether or not to return the final recurrent layer outputs of the model
                at each time step. See the return section of the `forward` function
                docstring for more detail.

                default: `False`
            output_last_state (`bool`, *optional*):
                Whether or not to return the final hidden-state (+ final cell-state for
                LSTMs) of the recurrent layers for each input sequence in the input
                batch. See the return section of the `forward` function docstring for
                more detail.

                default: `False`
        Returns:
            A `CausalRNNModelOutput` instance containing the keys `logits`, `loss`,
            `recurrent_outputs`, and `last_state` with the corresponding values.

            See the `CausalRNNModelOutput` docstring for more info.
        Raises:
            ValueError: if both `input_ids` and `inputs_embeds` are provided, or if
                neither are provided.
        """
        output_recurrent_outputs = (
            output_recurrent_outputs
            if output_recurrent_outputs is not None
            else self.output_recurrent_outputs
        )

        output_last_state = (
            output_last_state
            if output_last_state is not None
            else self.output_last_state
        )

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

        # inputs_embeds: (batch_size, sequence_length, embedding_dim)
        inputs_embeds = self.dropout(inputs_embeds)

        # rnn_out: (batch_size, sequence_length, D*hidden_dim)
        # h_out: (D*num_layers, batch_size, hidden_dim)
        # D=2 if bidirectional else 1
        rnn_out, h_out = self.recurrent(inputs_embeds)

        # logits: (batch_size, sequence_length, vocab_size)
        logits = self.lm_head(self.dropout(rnn_out)).contiguous()

        loss: Optional[torch.Tensor] = None

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)

            # Shift so that tokens < n predict n- HF `DataCollatorForLanguageModeling`
            # does not do this automatically, see:
            # https://huggingface.co/learn/nlp-course/en/chapter7/6?fw=pt#initializing-a-new-model
            # shift_logits: (batch_size, sequence_length-1, vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels: (batch_size, sequence_length-1)
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens and compute loss
            # reshape inputs (logits) and targets (labels) to
            # (batch_size*sequence_length-1, vocab_size)
            loss = F.cross_entropy(  # shape: (1,)
                shift_logits.view(-1, self.vocab_size), shift_labels.view(-1)
            )

        return CausalRNNModelOutput(
            logits=logits,
            loss=loss,
            recurrent_outputs=rnn_out if output_recurrent_outputs else None,
            last_state=tuple(h_out) if output_last_state else None,
        )
