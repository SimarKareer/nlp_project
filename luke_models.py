from jsonschema import RefResolutionError
from transformers import LukeForEntitySpanClassification, LukeModel, LukePreTrainedModel, LukeConfig, ReformerConfig, ReformerModel
from transformers.models.luke.modeling_luke import BaseLukeModelOutputWithPooling, EntitySpanClassificationOutput
from typing import Optional, Union, Tuple
import torch
import numpy as np
import torch.nn as nn



class LukeReformer(LukeModel):

    def __init__(self, luke_config: LukeConfig, reformer_config: ReformerConfig, add_pooling_layer: bool = True):
        super().__init__(luke_config, add_pooling_layer)

        self.reformer = ReformerModel(reformer_config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        entity_ids: Optional[torch.LongTensor] = None,
        entity_attention_mask: Optional[torch.FloatTensor] = None,
        entity_token_type_ids: Optional[torch.LongTensor] = None,
        entity_position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseLukeModelOutputWithPooling]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from transformers import LukeTokenizer, LukeModel
        >>> tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-base")
        >>> model = LukeModel.from_pretrained("studio-ousia/luke-base")
        # Compute the contextualized entity representation corresponding to the entity mention "Beyoncé"
        >>> text = "Beyoncé lives in Los Angeles."
        >>> entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"
        >>> encoding = tokenizer(text, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")
        >>> outputs = model(**encoding)
        >>> word_last_hidden_state = outputs.last_hidden_state
        >>> entity_last_hidden_state = outputs.entity_last_hidden_state
        # Input Wikipedia entities to obtain enriched contextualized representations of word tokens
        >>> text = "Beyoncé lives in Los Angeles."
        >>> entities = [
        ...     "Beyoncé",
        ...     "Los Angeles",
        >>> ]  # Wikipedia entity titles corresponding to the entity mentions "Beyoncé" and "Los Angeles"
        >>> entity_spans = [
        ...     (0, 7),
        ...     (17, 28),
        >>> ]  # character-based entity spans corresponding to "Beyoncé" and "Los Angeles"
        >>> encoding = tokenizer(
        ...     text, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt"
        ... )
        >>> outputs = model(**encoding)
        >>> word_last_hidden_state = outputs.last_hidden_state
        >>> entity_last_hidden_state = outputs.entity_last_hidden_state
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if entity_ids is not None:
            entity_seq_length = entity_ids.size(1)
            if entity_attention_mask is None:
                entity_attention_mask = torch.ones((batch_size, entity_seq_length), device=device)
            if entity_token_type_ids is None:
                entity_token_type_ids = torch.zeros((batch_size, entity_seq_length), dtype=torch.long, device=device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # First, compute word embeddings
        word_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Second, compute extended attention mask
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, entity_attention_mask)

        # Third, compute entity embeddings and concatenate with word embeddings
        if entity_ids is None:
            entity_embedding_output = None
        else:
            entity_embedding_output = self.entity_embeddings(entity_ids, entity_position_ids, entity_token_type_ids)


        """
        start of our code
        """
        ########### WORD EMBEDDING OVERRIDE -> ENT EMBEDDING #########

        # entity_position_ids = np.array([[[1, 2, -1], [5, 6, -1]]]) 
        # position_ids=None

        # word_embedding_output=np.zeros((1, 9, 4)) 

        # entity_embedding_output = np.array([[[1, 2, 3, 4], [5, 6, 7, 9]]]) 


        for i in range(entity_embedding_output.shape[0]):  
          for j in range(entity_position_ids[i].shape[0]):
            pos_ids = entity_position_ids[i, j][entity_position_ids[i, j] != -1]
            # print('WORD: ' ,word_embedding_output[i][pos_ids])
            # print('ENT: ', entity_embedding_output[i])
            # print('POS_ID: ', pos_ids)
            word_embedding_output[i][pos_ids] = entity_embedding_output[i, j]


        ########### REFORMER ENCODER ############

        # # Fourth, send embeddings through the model
        # encoder_outputs = self.encoder(
        #     word_embedding_output,
        #     entity_embedding_output,
        #     attention_mask=extended_attention_mask,
        #     head_mask=head_mask,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     return_dict=return_dict,
        # )


        print("Word embedding output shape: ", word_embedding_output.shape)
        print("head_mask shape: ", head_mask)
        print("Attention Mask shape: ", attention_mask.shape)
        print("Output hidden shape: ", output_hidden_states)
        print("Output Attn shape: ", output_attentions)
        encoder_outputs = self.reformer.encoder(
            hidden_states=word_embedding_output,
            head_mask=head_mask,
            attention_mask=attention_mask,
            # num_hashes=num_hashes,
            # past_buckets_states=past_buckets_states,
            # use_cache=use_cache,
            # orig_sequence_length=orig_sequence_length,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        """
        end of our code
        """

        ##########################################

        # Fifth, get the output. LukeModel outputs the same as BertModel, namely sequence_output of shape (batch_size, seq_len, hidden_size)
        sequence_output = encoder_outputs[0]

        # Sixth, we compute the pooled_output, word_sequence_output and entity_sequence_output based on the sequence_output
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseLukeModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            entity_last_hidden_state=encoder_outputs.entity_last_hidden_state,
            entity_hidden_states=encoder_outputs.entity_hidden_states,
        )


class LukeReformerForEntitySpanClassification(LukeForEntitySpanClassification):
    def __init__(self, luke_config, reformer_config):
        super().__init__(luke_config)

        self.luke = LukeReformer(luke_config, reformer_config)

        # Initialize weights and apply final processing
        self.post_init()