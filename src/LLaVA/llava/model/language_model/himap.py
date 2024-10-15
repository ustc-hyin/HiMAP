import torch
from typing import Tuple
from transformers.utils import logging
from .himap_modeling_llama import LlamaModel
from .himap_configuration_llama import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union

logger = logging.get_logger(__name__)

class Himap_LlamaModel(LlamaModel):

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        self.config = config
        # hmapv hyperparameter
        self.hmap_v_sys_length = config.hmap_v_sys_length
        self.hmap_v_img_length = config.hmap_v_img_length
        self.hmap_v_attn_txt_layer = config.hmap_v_attn_txt_layer
        self.hmap_v_attn_txt_rank = config.hmap_v_attn_txt_rank
        self.hmap_v_attn_img_layer = config.hmap_v_attn_img_layer
        self.hmap_v_attn_img_rank = config.hmap_v_attn_img_rank
        self.use_hmap_v = config.use_hmap_v 

    def reset_hmapv(self):
        self.hmap_v_sys_length = self.config.hmap_v_sys_length
        self.hmap_v_img_length = self.config.hmap_v_img_length
        self.hmap_v_attn_txt_layer = self.config.hmap_v_attn_txt_layer
        self.hmap_v_attn_txt_rank = self.config.hmap_v_attn_txt_rank
        self.hmap_v_attn_img_layer = self.config.hmap_v_attn_img_layer
        self.hmap_v_attn_img_rank = self.config.hmap_v_attn_img_rank
        self.use_hmap_v = self.config.use_hmap_v  

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                USE_HMAP_V = self.use_hmap_v
                SYS_LENGTH = self.hmap_v_sys_length
                IMG_LENGTH = self.hmap_v_img_length
                TXT_LAYER = self.hmap_v_attn_txt_layer
                TXT_ATTN_RANK = self.hmap_v_attn_txt_rank
                IMG_LAYER = self.hmap_v_attn_img_layer
                IMG_ATTN_RANK = self.hmap_v_attn_img_rank

                if TXT_LAYER:
                    assert TXT_LAYER > 0, "txt attn layer should be larger than 0"
                if IMG_LAYER:
                    assert IMG_LAYER > TXT_LAYER, "img attn layer should be larger than txt attn layer"
                if TXT_ATTN_RANK and IMG_ATTN_RANK:
                    assert TXT_ATTN_RANK >= IMG_ATTN_RANK, "txt attn rank should be larger than img attn rank"

                # IMAGE TOKEN PRUNING BEGIN --HiMAP TECHNIQUE 
                if USE_HMAP_V:
                    
                    # Before image tokens pruning
                    if idx < TXT_LAYER:
                        new_attention_mask = attention_mask

                    # image token pruning according to img2txt information
                    elif idx == TXT_LAYER:
                        # compute the img2txt attention score
                        txt_layer_attn = layer_outputs[1]
                        txt_layer_attn_avg = torch.mean(txt_layer_attn, dim=1)[0]                        
                        img2txt_attn = torch.sum(
                            txt_layer_attn_avg[SYS_LENGTH+IMG_LENGTH:, SYS_LENGTH:SYS_LENGTH+IMG_LENGTH], dim=0
                        )
                        # get the indexs of selected image tokens
                        img2txt_attn_topk_index = img2txt_attn.topk(TXT_ATTN_RANK).indices + SYS_LENGTH
                        txt_keep_indexs = torch.cat(
                            (
                                torch.arange(SYS_LENGTH, device=device),
                                img2txt_attn_topk_index,
                                torch.arange(SYS_LENGTH+IMG_LENGTH, seq_length_with_past, device=device)
                            )
                        )
                        txt_keep_indexs = txt_keep_indexs.sort().values
                        # update the hidden states, position ids and attention mask
                        txt_seq_length = txt_keep_indexs.shape[0]
                        hidden_states = hidden_states[:, txt_keep_indexs, :]
                        position_ids = txt_keep_indexs.unsqueeze(0)
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            None, (batch_size, txt_seq_length), inputs_embeds, 0
                        )                        

                    # image token pruning according to img2img information
                    elif idx == IMG_LAYER:
                        # compute the img2img attention score
                        img_layer_attn = layer_outputs[1]
                        img_layer_attn_avg = torch.mean(img_layer_attn, dim=1)[0]
                        img2img_attn = torch.sum(
                            img_layer_attn_avg[SYS_LENGTH:SYS_LENGTH+TXT_ATTN_RANK, SYS_LENGTH:SYS_LENGTH+TXT_ATTN_RANK], dim=0
                        )
                        # img2img_attn = torch.sum(
                        #     img_layer_attn_avg[SYS_LENGTH+TXT_ATTN_RANK:, SYS_LENGTH:SYS_LENGTH+TXT_ATTN_RANK], dim=0
                        # )
                        # get the indexs of selected image tokens
                        img2img_attn_topk_index = img2img_attn.topk(IMG_ATTN_RANK).indices + SYS_LENGTH
                        img_keep_indexs = torch.cat(
                            (
                                torch.arange(SYS_LENGTH, device=device),
                                img2img_attn_topk_index,
                                torch.arange(SYS_LENGTH+TXT_ATTN_RANK, txt_seq_length, device=device)
                            )
                        ) 
                        img_keep_indexs = img_keep_indexs.sort().values
                        # update the hidden states, position ids and attention mask
                        img_seq_length = img_keep_indexs.shape[0]  
                        hidden_states = hidden_states[:, img_keep_indexs, :]
                        position_ids = txt_keep_indexs[img_keep_indexs].unsqueeze(0)
                        new_attention_mask = self._prepare_decoder_attention_mask(
                            None, (batch_size, img_seq_length), inputs_embeds, 0
                        ) 

                else: 
                    new_attention_mask = attention_mask
                    
                # print(idx, hidden_states.shape, new_attention_mask.shape, position_ids.shape)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=new_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            # change the code to make llama model will not save attention scores
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                # all_self_attns = None

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
