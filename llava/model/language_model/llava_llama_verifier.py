#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llava_llama import LlavaLlamaModel


class VerifierConfig(LlamaConfig):
    model_type = "llava_vision_verifier"

    def __init__(self, alpha_type):
        super(LlamaConfig, self).__init__()
        self.alpha_type = alpha_type


class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scaling = hidden_size ** -0.5
            
    def forward(self, query, key, value):
        
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        
        # 为了数值稳定性，减去attention_scores的最大值 （softmax 上下溢问题）
        # scores = scores - torch.max(scores, dim=-1, keepdim=True)[0]
        
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)

        # import pdb; pdb.set_trace()
        
        return context        

class LlavaLlamaVerifierForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = VerifierConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)
        # self.W = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.cross_attn = CrossAttention(config.hidden_size)
        config.output_hidden_states = True

        if config.alpha_type is None:
            self.alpha = None
        elif config.alpha_type == "scalar":
            if config.freeze_alpha:
                self.alpha = 1.0 # must be float
            else:
                self.alpha = nn.Parameter(torch.tensor(0.1))
        elif config.alpha_type == "vector":
            self.alpha = nn.Parameter(torch.zeros(config.hidden_size))
        elif config.alpha_type == "matrix":
            self.alpha = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.tmp_new_vision_embeds = None
        
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        postive:torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # text input_ids -> model.embed_tokens
        # images -> mm_projector(vision_tower(images))
        # concat text_embedding and image_embedding to inputs_embeds
        
        # import pdb; pdb.set_trace()
        
        input_ids, attention_mask, past_key_values, inputs_embeds, labels, new_vision_embeds, length_group = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, output_vision_embed=True)
        # length_group = (system_len, image_len, user_query_len)
        # new_vision_embeds.shape = [bz, image_len, 4096], image_len = 576
        
        # import pdb; pdb.set_trace()
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]

        # hidden_states = outputs.hidden_states[-2] # penultimate layer
        
        ##############################
        ## Vision Verifier
        ##############################
            
        if input_ids is None: # Training & First inference
            system_len, image_len, user_query_len = length_group
            output_vision_embeds = hidden_states[:, system_len:system_len+image_len, :]
            text_embeds = hidden_states[:, system_len+image_len:, :]
            
            self.tmp_new_vision_embeds = output_vision_embeds
            assert output_vision_embeds.shape[1] == image_len
            
            vision_cross = torch.zeros_like(hidden_states)
            vision_cross[:, system_len+image_len:, :] = self.cross_attn(text_embeds, output_vision_embeds, output_vision_embeds)
            
        
        elif input_ids.shape[1] == 1: # Inference
            assert self.tmp_new_vision_embeds != None
            vision_cross = self.cross_attn(hidden_states, self.tmp_new_vision_embeds, self.tmp_new_vision_embeds)

        if self.alpha is None:
            hidden_states = hidden_states + vision_cross
        elif isinstance(self.alpha, nn.Linear):
            hidden_states = hidden_states + self.alpha(vision_cross)
        elif isinstance(self.alpha, nn.Parameter) or isinstance(self.alpha, float):
            hidden_states = hidden_states + self.alpha * vision_cross


            ##############################
            ## Controller
            ##############################
            # c+εWc
            # if not (postive is None):
            #     postive = postive.unsqueeze(1).unsqueeze(2)
            #     tensor_type = hidden_states.dtype
            #     hidden_states = hidden_states + postive * (self.W(hidden_states))
            #     hidden_states = hidden_states.to(tensor_type)
            # elif self.sigma:
            #     tensor_type = hidden_states.dtype
            #     hidden_states = hidden_states + self.sigma * (self.W(hidden_states))
            #     hidden_states = hidden_states.to(tensor_type)
            
            
        
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None: # Training Stage
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava_vision_verifier", VerifierConfig)
AutoModelForCausalLM.register(VerifierConfig, LlavaLlamaVerifierForCausalLM)