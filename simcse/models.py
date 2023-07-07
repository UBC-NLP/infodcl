import torch, json
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
from transformers import RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaForSequenceClassification
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.activations import gelu
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import logging
from pytorch_metric_learning import losses

from .loss import Soft_SupConLoss_PMI, Soft_SupConLoss_CLS

logger = logging.get_logger(__name__)

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)

        return x

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError

def cl_init(cls, config):
    """
    Contrastive learning class init function.
    """
    cls.pooler_type = cls.model_args.pooler_type
    cls.pooler = Pooler(cls.model_args.pooler_type)
    if cls.model_args.pooler_type == "cls":
        cls.mlp = MLPLayer(config)
    cls.sim = Similarity(temp=cls.model_args.temp)
    cls.init_weights()

def cl_forward(cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    mlm_input_ids=None,
    mlm_labels=None,
):  
    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict
    ori_input_ids = input_ids
    batch_size = input_ids.size(0)
    # Number of sentences in one instance
    # 2: pair instance; 3: pair instance with a hard negative
    num_sent = input_ids.size(1)

    # supervised labels
    if labels is not None:
        supervised_labels = labels
        if len(supervised_labels.shape) == 1:  # copy label
            supervised_labels = supervised_labels.unsqueeze(1)
            supervised_labels = torch.cat([supervised_labels] * num_sent, 1)
            supervised_labels = supervised_labels.view(-1)  # size (bs * num_sent)

    mlm_outputs = None

    # Flatten input for encoding
    input_ids = input_ids.view((-1, input_ids.size(-1))) # (bs * num_sent, len)
    attention_mask = attention_mask.view((-1, attention_mask.size(-1))) # (bs * num_sent len)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1))) # (bs * num_sent, len)

    # Get raw embeddings
    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    # MLM auxiliary objective
    if mlm_input_ids is not None:
        mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
        mlm_outputs = encoder(
            mlm_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True if cls.model_args.pooler_type in ['avg_top2', 'avg_first_last'] else False,
            return_dict=True,
        )

    # Pooling
    pooler_output = cls.pooler(attention_mask, outputs)
    # If using "cls", we add an extra MLP layer
    # (same as BERT's original implementation) over the representation.
    if cls.pooler_type == "cls":
        pooler_output = cls.mlp(pooler_output)

    pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1))) # (bs, num_sent, hidden)
    
    if cls.model_args.do_cls or cls.model_args.do_soft_supervise_cls:
        # logger.info("***** Supervise Cross Entropy loss *****")
        loss_fct = nn.CrossEntropyLoss()
        pooler_output_cls = pooler_output.view((-1, pooler_output.size(-1)))
        x = cls.cls_dropout(pooler_output_cls)
        logit = cls.cls_output(x)

        if num_sent == 2 and cls.model_args.do_supervise == False:  # only take one sentence 
            logit_cls = logit.view((batch_size, num_sent, -1))[:,0]
            cls_labels = supervised_labels.view((batch_size, num_sent))[:,0]

        elif num_sent == 2 and cls.model_args.do_supervise == True: # take two sentences
            logit_cls = logit
            cls_labels = supervised_labels

        elif num_sent == 4 and cls.model_args.do_supervise == True:  # take two sentences from four.
            logit_cls = logit.view((batch_size, num_sent, -1))[:,[0,2]].view((batch_size * 2, -1))
            cls_labels = supervised_labels.view((batch_size, num_sent, 1))[:,[0,2]].view((-1))

        if cls.model_args.do_cls:
            cls_loss = loss_fct(logit_cls, cls_labels)
            loss = cls.model_args.cls_weight * cls_loss

    if cls.model_args.dual_training_cls:
        # logger.info("***** Supervise CLS weight *****")
        loss_weight_cls, logit = cls.cls_weight_network(input_ids, attention_mask, labels=supervised_labels)[:2]
        try:
            loss += cls.model_args.cls_weight * loss_weight_cls
        except:
            loss = cls.model_args.cls_weight * loss_weight_cls

    if cls.model_args.do_supervise:
        # logger.info("***** Supervise CL loss *****")
        if cls.model_args.do_soft_supervise_pmi and cls.model_args.do_soft_supervise_cls:
            loss_supercl_pmi = Soft_SupConLoss_PMI(weights = cls.supercl_weights, num_classes = cls.label_size, temperature=cls.model_args.temp, device=cls.device)
            loss_supercl_cls = Soft_SupConLoss_CLS(num_classes = cls.label_size, temperature=cls.model_args.temp, device=cls.device)
        elif cls.model_args.do_soft_supervise_pmi:
            # logger.info("***** Soft Supervise CL loss *****")
            loss_supercl = Soft_SupConLoss_PMI(weights = cls.supercl_weights, num_classes = cls.label_size, temperature=cls.model_args.temp, device=cls.device)
        elif cls.model_args.do_soft_supervise_cls:
            # logger.info("***** Soft Supervise CL loss with CLS weight*****")
            loss_supercl = Soft_SupConLoss_CLS(num_classes = cls.label_size, temperature=cls.model_args.temp, device=cls.device)
        else:
            loss_supercl = losses.SupConLoss(temperature=cls.model_args.temp)
        
        pooler_output_sup = pooler_output.view((-1, pooler_output.size(-1))) #(bs * num_sent, hidden)

        # Gather all embeddings and labels if using distributed training
        if dist.is_initialized() and cls.training:
            # Dummy vectors for allgather
            pooler_output_list = [torch.zeros_like(pooler_output_sup) for _ in range(dist.get_world_size())]
            supervised_labels_list = [torch.zeros_like(supervised_labels) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=pooler_output_list, tensor=pooler_output_sup.contiguous())
            dist.all_gather(tensor_list=supervised_labels_list, tensor=supervised_labels.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            pooler_output_list[dist.get_rank()] = pooler_output_sup
            pooler_output_sup = torch.cat(pooler_output_list, 0)

            supervised_labels_all = torch.cat(supervised_labels_list, 0)

            if cls.model_args.do_soft_supervise_cls:
                logit_list = [torch.zeros_like(logit) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=logit_list, tensor=logit.contiguous())

                logit_all = torch.cat(logit_list, 0)
                
        if cls.model_args.do_soft_supervise_pmi and cls.model_args.do_soft_supervise_cls:
            cl_loss_pmi = loss_supercl_pmi(pooler_output_sup, supervised_labels_all) * (1.0 - cls.model_args.cls_weight_scale)
            # print("pmi loss", cl_loss_pmi)
            cl_loss_cls = loss_supercl_cls(pooler_output_sup, supervised_labels_all, logit_all) * cls.model_args.cls_weight_scale
            cl_loss = cl_loss_pmi + cl_loss_cls
            # print("cls loss", cl_loss_cls) 

        elif cls.model_args.do_soft_supervise_cls:
            cl_loss = loss_supercl(pooler_output_sup, supervised_labels_all, logit_all) 
        else:
            cl_loss = loss_supercl(pooler_output_sup, supervised_labels_all)

        try:
            loss += cl_loss * cls.model_args.contrastive_weight
        except NameError:
            loss = cl_loss

        cos_sim = None

    if cls.model_args.do_selfsupervise:
        # logger.info("***** Self-Supervise CL loss, SimCSE *****")
        # Separate representation
        if num_sent == 2:
            z1, z2 = pooler_output[:,0], pooler_output[:,1]

        if num_sent == 4:
            z1, z2, z3, z4 = pooler_output[:,0], pooler_output[:,1], pooler_output[:,2], pooler_output[:,3]
            z1 = torch.cat([z1, z3], 0)
            z2 = torch.cat([z2, z4], 0)

        # Hard negative
        if num_sent == 3:
            z3 = pooler_output[:, 2]

        # Gather all embeddings if using distributed training
        if dist.is_initialized() and cls.training:
            # Gather hard negative
            if num_sent >= 3:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)
        cos_sim = cls.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # Hard negative
        if num_sent >= 3:
            z1_z3_cos = cls.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(cls.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        if num_sent == 3:
            # Note that weights are actually logits of weights
            z3_weight = cls.model_args.hard_negative_weight
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(cls.device)
            cos_sim = cos_sim + weights

        cl_loss = loss_fct(cos_sim, labels) * cls.model_args.contrastive_weight

        try:
            loss += cl_loss
        except NameError:
            loss = cl_loss


    # Calculate loss for MLM
    if mlm_outputs is not None and mlm_labels is not None:
        # logger.info("***** MLM loss *****")
        loss_fct = nn.CrossEntropyLoss()
        mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
        prediction_scores = cls.lm_head(mlm_outputs.last_hidden_state)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, cls.config.vocab_size), mlm_labels.view(-1))
        loss = loss + cls.model_args.mlm_weight * masked_lm_loss

    if not return_dict:
        output = (cos_sim,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output
    
    return SequenceClassifierOutput(
        loss=loss,
        logits=cos_sim,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def sentemb_forward(
    cls,
    encoder,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):

    return_dict = return_dict if return_dict is not None else cls.config.use_return_dict

    outputs = encoder(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True if cls.pooler_type in ['avg_top2', 'avg_first_last'] else False,
        return_dict=True,
    )

    pooler_output = cls.pooler(attention_mask, outputs)
    if cls.pooler_type == "cls" and not cls.model_args.mlp_only_train:
        pooler_output = cls.mlp(pooler_output)

    if not return_dict:
        return (outputs[0], pooler_output) + outputs[2:]

    return BaseModelOutputWithPoolingAndCrossAttentions(
        pooler_output=pooler_output,
        last_hidden_state=outputs.last_hidden_state,
        hidden_states=outputs.hidden_states,
    )


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, label_size, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.bert = BertModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = BertLMPredictionHead(config)

        if self.model_args.do_cls:
            self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.cls_output = nn.Linear(config.hidden_size, label_size)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.bert,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )



class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, label_size, *model_args, **model_kargs):
        super().__init__(config)
        self.model_args = model_kargs["model_args"]
        self.label_size = label_size

        self.roberta = RobertaModel(config, add_pooling_layer=False)

        if self.model_args.do_mlm:
            self.lm_head = RobertaLMHead(config)

        if self.model_args.do_cls or self.model_args.do_soft_supervise_cls:
            self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.cls_output = nn.Linear(config.hidden_size, label_size)

        if self.model_args.do_soft_supervise_pmi is True and self.model_args.supercl_pmiweights_file is not None:
            self.supercl_weights = torch.load(self.model_args.supercl_pmiweights_file).to(self.device)

        if self.model_args.dual_training_cls is True:
            self.cls_weight_network = RobertaForSequenceClassification.from_pretrained(self.model_args.model_name_or_path, num_labels=self.label_size)

        cl_init(self, config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sent_emb=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        if sent_emb:
            return sentemb_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return cl_forward(self, self.roberta,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                mlm_input_ids=mlm_input_ids,
                mlm_labels=mlm_labels,
            )
