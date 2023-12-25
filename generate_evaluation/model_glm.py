import torch
import torch.nn as nn
from transformers import BertModel,AutoTokenizer, AutoModel
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        return self.activation(self.fc(inputs))


class Similarity(nn.Module):
    def __init__(self, temp=0.05):
        super(Similarity, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.temp = temp

    def forward(self, x, y):
        return self.cos_sim(x, y) / self.temp


class Ranker(nn.Module):
    def __init__(self, config):
        super(Ranker, self).__init__()
        self.config = config
        # self.plm = BertModel.from_pretrained(config.model_name)
        self.plm = AutoModel.from_pretrained(config.model_name, trust_remote_code=True).half().cuda()
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                 lora_dropout=0.1,  # ['dense','dense_h_to_4h','dense_4h_to_h'] # 'query_key_value',
                                 target_modules=['query_key_value', ], )
        self.plm = get_peft_model(self.plm, peft_config)
        # peft_path = "chatglm_cheekpoint/checkpoint-12000/chatglm-lora.pt"
        # self.plm.load_state_dict(torch.load(peft_path), strict=False)

        self.similarity = Similarity()

        # 设置 special_tokens
        self.plm.resize_token_embeddings(len(config.tokenizer))
        # 添加线性层,用于增加 词loss
        self.Linear=nn.Linear(4096, 1)

    def forward(self, reference, context):
        #         选取 special token <e1>
        asss, pres = torch.max(reference.input_ids, dim=1)
        outputs = self.plm.transformer(**reference, output_hidden_states=True).last_hidden_state
        outputs = torch.permute(outputs, (1, 0, 2))
#         ref_outputs = torch.mean(outputs, dim=0, keepdim=False)
        ref_outputs = outputs[np.arange(outputs.shape[0]), pres]

        asss, pres = torch.max(context.input_ids, dim=1)
        outputs = self.plm.transformer(**context, output_hidden_states=True).last_hidden_state
        outputs = torch.permute(outputs, (1, 0, 2))
#         context_outputs=torch.mean(outputs, dim=0, keepdim=False)
        context_outputs = outputs[np.arange(outputs.shape[0]), pres]
#         print("111111111111111111111111111")
#         print(ref_outputs.shape)
#         print(context_outputs.shape)

        # ref_outputs = self.plm(**reference).pooler_output  # [batch_size, hidden_size]
        # context_outputs = self.plm(**context).pooler_output  # [batch_size * num_candidates, hidden_size]

        batch_size = ref_outputs.size(0)
        hidden_size = ref_outputs.size(-1)

        context_outputs = context_outputs.reshape(-1, self.config.num_candidates, hidden_size)
        ref_outputs = ref_outputs.unsqueeze(1).repeat(1, self.config.num_candidates, 1)
#         print("22222222222222222222222222")
#         print(ref_outputs.shape)
#         print(context_outputs.shape)
        a=self.similarity(context_outputs, ref_outputs)
        b=torch.squeeze(self.Linear(context_outputs.to(torch.float32)), dim=-1)
        return a,b


#         return self.similarity(context_outputs, ref_outputs), torch.squeeze(self.Linear(context_outputs), dim=-1)
