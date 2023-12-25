from transformers import BertTokenizer, BertConfig, T5Tokenizer, BartTokenizer
import os
import torch




class RankerConfig:
    def __init__(self, batch_size=4, num_epochs=10, lr=4e-5, mode='add_b', gradient_accumulation_steps=512,
                 ablation='all', top_k=12, gen_truncation=0, truncation=150, no_extract=False, context=False, version=1,
                 ratio=0, wandb=False):
        self.current_path = os.path.dirname(__file__)
        self.version = f"_v{version}" if version > 0 else ''
        self.ratio = f"_ratio{ratio}" if ratio > 0 else ''
        self.num_candidates = top_k
        self.truncate_len = truncation
        self.short = ""  # ["_short", ""]
        self.ABLATIONS = ['_nofull', '_notruth', '_nochar', '_noword', '_noheu', '_all_v2', '_all']
        self.MODES = ['add_b', 'add_s', 'sub', 'none', 'nofuse']
        self.post_edit = f"_{ablation}"
        assert self.post_edit in self.ABLATIONS
        self.extract = "" if no_extract else "_extract"  # ["_extract", ""]
        self.gen_truncation = f"_truncate{gen_truncation}" if gen_truncation > 0 and context and no_extract else ''
        self.context = "_context" if context else ''
        self.uni = ""
        assert mode in self.MODES
        self.mode = mode

        #         glm
        use_context= "_context" if context else ""
        self.train_path = os.path.join(self.current_path,
                                       f'data/rank_data/glm_v2_ranker_top12{use_context}_train_bs32t05p1_2.txt')
        self.val_path = os.path.join(self.current_path,
                                     f'data/rank_data/glm_v2_ranker_top12{use_context}_val_bs32t05p1_2.txt')
        self.test_path = os.path.join(self.current_path,
                                      f'data/rank_data/glm_v2_ranker_top12{use_context}_test_bs32t05p1_2.txt')


        self.model_name = os.path.join(self.current_path, 'model/chinese-macbert-base')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        # 设置添加 special_tokens
        add_special_tokens = ["</e1>","<e1>" ]
        self.tokenizer.add_special_tokens({"additional_special_tokens": add_special_tokens})

        self.bert_config = BertConfig.from_pretrained(self.model_name)
        self.max_len = 512
#         self.max_len = 50
        self.temp = 0.05
        self.hidden_size = self.bert_config.hidden_size
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.dropout_rate = 0.3
        self.clip_norm = 1.0
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = 0.01
        self.learning_rate = lr
        self.save_path = f"glm{self.version}_{str(lr).split('-')[0]}_ranker{self.ratio}{self.context}{self.gen_truncation}{self.uni}{self.extract}{self.post_edit}{self.short}_top{self.num_candidates}_{mode}_{self.batch_size}_accu_{self.gradient_accumulation_steps}_{self.num_epochs}_contrastive{use_context}.pth"
        self.logging_file_name = f"glm{self.version}_{str(lr).split('-')[0]}_ranker{self.ratio}{self.context}{self.uni}{self.extract}{self.post_edit}_truncate{self.truncate_len}{self.short}_top{self.num_candidates}_{mode}_logging_{self.batch_size}_accu_{self.gradient_accumulation_steps}_{self.num_epochs}_contrastive{use_context}.log"
        

        self.wandb = wandb
        if not wandb:
            os.environ["WANDB_DISABLED"] = "true"


if __name__ == '__main__':
    config = RankerConfig(context=True, no_extract=True, version=2, gen_truncation=100, truncation=100)
    print(config.train_path)
    print(config.num_candidates)
    print(config.logging_file_name)
    print(os.path.exists(config.train_path))
    print(config.context)



