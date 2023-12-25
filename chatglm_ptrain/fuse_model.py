import argparse
from pprint import pprint
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='model/chatglm2-6b',
                        help='how ')
    parser.add_argument('--peft_model', type=str, default='model/peft',
                        help='how ')
    parser.add_argument('--output_model', type=str, default='model/output',
                        help='how ')

    return parser.parse_args()

if __name__ == '__main__':

    args = set_args()
    pprint(args)
    model_name_or_path = args.model



    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)#.half().cuda()


    peft_model_id = args.peft_model
    
    model = PeftModel.from_pretrained(model, peft_model_id)
    model = model.eval()

    # 合并lora
    model_merge = model.merge_and_unload()
    merge_lora_model_path = args.output_model

    model_merge.save_pretrained(merge_lora_model_path, max_shard_size="2GB")
    tokenizer.save_pretrained(merge_lora_model_path)
    