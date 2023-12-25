from transformers import AutoTokenizer, AutoModel
from peft import PeftModel, PeftConfig
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


model_name_or_path = "../model/chatglm2-6b"


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='auto', torch_dtype=torch.bfloat16)#.half().cuda()


model = model.eval()

peft_model_id = "output/adgen-chatglm2-6b-lora_version3/checkpoint-14000"
model = PeftModel.from_pretrained(model, peft_model_id)
model = model.eval()



file=""


text="独树镇的上下文是：独树镇，隶属河南省南阳市方城县，地处南阳，平顶山两市结合部，方城县东北部，东部及东南部邻近杨楼乡，南部及西南部与古庄店乡接壤，西部和西北部与杨集乡毗邻\n类型是：截取型\n独树镇的简称是："
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with torch.autocast("cuda"):
    encoded = tokenizer(text,return_tensors='pt', padding=True, truncation=True)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    outputs = model.generate(**encoded,max_new_tokens=32, num_beams=32, num_return_sequences=4, temperature=0.05,
                             repetition_penalty=1.2)
    generates=tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generates = [a.split("：")[-1].replace(" ","").replace("。","").replace("）","").replace(")","") for a in generates]
    print(generates)