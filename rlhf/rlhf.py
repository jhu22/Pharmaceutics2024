import smiles_gpt as gpt
checkpoint = 'checkpoints/UVsmi'
hyperparams = {"batch_size": 256, "max_epochs": 30, "min_epochs": 15,
               "max_length": 512, "learning_rate": 5e-4, "weight_decay": 0.0,
               "adam_eps": 1e-8, "adam_betas": (0.9, 0.999),
               "scheduler_T_max": 150_000, "final_learning_rate": 5e-8,
               "vocab_size": 1_000, "min_frequency": 2, "top_p": 0.96,
               "n_layer": 6, "n_head": 12, "n_embd": 12 * 48}
tokenizer = gpt.SMILESBPETokenizer.get_hf_tokenizer('checkpoints/benchmark-10m/tokenizer.json', max_length=hyperparams["max_length"])


from transformers import GPT2Config, GPT2LMHeadModel
config = GPT2Config(vocab_size=tokenizer.vocab_size,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    n_layer=hyperparams["n_layer"],
                    n_head=hyperparams["n_head"],
                    n_embd=hyperparams["n_embd"],
                    n_positions=hyperparams["max_length"],
                    n_ctx=hyperparams["max_length"])
model = GPT2LMHeadModel(config)
model = GPT2LMHeadModel.from_pretrained(f"{checkpoint}/model", output_attentions=True)

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)

import json
def load_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data
data_dict = load_json_file('data/kto_dict.json')

from datasets import Dataset
dataset = Dataset.from_dict(data_dict)

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="./results",  # 训练结果的输出目录
    overwrite_output_dir=True,  # 是否覆盖之前的输出目录
    num_train_epochs=15,  # 训练的 epochs 数
    per_device_train_batch_size=8,  # 每个设备的训练批量大小
    save_steps=100,  # 每隔多少步保存一次模型
    logging_steps=50,  # 每隔多少步记录一次日志信息
    save_total_limit=10,  # 最多保存多少个模型
    evaluation_strategy="steps",  # 在何时进行评估（steps 或 epoch）
    eval_steps=100,  # 每隔多少步进行一次评估
    logging_dir="./logs",  # 日志输出目录
    do_train=True,  # 是否进行训练
    do_eval=True,  # 是否进行评估
    # 更多参数可以根据需要设置
)



dpo_trainer = DPOTrainer(
    model,
    model_ref,
    args=training_args,
    beta=0.1,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)


from trl import KTOConfig
training_args = KTOConfig(
    beta=0.1,
    desirable_weight=1.0,
    undesirable_weight=1.0,
    output_dir="./results",  # 训练结果的输出目录
    overwrite_output_dir=True,  # 是否覆盖之前的输出目录
    num_train_epochs=15,  # 训练的 epochs 数
    per_device_train_batch_size=8,  # 每个设备的训练批量大小
    save_steps=100,  # 每隔多少步保存一次模型
    logging_steps=50,  # 每隔多少步记录一次日志信息
    save_total_limit=10,  # 最多保存多少个模型
    evaluation_strategy="steps",  # 在何时进行评估（steps 或 epoch）
    eval_steps=100,  # 每隔多少步进行一次评估
    logging_dir="./logs",  # 日志输出目录
    do_train=True,  # 是否进行训练
    do_eval=True,  # 是否进行评估
)


from trl import KTOTrainer
kto_trainer = KTOTrainer(
    model,
    ref_model=None,
    #beta=0.1,
    args = training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer
)


from trl import DPOTrainer
dpo_trainer = DPOTrainer(
    model,
    ref_model=None,
    beta=0.1,
    args = training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer
)


from trl import CPOTrainer
cpo_trainer = CPOTrainer(
    model,
    args = training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    tokenizer=tokenizer
)

training_args = CPOConfig(
    beta=0.1,
    output_dir="./results",  # 训练结果的输出目录
    overwrite_output_dir=True,  # 是否覆盖之前的输出目录
    num_train_epochs=15,  # 训练的 epochs 数
    per_device_train_batch_size=8,  # 每个设备的训练批量大小
    save_steps=100,  # 每隔多少步保存一次模型
    logging_steps=50,  # 每隔多少步记录一次日志信息
    save_total_limit=10,  # 最多保存多少个模型
    evaluation_strategy="steps",  # 在何时进行评估（steps 或 epoch）
    eval_steps=100,  # 每隔多少步进行一次评估
    logging_dir="./logs",  # 日志输出目录
    do_train=True,  # 是否进行训练
    do_eval=True,  # 是否进行评估
)



training_args = CRPOConfig(
    beta=0.1,
    output_dir="./results",  # 训练结果的输出目录
    overwrite_output_dir=True,  # 是否覆盖之前的输出目录
    num_train_epochs=15,  # 训练的 epochs 数
    per_device_train_batch_size=8,  # 每个设备的训练批量大小
    save_steps=100,  # 每隔多少步保存一次模型
    logging_steps=50,  # 每隔多少步记录一次日志信息
    save_total_limit=10,  # 最多保存多少个模型
    evaluation_strategy="steps",  # 在何时进行评估（steps 或 epoch）
    eval_steps=100,  # 每隔多少步进行一次评估
    logging_dir="./logs",  # 日志输出目录
    do_train=True,  # 是否进行训练
    do_eval=True,  # 是否进行评估
)


cpo_trainer.train()
dpo_trainer.train()



kto_trainer.train()
model.save_pretrained("checkpoints/ktoUV/model/")

import torch
import tqdm
model.eval()

generated_smiles_list = []
n_generated = 1_000
for _ in tqdm.tqdm(range(n_generated)):

    # Generate from "<s>" so that the next token is arbitrary.
    smiles_start = torch.LongTensor([[tokenizer.bos_token_id]])
    # Get generated token IDs.
    smiles_start = smiles_start.to('cuda')
    generated_ids = model.generate(smiles_start,
                                   max_length=hyperparams["max_length"],
                                   do_sample=True, top_p=hyperparams["top_p"],
                                   pad_token_id=tokenizer.eos_token_id)
    # Decode the IDs into tokens and remove "<s>" and "</s>".
    generated_ids = generated_ids.to('cuda')
    generated_smiles = tokenizer.decode(generated_ids[0],
                                        skip_special_tokens=True)
    generated_smiles_list.append(generated_smiles)


for _ in tqdm.tqdm(range(n_generated)):
    smiles_start = torch.LongTensor([[tokenizer.bos_token_id]])
    smiles_start = smiles_start.to('cuda')
    generated_ids = model.generate(smiles_start,max_length=hyperparams["max_length"],do_sample=True, top_p=hyperparams["top_p"],pad_token_id=tokenizer.eos_token_id)
    generated_ids = generated_ids.to('cuda')
    generated_smiles = tokenizer.decode(generated_ids[0],skip_special_tokens=True)
    generated_smiles_list.append(generated_smiles)


ngenerated_smiles_list = []
np.save('kto-uv.npy', generated_smiles_list)
