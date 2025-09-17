from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_dolly_data_loader(config, tokenizer, split="train"):
    # 加载数据集
    dataset = load_dataset("databricks/databricks-dolly-15k", split=split)
    
    def tokenize_function(examples):
        # 组合指令和响应
        texts = [f"Instruction: {instruction}\nResponse: {response}" 
                for instruction, response in zip(examples["instruction"], examples["response"])]
        
        # 标记化
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=config.max_length,
            return_tensors="pt"
        )
        
        # 设置标签
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # 应用标记化函数
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 设置格式为PyTorch张量
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.batch_size,
        shuffle=True if split == "train" else False
    )
    
    return dataloader