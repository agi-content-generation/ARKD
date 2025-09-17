from rouge_score import rouge_scorer

def calculate_rouge_l(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)['rougeL']
        scores.append(score.fmeasure)
    
    return sum(scores) / len(scores) if scores else 0

def evaluate_model(student_model, teacher_model, tokenizer, dataloader, device, config):
    student_model.eval()
    teacher_model.eval()
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 生成响应
            student_outputs = student_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 解码预测和参考文本
            predictions = tokenizer.batch_decode(student_outputs, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # 计算ROUGE-L分数
    rouge_score = calculate_rouge_l(all_predictions, all_references)
    
    student_model.train()
    teacher_model.train()
    
    return rouge_score