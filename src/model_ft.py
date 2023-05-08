from transformers import Seq2SeqTrainingArguments, AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
import math
import torch
import torch.multiprocessing as mp


def run():
    torch.cuda.empty_cache()
    checkpoint = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    dataset = load_dataset('dkuntso/gen-qm-17000')

    def preprocess_function(examples):
        inputs = examples['utterance']
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

        labels = tokenizer(text_target=examples["answer"], max_length=256, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_ds = dataset.map(
        preprocess_function,
        remove_columns=dataset["train"].column_names,
        batched=True
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

    training_args = Seq2SeqTrainingArguments(
        output_dir="/home/ec2-user/projects/models/gen-qm-17-small",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=60,
        push_to_hub=True,
        predict_with_generate=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    tokenizer.save_pretrained("/home/ec2-user/projects/models/gen-qm-17-small")
    trainer.push_to_hub(commit_message='Seq2Seq')

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    run()