import torch
from transformers import BertTokenizer, BertForTokenClassification


def predict(model, tokenizer, sentence):
    tokens = sentence.split()
    encoding = tokenizer(tokens, return_tensors="pt", is_split_into_words=True, padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**encoding)
    
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
    tokenized_sentence = tokenizer.convert_ids_to_tokens(encoding['input_ids'].squeeze().tolist())
    
    return list(zip(tokenized_sentence, predictions))

if __name__ == "__main__":
    model_path = "./ner_best_model"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForTokenClassification.from_pretrained(model_path)
    model.eval()
    
    sentence = input("Enter a sentence: ")
    predictions = predict(model, tokenizer, sentence)
    
    print("\nPredictions:")
    for token, label in predictions:
        print(f"{token}: {label}")

