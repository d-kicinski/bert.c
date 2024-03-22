from transformers import BertModel, BertTokenizer


def show_bert():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = BertModel.from_pretrained(model_name)
    print(model.encoder.layer[0].attention.self.query)
    print(model.encoder.layer[0].attention.self.key)
    print(model.encoder.layer[0].attention.self.value)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    text = "hello there"
    model_input = tokenizer(text, return_tensors="pt")
    print(model_input)

    embeddings = model(input_ids=model_input["input_ids"]).pooler_output
    print(embeddings.shape)


if __name__ == '__main__':
    show_bert()
