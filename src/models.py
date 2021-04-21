class Rationale_With_Labels(torch.nn.Module):
    def __init__(self, D_in, num_labels):
        super(Transform, self).__init__()
        self.embeddings = AutoModelForTokenClassification.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(D_in*2, num_labels)

    def forward(self, input_ids, mask, attn):
        outputs = self.embeddings(input_ids, mask, labels = attn)
        loss = outputs.loss
        logits = outputs.logits 
        outputs = self.embeddings.bert(input_ids, mask)
        # out = outputs.last_hidden_state
        out=outputs[0]
        mean_pooling = torch.mean(out, 1)
        max_pooling, _ = torch.max(out, 1)

        embed = torch.cat((mean_pooling, max_pooling), 1)
        y_pred = self.classifier(self.dropout(embed))
        return y_pred, loss, logits