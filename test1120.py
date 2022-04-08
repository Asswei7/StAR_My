
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("roberta-large")
model = AutoModelForMaskedLM.from_pretrained("roberta-large")
