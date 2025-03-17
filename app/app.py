import gradio as gr
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model and tokenizer
model_path = "yourusername/product-review-sentiment-analyzer"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Define sentiment labels
labels = ["Negative", "Positive", "Neutral"]

# Define prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probabilities, dim=-1).item()
    
    return {labels[i]: float(probabilities[0][i]) for i in range(len(labels))}

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter a product review..."),
    outputs=gr.Label(num_top_classes=3),
    title="Product Review Sentiment Analyzer",
    description="Analyze the sentiment of product reviews as Positive, Negative, or Neutral."
)

# Launch app
demo.launch()
