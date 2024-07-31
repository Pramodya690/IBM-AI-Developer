# pip install transformers pillow torch torchvision torchaudio

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load an image
image = Image.open("src/man.jpg")

# Prepare image
inputs = processor(image, return_tensors="pt")

# Generate captions with max_new_tokens parameter
outputs = model.generate(**inputs, max_new_tokens=30)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption:", caption)
