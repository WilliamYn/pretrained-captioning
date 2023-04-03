# General imports
import base64
from PIL import Image
import json
from io import BytesIO
import os
# Flask imports
from flask import Flask, request
from flask_cors import CORS
import logging
# AI model imports
import torch
from lavis.models import load_model_and_preprocess
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from transformers import CLIPProcessor, CLIPModel


# Define these values with the values of the environment variables or default values will be used
caption_model_name = os.environ.get('caption_model_name', 'blip_caption')
caption_model_type_name = os.environ.get('caption_model_type_name', 'base_coco')
total_captions_number = int(os.environ.get('total_captions_number', 5))

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Load captioning model (based on environment variable) and zero-shot classification model (CLIP)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
caption_model, caption_vis_processors, _ = load_model_and_preprocess(name=caption_model_name, model_type=caption_model_type_name, is_eval=True, device=device)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def get_image_caption(raw_image) -> str:
    """
    get_image_caption generates the caption of the image to return to the client
    By setting use_nucleus_sampling to false, wee use beam search for the captioning, which generates the
    single caption that is the most accurate representation of the image. 
    :raw_image:         The image to caption
    return              A string of the image caption
    """
    image = caption_vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    caption = caption_model.generate({"image": image}, use_nucleus_sampling=False)
    return caption[0]

def generate_different_captions(raw_image) -> list:
    """
    Function that generates [total_captions_number] different captions describing the image. Wee use nucleus sampling to 
    randomize the captions. These captions will be used to create the image tags. 
    :raw_image:         The image to caption
    return              An array of [total_captions_number] different captions
    """
    image = caption_vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    all_captions = []
    while(len(all_captions) < total_captions_number):
        generated_caption = caption_model.generate({"image": image}, use_nucleus_sampling=True)
        if generated_caption[0] not in all_captions:
            all_captions += generated_caption
    return all_captions

def get_tags_from_captions(captions: list) -> list:
    """
    Function that splits the different captions into keywords to create the image tags
    :captions:          An array of the [total_captions_number] captions generated with nucleus sampling and the caption generated with beam searrch
    return              An array of words (tags)
    """
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+')
    tags = set()
    for caption in captions:
        words = tokenizer.tokenize(caption)
        for word in words:
            if word not in stop_words:
                tags.add(word)
    return list(tags)

def zero_shot_classification(raw_image, classes):
    """
    Takes an image and a list of words (the potential tags) and gives a score to each word describinging the
    percentage of the image occupied by that word by using a Zero Shot classification model
    :raw_image:         The image on which to apply zero shot classification
    :classes:           A list of tags
    return              Tuples with tags and the percentage of the tag in the image
    """
    inputs = processor(text=classes, images=raw_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]
    return zip(classes, probs)

@app.route("/hello-world", methods=["GET", "POST"])
def hello_world():
    """
    Used to test if the flask app is active
    """
    return "Hello World - This is the captioning service"

@app.route("/", methods=["GET", "POST"])
def handle_request():
    """
    Receives a GET or a POST request with {"image": base64_encoding} as the body and handles
    :return:        A JSON object with the following keys
                        - tags: an array of tuples, each tuple having the tag and the percentage of the image that the tag occupies
                        - captions: an array of a single caption describing the image (usually the most accurate caption)
                        - english_cap: an array of all the captions (used in the translation microservice)
    """
    if request.method == "POST" or request.method == "GET":
        # Extract image base64 from body of the request
        request_data = request.get_json()
        json_key = "image"
        if json_key not in request_data:
            return f"The JSON data must have {json_key} as a key with the base64 encoding", 400
        
        base64_str = request_data[json_key]
        utf8_encoding = base64_str.encode(encoding='utf-8')
        image_bytes_io = BytesIO(base64.b64decode(utf8_encoding))
        
        # Open image received in request
        op_image = None
        try:
            op_image = Image.open(image_bytes_io)
        except Exception as e:
            return f"Could not open image: {e}", 400

        raw_image = op_image.convert("RGB")
        image_caption = get_image_caption(raw_image)            # The most accurate caption of the image
        all_captions = generate_different_captions(raw_image)   # An array of multiple captions used to generate the tags of the image
        all_captions.append(image_caption)
        tags = get_tags_from_captions(all_captions)
        tags_and_probabilities = zero_shot_classification(raw_image, tags)

        op_image.close()
        image_bytes_io.close()

        return json.dumps({"tags": [(tag, float(prob)) for tag, prob in tags_and_probabilities], "captions": [image_caption], "english_cap": all_captions}, separators=(',', ':')), 200
    else:
        return "This endpoint only accepts GET and POST requests", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
