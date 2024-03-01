# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import warnings
# warnings.filterwarnings('ignore')
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import keras
# from keras import layers
# from keras.applications import efficientnet
# from keras.layers import TextVectorization
# from keras.preprocessing.image import load_img, img_to_array
# from sklearn.model_selection import train_test_split
# from nltk.translate.bleu_score import corpus_bleu

# keras.utils.set_random_seed(111)

# IMAGES_PATH = "/Datasets/Captions/Images/"
# CAPTIONS_PATH = "Datasets/Captions/captions.txt"

# # Desired image dimensions
# IMAGE_SIZE = (299, 299)

# # Fixed length allowed for any sequence
# SEQ_LENGTH = 25

# # Vocabulary size
# VOCAB_SIZE = 10000

# # Dimension for the image embeddings and token embeddings
# EMBED_DIM = 512

# # Per-layer units in the feed-forward network
# FF_DIM = 512

# # Batch size
# BATCH_SIZE = 512

# # Number of epochs
# EPOCHS = 35


# class CaptionGenerator:
#     # Loads captions (text) data and maps them to corresponding images.
#     def load_captions_data(self, filename):
#         with open(filename) as caption_file:
#             caption_data = caption_file.readlines()[1:]
#             caption_mapping = {}
#             text_data = []
#             images_to_skip = set()

#             for line in caption_data:
#                 line = line.rstrip("\n")
#                 # Each image is repeated five times for the five different captions.
#                 # Image name and captions are separated using a comma
#                 img_name, caption = line.split(",", 1)
#                 img_name = os.path.join(IMAGES_PATH, img_name.strip())
#                 # Removing caption that are either too short to too long
#                 tokens = caption.strip().split()
#                 if len(tokens) < 5 or len(tokens) > SEQ_LENGTH:
#                     images_to_skip.add(img_name)
#                     continue

#                 if img_name.endswith("jpg") and img_name not in images_to_skip:
#                     # A start and an end token must be added to each caption
#                     caption = "<start> " + caption.strip() + " <end>"
#                     text_data.append(caption)

#                     if img_name in caption_mapping:
#                         caption_mapping[img_name].append(caption)
#                     else:
#                         caption_mapping[img_name] = [caption]

#             for img_name in images_to_skip:
#                 if img_name in caption_mapping:
#                     del caption_mapping[img_name]

#             return caption_mapping, text_data

#     # Splits the dataset into training, validation, and test sets
#     def train_val_split(self, caption_data:dict, validation_size=0.2, test_size=0.05, shuffle=True):
#         # Getting the list of all image names
#         all_images = list(caption_data.keys())
        
#         # Shuffle if necessary
#         if shuffle:
#             np.random.shuffle(all_images)
        
#         train_keys, validation_keys = train_test_split(all_images, test_size=validation_size, random_state=42)
#         validation_keys, test_keys = train_test_split(validation_keys, test_size=test_size, random_state=42)
        
#         training_data = {img_name: caption_data[img_name] for img_name in train_keys}
#         validation_data = {img_name: caption_data[img_name] for img_name in validation_keys}
#         test_data = {img_name: caption_data[img_name] for img_name in test_keys}

#         # Return the splits
#         return training_data, validation_data, test_data
    
#     def visualaization(data, num_of_images):
#         count = 1
#         fig = plt.figure(figsize=(10,20))
#         for filename in list(data.keys())[100:100+num_of_images]:
#             captions = data[filename]
#             image_load = load_img(filename, target_size=(199,199,3))

#             ax = fig.add_subplot(num_of_images,2,count,xticks=[],yticks=[])
#             ax.imshow(image_load)
#             count += 1

#             ax = fig.add_subplot(num_of_images,2,count)
#             plt.axis('off')
#             ax.plot()
#             ax.set_xlim(0,1)
#             ax.set_ylim(0,len(captions))
#             for i, caption in enumerate(captions):
#                 ax.text(0,i,caption,fontsize=20)
#             count += 1
#         plt.show()

#     def custom_standardization(self, input_string):
#         # Lowercasing all of the captions
#         lowercase = tf.strings.lower(input_string)
#         # Charecters to remove
#         strip_chars = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~1234567890"
#         return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
    
    
    
# # Loading the dataset
# caption_generator= CaptionGenerator()
# captions_mapping, text_data = caption_generator.load_captions_data(CAPTIONS_PATH)

# # Spliting the dataset
# train_data, validation_data, test_data = caption_generator.train_val_split(captions_mapping)
# print(f"Total number of samples: {len(captions_mapping)}")
# print(f"----> Number of training samples: {len(train_data)}")
# print(f"----> Number of validation samples: {len(validation_data)}")
# print(f"----> Number of test samples: {len(test_data)}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained GPT-3 model (you can fine-tune your own model)
model_name = "EleutherAI/gpt-neo-2.7B"  # Example: GPT-3-like model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_meme_caption(image_description, temperature=0.7, max_length=50):
    # Tokenize the image description
    input_ids = tokenizer.encode(image_description, return_tensors="pt", add_special_tokens=True)

    # Generate text using the model
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=temperature,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )

    # Decode the generated caption
    generated_caption = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_caption

# Example usage:
image_description = "A cat wearing sunglasses"
generated_meme_caption = generate_meme_caption(image_description)
print("Generated Meme Caption:", generated_meme_caption)

