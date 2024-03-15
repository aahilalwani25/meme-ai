import os
import math
import cv2
import numpy as np
import onnxruntime

__labels = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

#
def read_image(image_path, target_size=320):
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    aspect = img_width / img_height

    if img_height > img_width:
        new_height = target_size
        new_width = int(round(target_size * aspect))
    else:
        new_width = target_size
        new_height = int(round(target_size / aspect))

    resize_factor = math.sqrt(
        (img_width**2 + img_height**2) / (new_width**2 + new_height**2)
    )

    img = cv2.resize(img, (new_width, new_height))

    pad_x = target_size - new_width
    pad_y = target_size - new_height

    pad_top, pad_bottom = [int(i) for i in np.floor([pad_y, pad_y]) / 2]
    pad_left, pad_right = [int(i) for i in np.floor([pad_x, pad_x]) / 2]

    img = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    img = cv2.resize(img, (target_size, target_size))

    image_data = img.astype("float32") / 255.0  # normalize
    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = np.expand_dims(image_data, axis=0)

    return image_data, resize_factor, pad_left, pad_top


def post_process(output, resize_factor, pad_left, pad_top):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int(round((x - w * 0.5 - pad_left) * resize_factor))
            top = int(round((y - h * 0.5 - pad_top) * resize_factor))
            width = int(round(w * resize_factor))
            height = int(round(h * resize_factor))
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        detections.append(
            {"class": __labels[class_id], "score": float(score), "box": box}
        )

    return detections


class NudeDetector:
    def __init__(self):
        self.onnx_session = onnxruntime.InferenceSession(
            os.path.join(os.path.dirname('best/'), "best.onnx"),
        )

        model_inputs = self.onnx_session.get_inputs()
        print(model_inputs)
        input_shape = model_inputs[0].shape
        print(input_shape)
        self.input_width = input_shape[2]  # 320
        self.input_height = input_shape[3]  # 320
        self.input_name = model_inputs[0].name
        print('input name: ',self.input_name)

    def detect(self, image_path):
        preprocessed_image, resize_factor, pad_left, pad_top = read_image(
            image_path, self.input_width
        )
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = post_process(outputs, resize_factor, pad_left, pad_top)

        return detections

    def censor(self, image_path, classes=[], output_path=None):
        detections = self.detect(image_path)
        if classes:
            detections = [
                detection for detection in detections if detection["class"] in classes
            ]

        img = cv2.imread(image_path)

        for detection in detections:
            box = detection["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]
            # change these pixels to pure black
            img[y : y + h, x : x + w] = (0, 0, 0)

        if not output_path:
            image_path, ext = os.path.splitext(image_path)
            output_path = f"{image_path}_censored{ext}"

        cv2.imwrite(output_path, img)

        return output_path
    
    def is_nude(self,detector: list)->bool:

        is_nude= False
        #check for each output label
        for det in detector:
            if(det['class']=='FEMALE_GENITALIA_EXPOSED' and det['score']>0.6):
                is_nude= True
                break
            if(det['class']=='MALE_GENITALIA_EXPOSED' and det['score']>0.6):
                is_nude= True
                break
            if(det['class']=='FEMALE_BREAST_EXPOSED' and det['score']>0.6):
                is_nude= True
                break
            if(det['class']=='BUTTOCKS_EXPOSED' and det['score']>0.6):
                is_nude= True
                break
        return is_nude



# import numpy as np
# import cv2

# def is_skin(normalized_r, normalized_g, normalized_b, h, s, v):
#     # Define your skin color model and threshold values
#     # Example: You can use a range of HSV values and normalized RGB values to classify skin pixels
#     # You would need to adjust these values based on your specific skin color model
#     # Define your own threshold values based on your model
#     skin_hsv_lower = (0, 20, 70)
#     skin_hsv_upper = (30, 255, 255)
#     skin_normalized_rgb_lower = (0.75, 0.4, 0.3)
#     skin_normalized_rgb_upper = (1.0, 0.6, 0.5)

#     if (skin_hsv_lower[0] <= h <= skin_hsv_upper[0] and
#         skin_hsv_lower[1] <= s <= skin_hsv_upper[1] and
#         skin_hsv_lower[2] <= v <= skin_hsv_upper[2] and
#         skin_normalized_rgb_lower[0] <= normalized_r <= skin_normalized_rgb_upper[0] and
#         skin_normalized_rgb_lower[1] <= normalized_g <= skin_normalized_rgb_upper[1] and
#         skin_normalized_rgb_lower[2] <= normalized_b <= skin_normalized_rgb_upper[2]):
#         return True
#     else:
#         return False

# def calculate_percentage(skin_region, image_shape):
#     num_skin_pixels = len(skin_region)
#     total_pixels_in_region = (image_shape[0] * image_shape[1]) if skin_region else 1  # Avoid division by zero

#     percentage = (num_skin_pixels / total_pixels_in_region) * 100
#     return percentage

# def is_valid_pixel(x, y, image):
#         return 0 <= x < image.shape[1] and 0 <= y < image.shape[0]

# def is_skin_pixel(x, y, image):
#     r, g, b = image[y, x]
#     normalized_r = r / 255.0
#     normalized_g = g / 255.0
#     normalized_b = b / 255.0
#     hsv = cv2.cvtColor(np.uint8([[image[y, x]]]), cv2.COLOR_BGR2HSV)
#     h, s, v = hsv[0][0]

#     return is_skin(normalized_r, normalized_g, normalized_b, h, s, v)
# # Function to find connected skin pixels to form skin regions (using DFS)
# def find_connected_region(image, seed_pixel):
#     stack = [seed_pixel]
#     connected_region = []
#     while stack:
#         x, y = stack.pop()
#         if is_valid_pixel(x, y, image) and is_skin_pixel(x, y, image):
#             connected_region.append((x, y))
#             image[y, x] = [0, 0, 0]  # Mark the pixel as visited (optional)
#             for dx in [-1, 0, 1]:
#                 for dy in [-1, 0, 1]:
#                     if dx == 0 and dy == 0:
#                         continue
#                     stack.append((x + dx, y + dy))

#     return connected_region

# # Function to find the bounding polygon of a skin region
# def find_bounding_polygon(skin_region):
#     x_values = [x for x, y in skin_region]
#     y_values = [y for x, y in skin_region]

#     leftmost = min(x_values)
#     uppermost = min(y_values)
#     rightmost = max(x_values)
#     lowermost = max(y_values)

#     return leftmost, uppermost, rightmost, lowermost

# # Function to calculate the area of a bounding polygon
# def calculate_polygon_area(leftmost, uppermost, rightmost, lowermost):
#     width = rightmost - leftmost
#     height = lowermost - uppermost
#     return width * height

# # Function to count skin pixels within the bounding polygon
# def count_skin_pixels_in_polygon(image, leftmost, uppermost, rightmost, lowermost):
#     skin_count = 0
#     for x in range(leftmost, rightmost + 1):
#         for y in range(uppermost, lowermost + 1):
#             if is_skin_pixel(x, y, image=image):
#                 skin_count += 1
#     return skin_count

# # Function to calculate the average intensity of skin pixels inside the bounding polygon
# def calculate_average_intensity(image, leftmost, uppermost, rightmost, lowermost):
#     total_intensity = 0
#     num_pixels = 0
#     for x in range(leftmost, rightmost + 1):
#         for y in range(uppermost, lowermost + 1):
#             if is_skin_pixel(x, y, image=image):
#                 r, g, b = image[y, x]
#                 intensity = (r + g + b) / 3
#                 total_intensity += intensity
#                 num_pixels += 1
#     if num_pixels == 0:
#         return 0
#     return total_intensity / num_pixels


# def is_nude(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image not found or cannot be loaded.")

#     # Initialize variables for skin detection
#     total_pixels = image.shape[0] * image.shape[1]
#     binary_mask = np.zeros_like(image)
#     total_skin_count = 0
#     largest_skin_count = 0
#     second_largest_skin_count = 0
#     third_largest_skin_count = 0
#     num_skin_regions = 0
#     average_intensity = 0.0

#     # Initialize a list to store connected skin regions
#     skin_regions = []

#     # Loop through the image pixels
#     for x in range(image.shape[0]):
#         for y in range(image.shape[1]):
#             # 2. Obtain the RGB component values
#             r, g, b = image[x, y]

#             # 3. Calculate the corresponding Normalized RGB and HSV values
#             normalized_r = r / 255.0
#             normalized_g = g / 255.0
#             normalized_b = b / 255.0
#             # Calculate HSV values using OpenCV
#             hsv = cv2.cvtColor(np.uint8([[image[x, y]]]), cv2.COLOR_BGR2HSV)
#             h, s, v = hsv[0][0]

#             # 4. Determine if the pixel color satisfies the parameters for being skin
#             # - You'll need to define the skin color distribution model and the threshold values
#             if is_skin(normalized_r, normalized_g, normalized_b, h, s, v):
#                 total_skin_count += 1

#                 # 5. Label each pixel as skin or non-skin
#                 binary_mask[x, y] = 1

#                 # 7. Identify connected skin pixels to form skin regions
#                 connected_region = find_connected_region(image, x)
#                 if connected_region:
#                     skin_regions.append(connected_region)

#     # 6. Calculate the percentage of skin pixels relative to the size of the image
#     skin_percentage = (total_skin_count / total_pixels) * 100

#     # 8. Count the number of skin regions
#     num_skin_regions = len(skin_regions)
    

#     # 9. Identify pixels belonging to the three largest skin regions
#     sorted_skin_regions = sorted(skin_regions, key=lambda region: len(region), reverse=True)
#     if sorted_skin_regions:
#         largest_skin_region = sorted_skin_regions[0]
#     # Access other elements as needed
#     else:
#         largest_skin_region = []
#     largest_skin_region = sorted_skin_regions[0]
#     second_largest_skin_region = sorted_skin_regions[1]
#     third_largest_skin_region = sorted_skin_regions[2]

#     # 10. Calculate the percentage of the largest skin region relative to the image size
#     largest_skin_count = len(largest_skin_region)
#     largest_skin_percentage = (largest_skin_count / total_pixels) * 100

#     # 11. Identify the leftmost, the uppermost, the rightmost, and the lowermost skin pixels of the three largest skin regions
#     leftmost, uppermost, rightmost, lowermost = find_bounding_polygon(largest_skin_region)

#     # 12. Calculate the area of the bounding polygon
#     area_of_polygon = calculate_polygon_area(leftmost, uppermost, rightmost, lowermost)

#     # 13. Count the number of skin pixels within the bounding polygon
#     skin_pixels_in_polygon = count_skin_pixels_in_polygon(image, leftmost, uppermost, rightmost, lowermost)

#     # 14. Calculate the percentage of the skin pixels within the bounding polygon relative to the area of the polygon
#     percentage_within_polygon = (skin_pixels_in_polygon / area_of_polygon) * 100

#     # 15. Calculate the average intensity of the pixels inside the bounding polygon
#     average_intensity = calculate_average_intensity(image, leftmost, uppermost, rightmost, lowermost)

#     # 16. Classify the image based on the conditions you've defined
#     if skin_percentage < 15:
#         return "Not Nude"
#     elif (largest_skin_percentage < 0.35 and
#           calculate_percentage(second_largest_skin_region) < 0.30 and
#           calculate_percentage(third_largest_skin_region) < 0.30):
#         return "Not Nude"
#     elif largest_skin_percentage < 0.45:
#         return "Not Nude"
#     elif (skin_percentage < 0.30 and percentage_within_polygon < 0.55):
#         return "Not Nude"
#     elif (num_skin_regions > 60 and average_intensity < 0.25):
#         return "Not Nude"
#     else:
#         return "Nude"


# # Function to check if a pixel is a skin pixel based on your defined skin color model and thresholds

# # Example usage:
# image_path = "memes_dataset/train/meme/5jy6j5.png"
# result = is_nude(image_path)
# print("Result:", result)


# import cv2
# import numpy as np

# def read_image(image_path):
#     image= cv2.imread(image_path)
#     return image

# def process_image(image):

#     #scan images row wise
#     for row in image:
#         #scan image column wise
#         for pixel in row:
#             r,g,b= pixel
#             #check if pixel has skin tone
#             if ((r>220)&(g>220)&(b<80)):
#                 return True
            
#     return False
            



# image_path = "memes_dataset/test/meme/7t6xqh.jpg"
# image = read_image(image_path)
# print(process_image(image))
# #print("Result:", result)