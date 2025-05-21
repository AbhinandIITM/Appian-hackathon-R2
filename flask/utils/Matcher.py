
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image, ImageDraw, ImageFont
import re
import os

class Matcher():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to('cuda')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.crops_path = 'static/crops'

    def find_similar_type(self,label,type_embed_path,threshold):
        pass


    def process_bbox(self,image_path, bbox_string,processed_path,similarity_threshold=0.9):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        font = ImageFont.truetype("flask/fonts/DejaVuSans-Bold.ttf", size=20)
        pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([a-zA-Z]+)"
        matches = re.findall(pattern, bbox_string)

        for idx, (ymin_str, xmin_str, ymax_str, xmax_str, label) in enumerate(matches):
          
            matched_type = self.find_similar_type(label=label, type_embed_path='flask/dataset/NPZ/types.npz',threshold= similarity_threshold)
            display_label = matched_type if matched_type else label
            
            if display_label != 'none':
                ymin = int(ymin_str) / 1000 * height
                xmin = int(xmin_str) / 1000 * width
                ymax = int(ymax_str) / 1000 * height
                xmax = int(xmax_str) / 1000 * width

           
                crop = image.crop((xmin, ymin, xmax, ymax))
                crop_filename = f"{display_label}_{idx+1}.png"
                crop_path = os.path.join(self.crops_path, crop_filename)
                os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                crop.save(crop_path)
                print(os.path.abspath(crop_path))

                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)
                text_position = (xmin, ymin-25)
                draw.text(text_position, label, fill='blue', font=font)

        image.save(processed_path)