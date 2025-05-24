from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image, ImageDraw, ImageFont
import re
import os
import numpy as np

class Matcher():
    def __init__(self, base_url="/images"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.ttf_path = "fonts/DejaVuSans-Bold.ttf"
        self.crops_path = "static/crops"
        self.npz_path=  "dataset/types.npz"
        self.image_npz_path = "dataset/NPZ"
        self.images_path = "dataset/images/classified_images_gemma"
        self.base_url = base_url  

    def find_similar_type(self, label, type_embed_path, threshold):
        data = np.load(type_embed_path)
        type_embeddings = data['embeddings']
        types = data['types']

        inputs = self.clip_processor(text=[label], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            label_emb = self.clip_model.get_text_features(**inputs)
        label_emb = label_emb / label_emb.norm(dim=-1, keepdim=True)
        label_emb_np = label_emb.cpu().numpy()

        sims = (type_embeddings @ label_emb_np.T).squeeze(1)

        max_idx = np.argmax(sims)
        max_sim = sims[max_idx]

        return types[max_idx] if max_sim >= threshold else ""

    def find_similar_images(self, labels, crops):
        results = []
        npzs_path = self.image_npz_path
        for label, crop in zip(labels, crops):
            if label == 'none':
                continue

            npz_path = os.path.join(npzs_path, f"{label}.npz")
            if not os.path.exists(npz_path):
                print(f"Missing embeddings for label: {label}")
                continue

            data = np.load(npz_path, allow_pickle=True)
            db_embeddings = data['embeddings']
            db_filenames = data['image_names']

            inputs = self.clip_processor(images=crop, return_tensors="pt").to(self.device)
            with torch.no_grad():
                crop_emb = self.clip_model.get_image_features(**inputs)
            crop_emb = crop_emb / crop_emb.norm(dim=-1, keepdim=True)
            crop_emb_np = crop_emb.cpu().numpy()

            sims = db_embeddings @ crop_emb_np.T  # (N, 1)
            top_k = sims.squeeze().argsort()[-5:][::-1]  # top 5

            top_urls = []
            for idx in top_k:
                filename = db_filenames[idx]
                # Construct local path
                image_path = os.path.join(self.images_path, label, filename)
                # Convert to web URL for Flask
                url_path = f"{self.base_url}/{label}/{filename}".replace("\\", "/")
                top_urls.append(url_path)

            results.append({'label': label, 'topk_paths': top_urls})

        return results

    def process_bbox(self, image_path, bbox_string, processed_path, similarity_threshold=0.9):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        font = ImageFont.truetype(self.ttf_path, size=20)
        pattern = r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([a-zA-Z]+)"
        matches = re.findall(pattern, bbox_string)
        display_labels = []
        crops = []

        for idx, (ymin_str, xmin_str, ymax_str, xmax_str, label) in enumerate(matches):
            matched_type = self.find_similar_type(label=label, type_embed_path=self.npz_path, threshold=similarity_threshold)
            display_label = matched_type if matched_type else 'none'
            display_labels.append(display_label)
            if display_label != 'none':
                ymin = int(ymin_str) / 1000 * height
                xmin = int(xmin_str) / 1000 * width
                ymax = int(ymax_str) / 1000 * height
                xmax = int(xmax_str) / 1000 * width

                crop = image.crop((xmin, ymin, xmax, ymax))
                crop_filename = f"{display_label}_{idx+1}.png"
                crop_path = os.path.join(self.crops_path, crop_filename)
                os.makedirs(os.path.dirname(crop_path), exist_ok=True)
                crops.append(crop)
                crop.save(crop_path)
                print(f"Saved crop: {os.path.abspath(crop_path)}")

                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=2)
                text_position = (xmin, ymin-25)
                draw.text(text_position, label, fill='blue', font=font)

        image.save(processed_path)

        # Filter out invalid labels and crops before finding similar images
        valid_pairs = [(label, crop) for label, crop in zip(display_labels, crops) if label != 'none']
        if not valid_pairs:
            return []

        valid_labels, valid_crops = zip(*valid_pairs)
        final_results = self.find_similar_images(valid_labels, valid_crops)
        return final_results
