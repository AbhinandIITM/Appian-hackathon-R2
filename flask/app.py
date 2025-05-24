from flask import Flask, render_template, request, url_for, redirect, session, send_file, abort
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from utils.Matcher import Matcher

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.secret_key = 'secretkey11'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

model_id = "google/paligemma-3b-mix-224"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu").eval()

matcher = Matcher()
def process_image(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    prompt = "<image> detect chair ; table ; sofa \n"

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        outputs = outputs[0][input_len:]

    result = processor.decode(outputs, skip_special_tokens=True)
    print(f"Model detection output: {result}")

    similar_results = matcher.process_bbox(image_path, result, output_path)
    return similar_results


@app.route('/', methods=['GET', 'POST'])
def home():
    original_url = None
    processed_url = None

    if request.method == 'POST':
        image = request.files.get('image')
        if image:
            filename = secure_filename(image.filename)
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(original_path)
            original_url = url_for('static', filename=f'uploads/{filename}')

            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
            processed_url = url_for('static', filename=f'processed/{processed_filename}')

            results = process_image(original_path, processed_path)

            # ✅ Clean image paths for rendering
            for result in results:
                result['topk_paths'] = [os.path.basename(path) for path in result['topk_paths']]

            session['results'] = results
            session['processed_url'] = processed_url
            return redirect(url_for('results'))

    return render_template('home.html', original_url=original_url, processed_url=processed_url)


@app.route('/images/<label>/<filename>')
def serve_image(label, filename):
    image_path = os.path.join('dataset', 'images', 'classified_images_gemma', label, filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    else:
        abort(404)


@app.route('/results')
def results():
    results_data = session.get('results', [])
    processed_url = session.get('processed_url', None)  # <-- add this
    return render_template('results.html', results=results_data, processed_url=processed_url)  # <-- include it


from flask import render_template, send_from_directory

@app.route('/view_image/<label>/<filename>')
def view_image(label, filename):
    return render_template('view_image.html', label=label, filename=filename)





if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
