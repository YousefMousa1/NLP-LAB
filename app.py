from flask import Flask, request, send_file, render_template
import os
from werkzeug.utils import secure_filename
from model import image_loader, style_transfer, save_tensor_as_image
app = Flask(__name__)

INPUT_DIR = "static/input"
OUTPUT_DIR = "static/output"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transfer", methods=["POST"])
def transfer():
    content_file = request.files.get("content")
    style_file = request.files.get("style")

    if not content_file or not style_file:
        return "Please upload both content and style images.", 400

    # Save input images
    content_path = os.path.join(INPUT_DIR, secure_filename(content_file.filename))
    style_path = os.path.join(INPUT_DIR, secure_filename(style_file.filename))
    output_path = os.path.join(OUTPUT_DIR, "result.jpg")

    content_file.save(content_path)
    style_file.save(style_path)

    # For now, just return the style image as the "output"
    content_tensor = image_loader(content_path)
    style_tensor = image_loader(style_path)
    result_tensor = style_transfer(content_tensor, style_tensor)
    save_tensor_as_image(result_tensor, output_path)
    return f"""
        <h2>Stylized Image:</h2>
        <img src="/{output_path}" style="max-width: 500px;"><br><br>
        <a href="/">Go Back</a>
    """

if __name__ == "__main__":
    if __name__ == "__main__":
        print("🚀 Starting Flask app...")
        app.run(debug=True, host="0.0.0.0")