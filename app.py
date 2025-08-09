from flask import Flask, render_template, request
import os
from model_loader import load_selected_model, preprocess_image

app = Flask(__name__)

# Dummy daftar model (akan dipakai di HTML)
models = [
    {
        "name": "DenseNet",
        "description": "DenseNet adalah CNN yang menghubungkan setiap layer ke semua layer berikutnya.",
        "versions": ["DenseNet-121", "DenseNet-169", "DenseNet-201"]
    },
    {
        "name": "EfficientNet",
        "description": "EfficientNet mengoptimalkan kedalaman, lebar, dan resolusi secara seimbang.",
        "versions": ["EfficientNet-B0", "EfficientNet-B3", "EfficientNet-B7"]
    }
]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET'])
def classify():
    return render_template('classify.html', models=models)

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    image_file = request.files['image']
    model_name = request.form['model_name']

    # Simpan gambar sementara
    temp_path = os.path.join("static", "uploaded_image.jpg")
    image_file.save(temp_path)

    # Load model sesuai pilihan user
    try:
        model = load_selected_model(model_name)
    except ValueError as e:
        return str(e), 400

    # Preprocess gambar
    img_array = preprocess_image(temp_path)

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = f"Prediksi Class {prediction.argmax()}"

    # Placeholder Grad-CAM dan Gemini
    gradcam_filename = "contoh_gradcam.jpg"
    explanation = f"Hasil prediksi menunjukkan {predicted_class}. Penjelasan detail akan dihasilkan oleh LLM Gemini."

    return render_template('classify.html', models=models, result={
        "prediction": predicted_class,
        "explanation": explanation,
        "gradcam": gradcam_filename
    })

if __name__ == '__main__':
    app.run(debug=True)
