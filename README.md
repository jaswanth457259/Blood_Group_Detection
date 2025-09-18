Blood Group Detection
Overview

This project is designed to detect human blood groups (A, B, AB, O) using machine learning techniques. It provides a web application where users can upload sample data/images, process them through a trained model, and receive predictions instantly.

Features

Preprocesses and extracts features from input samples.

Trains a classifier for blood group prediction.

Provides a Flask-based web interface for uploading samples.

Saves trained models for reuse.

Includes a Jupyter notebook for experimentation and evaluation.

Requirements

The following Python packages are required:

Flask

scikit-learn

pandas

numpy

matplotlib

(Optional) TensorFlow / PyTorch

You can install the required packages using the following command:

pip install -r requirements.txt

Setup

Clone the repository or download the project files.

Install the required Python packages (see above).

(Optional) Retrain the model using the provided notebook.

How to Run

Run the Flask application with:

python app.py


Once started, open the application in your browser. You will be able to:

Upload a sample file.

Process it through the trained model.

View the predicted blood group instantly.

Functionality

preprocess_data(): Cleans and prepares raw input data for model training/prediction.

train_model(): Trains a classifier and saves the model for reuse.

predict_blood_group(): Loads the trained model and predicts the blood group of the uploaded sample.

Flask Routes: Handle file uploads, run predictions, and return results via the web app.

Files

app.py: Main Flask web application.

classification.ipynb: Notebook for model training and testing.

models/: Directory to store trained models.

templates/: HTML templates for the web app.

static/uploads/: Stores uploaded files.

requirements.txt: Lists required dependencies.

Important Notes

Ensure all dependencies are installed before running the app.

Training accuracy depends on the quality and size of the dataset.

For production, consider deploying the model to a cloud platform.

Future Improvements

Extend support for real blood smear image classification.

Improve model accuracy with larger, more diverse datasets.

Add authentication and user accounts.

Deploy the app on cloud services (Heroku, AWS, etc.).

License

This project is licensed under the MIT License.
