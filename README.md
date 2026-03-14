# Brain Tumor Classification Web App

A full-stack medical imaging web application built with Django that enables
clinicians to upload MRI scans and receive automated brain tumor classifications
powered by trained Convolutional Neural Networks (CNNs).

🌐 [Live Demo](https://th1rteen0-brain-tumor-classification-vm7a.onrender.com/)

---

## Overview

This app provides an end-to-end pipeline for brain tumor detection from MRI
scans. It implements a two-stage approach: a binary CNN for rapid screening,
followed by a fine-tuned EfficientNetB1 for tumor-type classification, with
Grad-CAM heatmaps providing visual interpretability for clinical decision support.
Models achieve up to 98.86% accuracy across a dataset of 7,000+ medical images.

Patient records, scan uploads, and historical results are all managed within the
app, supporting longitudinal monitoring over time.

---

## Features

- **MRI Scan Upload** — Securely upload patient MRI scans for real-time classification
- **Binary Classification** — Detects the presence or absence of a brain tumor
- **Multi-Class Classification** — Identifies tumor type (glioma, meningioma, pituitary)
- **Grad-CAM Explainability** — Heatmap visualizations highlighting the MRI regions influencing each prediction
- **Patient Records (CRUD)** — Create, view, update, and delete patient profiles
- **Result History** — Track and review a patient's scan results over time
- **Cloud Storage** — MRI scans stored securely via Amazon S3

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django (Python) |
| Database | PostgreSQL |
| File Storage | Amazon S3 |
| ML Framework | TensorFlow / Keras |
| Model Architecture | Custom CNN (binary) + EfficientNetB1 (multi-class) |
| Model Training | Google Colab (see [notebooks repo](https://github.com/th1rteen0/brain-tumor-classification-models)) |
| Deployment | Render |

---

## Model Performance

| Model | Type | Accuracy | Dataset Size |
|---|---|---|---|
| Binary Classifier | Tumor vs. No Tumor | 97.02% | 7,000+ images |
| Multi-Class Classifier | 4-class tumor type | 98.86% | 7,000+ images |

Models were trained in Google Colab and exported as `.keras` files for inference
within the Django app. See the [notebooks repository](https://github.com/th1rteen0/brain-tumor-classification-models) for full training
code, preprocessing steps, and evaluation metrics.

---

## Getting Started

### Prerequisites
- Python 3.10+
- PostgreSQL
- An AWS account with an S3 bucket configured

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/brain-tumor-app.git
cd brain-tumor-app
```

2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

   Create a `.env` file in the root directory:
```
# Django
SECRET_KEY=your_django_secret_key
DEBUG=True

# Database (local/development)
DB_ENGINE=django.db.backends.postgresql
DB_HOST=localhost
DB_NAME=braintumor_db
DB_PASSWORD=your_db_password
DB_PORT=5432
DB_USER=your_db_user

# Database (Render/production)
DATABASE_URL=your_render_postgres_url

# AWS S3
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_STORAGE_BUCKET_NAME=your_bucket_name
```

5. **Apply migrations**
```bash
python manage.py migrate
```

6. **Place model files**
 
   > **Note on model storage:**
   > For **local development and testing**, download the `.keras` model files from the
   > [notebooks repo](https://github.com/th1rteen0/brain-tumor-classification-models)
   > and place them in the `models/` directory as shown below. For **production deployment**,
   > the models are loaded directly from Amazon S3.
 
```
models/
├── binary_model.keras
└── multi_model.keras
```

7. **Run the development server**
```bash
python manage.py runserver
```

---

## Project Structure
```
brain-tumor-app/
├── doctors/               # Main Django app (patients, scans, classification logic)
│   ├── migrations/        # Database migration files
│   ├── templates/         # HTML templates
│   ├── admin.py           # Admin panel config
│   ├── models.py          # Database models
│   ├── views.py           # Request handling and CNN inference calls
│   ├── urls.py            # App-level URL routing
│   └── forms.py           # Django forms
├── models/                # Saved .keras model files
│   ├── binary_model.keras
│   └── multi_model.keras
├── mysite/                # Django project config
│   ├── settings.py        # Project settings
│   ├── urls.py            # Root URL configuration
│   ├── wsgi.py            # WSGI entry point
│   └── asgi.py            # ASGI entry point
├── staticfiles/           # Collected static assets (CSS, JS, images)
├── manage.py              # Django management CLI
├── render.yaml            # Render deployment configuration
├── requirements.txt       # Python dependencies
└── runtime.txt            # Python version for Render
```

---

## Research

This project is accompanied by an IEEE paper detailing the model
architecture, dataset, training methodology, and evaluation results.

📄 [Read the Paper](https://drive.google.com/file/d/1-Tr_ZdCki3MIbFk6pB3Z-5wmAxz0eSmh/view?usp=sharing)

---

## Related Repository

The Colab notebooks used to train and evaluate the CNN models are maintained
separately for reference:

🔗 [Brain Tumor CNN Notebooks](https://github.com/th1rteen0/brain-tumor-classification-models)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Disclaimer

This application is intended as a diagnostic aid to assist radiologists, not a
replacement. It is **not approved for clinical or diagnostic use at this stage**.
Always consult a licensed medical professional for diagnosis and treatment decisions.

---

𝑺𝑬𝑬 𝒀𝑶𝑼 𝑺𝑷𝑨𝑪𝑬 𝑪𝑶𝑾𝑩𝑶𝒀 . . .💫
