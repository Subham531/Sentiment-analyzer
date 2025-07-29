Assamese Sentiment Analyzer:

This project focuses on building a sentiment analysis model for the Assamese language, a low-resource Indo-Aryan language spoken primarily in the northeastern region of India. The project was developed as part of a collaborative research effort to bring Natural Language Processing (NLP) capabilities to underrepresented Indian languages.

📌 Key Highlights
📍 Language Focus: Assamese (অসমীয়া), an Indo-Aryan language with scarce labeled datasets and NLP tools.

🤖 Model Type: BiLSTM (Bidirectional Long Short-Term Memory), trained from scratch and fine-tuned on cleaned, annotated sentiment datasets.

🧹 Data Handling: Involved data cleaning, preprocessing, class balancing, and sentiment labeling (Positive, Negative, Neutral).

📊 Evaluation: Measured performance using precision, recall, F1-score.

💾 Datasets Used:

Original corpus scraped and annotated manually.

Final balanced dataset with over 35,000+ labeled Assamese text samples.

💻 Tech Stack
Component	Description
Language	Python
Notebook	Google Colab
Model	BiLSTM (Keras / TensorFlow)
Libraries	NumPy, Pandas, Pytorch, scikit-learn, Matplotlib
Visualization Classification report
Version Control	GitHub
Deployment (optional)	Hugging Face Model Hub

📁 Project Structure
bash
Copy
Edit
sentiment-analyzer/
│
├── data/                     # Datasets (cleaned, balanced, test sets)
│   └── balanced_dataset.csv
│
├── src/                      # Source code
│   └── model.py              # Model building and training script
│
├── notebook/                 # Development notebooks
│   └── sentiment_analysis.ipynb
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
├── LICENSE
└── .gitignore
⚙️ Setup Instructions
Clone the repository

bash
Copy
Edit
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Notebook
Open the notebook file and run it via Google Colab or locally using Jupyter.

🧪 Evaluation Metrics
Metric	Value (Example)
Precision	0.45
Recall	0.43
F1-Score	0.42
Accuracy	43%

🌐 Motivation
Most Indian languages lack high-quality, labeled datasets and open-source models for NLP tasks like sentiment analysis. This project was initiated to:

Promote research in low-resource Indian languages.

Create an open dataset and baseline model for Assamese sentiment analysis.

Provide a starting point for future multilingual NLP projects in Indic languages.

🤝 Contributors
Team
- Subham Das
- Pulakala Pritvi Raj
- Arpita Baruah
- Bhupali Das


🧾 License
This project is licensed under the MIT License.

  
