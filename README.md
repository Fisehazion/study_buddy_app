AI-Powered Study Buddy
  
Description
AI-Powered Study Buddy is a web-based application designed to enhance the learning experience for students and academics. Built with Python and Streamlit, it allows users to upload notes in PDF, DOCX, or TXT formats and generate personalized study aids using advanced AI models from Hugging Face. The app leverages natural language processing (NLP) to create summaries, flashcards, quizzes, and answer questions based on uploaded content, making it ideal for exam preparation, research summarization, or efficient studying.
The app features a modern, intuitive interface with tabs for different functionalities, custom CSS for a polished look, and robust error handling to ensure a seamless experience even with edge cases like short or malformed documents.
Key Highlights

AI Integration: Uses facebook/bart-large-cnn for summarization, valhalla/t5-base-qg-hl for quiz generation, and distilbert-base-cased-distilled-squad for question-answering.
Structured Outputs: Summaries and flashcards are formatted with bullets and headers for clarity.
Diverse Flashcards: Generates varied questions using verbs and unique entities, avoiding repetition.
Professional Quizzes: Produces multiple-choice quizzes with relevant distractors, numbered choices, and clear explanations.
Duplicate Prevention: Filters similar content using sentence-transformers for unique flashcards and quizzes.
Robustness: Handles short/empty files, malformed PDFs, and model errors with user-friendly messages.

This project was developed as an intermediate Python/AI project with assistance from Grok 4 (xAI), emphasizing ease of deployment and educational impact.
Features

File Upload and Extraction: Supports PDF, DOCX, TXT. Uses pdfplumber for structured PDF extraction (tables, layouts) with pypdf fallback.
Summarization: Creates concise, bullet-formatted summaries using BART.
Flashcards: Generates diverse questions based on sentence context and entities, with formatted answers.
Quizzes: Produces professional multiple-choice questions with relevant distractors (SpaCy similarity), shuffled choices, and explanations. Supports immediate answers or scored submission.
Question-Answering: Answers user queries using DistilBERT.
UI Enhancements: Features tabs, progress bars, regenerate buttons, and custom CSS (gradients, shadows).
Export Options: Download flashcards as CSV.

Installation

Clone the Repository:
git clone https://github.com/yourusername/study-buddy-app.git
cd study-buddy-app


Create Conda Environment (recommended):
conda create -n study_buddy python=3.10
conda activate study_buddy


Install Dependencies:
pip install -r requirements.txt
python -m spacy download en_core_web_lg


Cache AI Models (run once to avoid runtime downloads):
python -c "from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer; pipeline('summarization', model='facebook/bart-large-cnn'); T5ForConditionalGeneration.from_pretrained('valhalla/t5-base-qg-hl', use_safetensors=True); T5Tokenizer.from_pretrained('valhalla/t5-base-qg-hl'); pipeline('question-answering', model='distilbert-base-cased-distilled-squad')"



Usage

Run the App Locally:
streamlit run app.py


Open http://localhost:8501 in your browser.


Using the App:

Upload a PDF, DOCX, or TXT file in the sidebar.
Select an action: Summarize, Generate Flashcards, Generate Quiz, or Ask a Question.
Click "Go!" to process.
For quizzes, choose immediate answers or scored submission.
Regenerate summaries/flashcards or export flashcards as CSV.



Tech Stack

Frontend: Streamlit (1.50.0) with custom CSS for interactive UI.
AI Models: Hugging Face Transformers (4.57.1) for summarization, quiz generation, and Q&A.
NLP: Spacy (3.8.7) with en_core_web_lg for entity extraction and distractors.
Text Processing: NLTK (3.8.1) for sentence tokenization, pdfplumber (0.11.7) and pypdf (6.1.1) for PDFs, python-docx (1.2.0) for DOCX.
Duplicate Filtering: sentence-transformers (3.1.1) for similarity checks.
Backend: PyTorch (2.9.0) for model inference, safetensors (0.6.2) for safe model loading.

Requirements
See requirements.txt for exact versions:
streamlit==1.50.0
transformers==4.57.1
torch==2.9.0
torchvision
torchaudio
nltk==3.8.1
spacy==3.8.7
pypdf==6.1.1
python-docx==1.2.0
pymupdf==1.26.5
safetensors==0.6.2
pdfplumber==0.11.7
sentence-transformers==3.1.1
pillow==11.3.0

Future Improvements

User Authentication:

Implement login (e.g., Streamlit Authenticator) to save study sessions or flashcards in a database (SQLite/PostgreSQL).
Add cloud storage (e.g., AWS S3) for persistent user uploads.


Enhanced AI Models:

Upgrade to google/pegasus-large for better summaries or fine-tune models on academic datasets.
Integrate Grok API or GPT-4 for more creative question generation.


UI/UX Polish:

Add dark mode toggle and responsive CSS for mobile devices.
Include animations (e.g., Lottie files) for loading or success states.


Export and Sharing:

Support Anki-compatible flashcard exports (CSV/JSON).
Generate shareable links for quizzes/summaries via Streamlit or QR codes.


Performance Optimization:

Cache model outputs with st.cache_data in app.py.
Deploy on cloud GPU (e.g., Hugging Face Spaces) for faster inference.


Integrations:

Add Wikipedia API for supplementary quiz explanations.
Include email/SMS study reminders via Twilio/SendGrid.


Testing and Security:

Add unit tests with pytest for utils.py functions.
Sanitize inputs and limit file sizes to prevent attacks.


Accessibility:

Ensure WCAG compliance with high-contrast CSS.
Support multiple languages using multilingual models (e.g., mBART).



License
MIT License (or your preferred license). See LICENSE file for details.
Contributing
Contributions are welcome! Please:

Fork the repo.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.

Contact
For questions or feedback, open an issue or contact [your email or GitHub handle].