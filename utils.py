# utils.py: Helper functions for file processing and AI tasks

import pypdf
from docx import Document
import nltk
import spacy
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from transformers import T5ForConditionalGeneration, T5Tokenizer
from safetensors.torch import load_file
import torch
import pdfplumber  # For structured PDF extraction
from sentence_transformers import SentenceTransformer, util  # For duplicate filtering
import logging  # For debugging
import re  # For text cleaning

# Suppress verbose logs
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data if not already (run once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load SpaCy model (try en_core_web_lg, fallback to en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer for duplicate filtering (all-MiniLM-L6-v2 for efficiency)
duplicate_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    """
    Custom function to clean extracted text: remove extra whitespace, handle encoding issues.
    """
    if not text:
        return ""
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text

def extract_text_from_file(uploaded_file):
    """
    Extract text from uploaded file based on format.
    Supports PDF, DOCX, TXT. Uses pdfplumber for structured PDFs, falls back to pypdf for robustness.
    """
    file_name = uploaded_file.name.lower()
    text = ""
    
    if file_name.endswith('.pdf'):
        try:
            # Use pdfplumber for structured extraction (handles tables, columns)
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text(layout=True)
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            logger.warning(f"pdfplumber failed: {str(e)}. Falling back to pypdf.")
            # Fallback to pypdf
            uploaded_file.seek(0)
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    
    elif file_name.endswith('.docx') or file_name.endswith('.doc'):
        # Extract from DOCX
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            if para.text.strip():
                text += para.text + "\n"
    
    elif file_name.endswith('.txt'):
        # Extract from TXT
        text = uploaded_file.read().decode('utf-8', errors='ignore')
    
    else:
        raise ValueError("Unsupported file format. Use PDF, DOCX, or TXT.")
    
    # Clean extracted text
    text = clean_text(text)
    return text if text else "No readable text found in the document."

def generate_summary(text, max_length=200):
    """
    Generate a high-quality summary using Hugging Face's summarization pipeline.
    Uses facebook/bart-large-cnn. Handles edge cases to avoid index errors.
    """
    if not text.strip() or len(text) < 50:
        return "Text is too short to summarize. Please upload a document with more content."
    
    # Dynamically adjust max_length and min_length
    input_length = len(text.split())
    max_length = min(max_length, max(50, input_length // 2))
    min_length = min(30, max_length // 2)
    
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        result = summarizer(text[:2000], max_length=max_length, min_length=min_length, do_sample=False, truncation=True)
        summary_text = result[0]['summary_text'].strip() if result else ""
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        return f"Summary generation failed: {str(e)}. Try a longer or different document."
    
    if not summary_text:
        return "No summary generated. Try a document with more content."
    
    # Post-process: Clean and format summary
    sentences = sent_tokenize(summary_text)
    if not sentences:
        return summary_text.capitalize()
    
    # Capitalize and use bullets for >2 sentences
    sentences = [s.strip().capitalize() for s in sentences if s.strip()]
    if len(sentences) > 2:
        summary = "**Key Points:**\n- " + "\n- ".join(sentences)
    else:
        summary = " ".join(sentences)
    
    return summary

def generate_flashcards(text, num_cards=5):
    """
    Generate diverse flashcards using SpaCy for context-based questions.
    Avoids duplicates using sentence-transformers similarity. Handles empty/short inputs.
    """
    if not text.strip() or len(text) < 50:
        return [{"question": "No content available", "answer": "Text is too short to generate flashcards."}]
    
    try:
        doc = nlp(text[:5000])  # Limit input for memory
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Flashcard text processing failed: {str(e)}")
        return [{"question": "No content available", "answer": f"Error processing text: {str(e)}"}]
    
    if not sentences:
        return [{"question": "No content available", "answer": "Text is too short to generate flashcards."}]
    
    # Avoid duplicates: Filter unique sentences using similarity
    unique_sentences = [sentences[0]]
    embeddings = [duplicate_model.encode(sentences[0])]
    for s in sentences[1:]:
        embedding = duplicate_model.encode(s)
        similarities = util.cos_sim(embedding, embeddings)[0]
        if all(sim < 0.8 for sim in similarities):
            unique_sentences.append(s)
            embeddings.append(embedding)
    
    flashcards = []
    used_entities = set()  # Track used entities to avoid repetitive questions
    for i in range(min(num_cards, len(unique_sentences))):
        sentence = unique_sentences[i]
        doc_sentence = nlp(sentence)
        
        # Try to generate a question based on key verbs or phrases
        verbs = [token.text for token in doc_sentence if token.pos_ == "VERB" and token.text.lower() not in ["is", "are", "was", "were"]]
        if verbs:
            question = f"What does the text say about {verbs[0]}?"
        else:
            # Fallback to entity-based question, but avoid repetition
            entities = [ent.text for ent in doc_sentence.ents if ent.label_ in ['PERSON', 'ORG', 'DATE', 'GPE', 'EVENT'] and ent.text not in used_entities]
            if entities:
                question = f"What role does {entities[0]} play in the text?"
                used_entities.add(entities[0])
            else:
                question = f"What is the main idea of: {sentence[:50]}...?"
        
        # Format answer
        answer_parts = [s.strip().capitalize() for s in sentence.split('. ') if s.strip()]
        answer = "- " + "\n- ".join(answer_parts) if len(answer_parts) > 1 else answer_parts[0]
        flashcards.append({"question": question, "answer": answer})
    
    return flashcards

def generate_quiz(text, num_questions=3):
    """
    Generate professional multiple-choice quiz with relevant distractors using T5 and SpaCy.
    Avoids duplicates using sentence-transformers. Includes clear explanations.
    Handles edge cases.
    """
    if not text.strip() or len(text) < 50:
        return [{"question": "No content available", "choices": [], "correct": "", "explanation": "Text is too short to generate quizzes."}]
    
    try:
        model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl", use_safetensors=True, device_map="cpu")
        tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
    except Exception as e:
        logger.error(f"Quiz model loading failed: {str(e)}")
        return [{"question": "Quiz generation failed", "choices": [], "correct": "", "explanation": f"Error loading model: {str(e)}"}]
    
    sentences = sent_tokenize(text)
    if not sentences:
        return [{"question": "No content available", "choices": [], "correct": "", "explanation": "Text is too short to generate quizzes."}]
    
    # Avoid duplicates: Filter unique sentences using similarity
    unique_sentences = [sentences[0]]
    embeddings = [duplicate_model.encode(sentences[0])]
    for s in sentences[1:]:
        embedding = duplicate_model.encode(s)
        similarities = util.cos_sim(embedding, embeddings)[0]
        if all(sim < 0.8 for sim in similarities):
            unique_sentences.append(s)
            embeddings.append(embedding)
    
    quiz = []
    
    try:
        doc = nlp(text[:5000])  # Limit input for memory
    except Exception as e:
        logger.error(f"Quiz text processing failed: {str(e)}")
        return [{"question": "Quiz generation failed", "choices": [], "correct": "", "explanation": f"Error processing text: {str(e)}"}]
    
    for sentence in unique_sentences[:min(num_questions, len(unique_sentences))]:
        # Use T5 with a refined prompt for concise, professional questions
        input_text = f"generate question: What does the text say about this? {sentence} <hl>{sentence}<hl>"
        try:
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(**inputs, max_new_tokens=64, num_beams=4)
            question = tokenizer.decode(outputs[0], skip_special_tokens=True).capitalize()
            # Shorten question if too long
            if len(question) > 100:
                question = question[:97] + "..."
        except Exception as e:
            logger.error(f"Question generation failed: {str(e)}")
            quiz.append({"question": "Error generating question", "choices": [], "correct": "", "explanation": f"Question generation failed: {str(e)}"})
            continue
        
        # Generate relevant distractors
        correct = sentence.strip()
        distractors = []
        try:
            correct_doc = nlp(correct)
            # Select phrases from other sentences with high similarity
            for other_sentence in unique_sentences:
                if other_sentence != sentence:
                    other_doc = nlp(other_sentence)
                    for chunk in other_doc.noun_chunks:
                        similarity = correct_doc.similarity(chunk.doc) if chunk.doc.has_vector else 0
                        if 0.6 < similarity < 0.9 and chunk.text not in distractors and chunk.text != correct:
                            distractors.append(chunk.text)
                            if len(distractors) >= 3:
                                break
                if len(distractors) >= 3:
                    break
        except Exception as e:
            logger.warning(f"Distractor generation failed: {str(e)}")
        
        # Fallback distractors
        while len(distractors) < 3:
            distractors.append(f"Related concept {len(distractors) + 1}")
        
        choices = [correct] + distractors[:3]
        import random
        random.shuffle(choices)
        
        # Add professional explanation
        explanation = f"This question is based on the text: '{sentence[:100]}{'...' if len(sentence) > 100 else ''}'"
        
        quiz.append({
            "question": question,
            "choices": choices,
            "correct": correct,
            "explanation": explanation
        })
    
    return quiz

def answer_question(text, question):
    """
    Answer questions with high accuracy using DistilBERT.
    Handles edge cases and model errors.
    """
    if not text.strip() or not question.strip() or len(text) < 50:
        return "No text or question provided, or text is too short. Please upload a document and enter a question."
    
    try:
        qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", device=-1)
        result = qa(question=question, context=text[:512])
        answer = result['answer'].strip().capitalize()
        return answer if answer else "No answer found in the text. Try rephrasing the question."
    except Exception as e:
        logger.error(f"Question answering failed: {str(e)}")
        return f"Failed to answer question: {str(e)}. Try rephrasing the question or uploading a different document."