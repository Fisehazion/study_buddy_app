# app.py: Main Streamlit app for AI-Powered Study Buddy

import streamlit as st
from utils import extract_text_from_file, generate_summary, generate_flashcards, generate_quiz, answer_question

# Set page config for cool theme
st.set_page_config(
    page_title="Askualu AI Study Buddy",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for engaging look (gradients, shadows, professional quiz styling)
st.markdown("""
    <style>
    .main {background: linear-gradient(to bottom right, #f0f8ff, #e6e6fa);}
    .stButton>button {background-color: #4CAF50; color: white; border: none; padding: 10px; border-radius: 5px; box-shadow: 0 4px #999;}
    .stExpander {border: 1px solid #ddd; border-radius: 5px; padding: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1);}
    .stTab {background-color: #ffffff; border-radius: 10px; padding: 10px;}
    .quiz-choice {margin-left: 20px;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for options
st.sidebar.title("🧠 Study Buddy Controls")
st.sidebar.markdown("Upload your notes and let's study smarter! 🚀")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your notes (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'])

# Action selection
action = st.sidebar.selectbox("What do you want to do?", ["Summarize", "Generate Flashcards", "Generate Quiz", "Ask a Question"])

# For question-asking mode
if action == "Ask a Question":
    user_question = st.sidebar.text_input("Enter your question here:")

# For quiz mode: Choose answer display
if action == "Generate Quiz":
    quiz_mode = st.sidebar.radio("Quiz Answer Mode", ["Show Answers Immediately", "Submit for Final Score"])

# Main content area with tabs for better organization
st.title("📖 Askualu AI-Powered Study Buddy")
st.markdown("Upload your study materials and get instant AI help! 🚀")

if uploaded_file:
    with st.spinner("Extracting text... 📄"):
        try:
            text = extract_text_from_file(uploaded_file)
            st.success("Text extracted successfully! Ready to process.")
            
            # Display raw text in expander, formatted with paragraphs
            with st.expander("View Raw Extracted Text"):
                formatted_text = "\n\n".join([p.strip() for p in text.split('\n') if p.strip()])
                st.markdown(formatted_text)
            
            # Tabs for features
            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Flashcards", "Quiz", "Q&A"])
            
            if st.sidebar.button("Go! ✨"):
                progress = st.progress(0)
                
                if action == "Summarize":
                    with tab1:
                        with st.spinner("Generating summary..."):
                            summary = generate_summary(text)
                            progress.progress(100)
                        st.subheader("Structured Summary")
                        st.markdown(summary)
                        if st.button("Regenerate Summary"):
                            with st.spinner("Regenerating summary..."):
                                summary = generate_summary(text)
                                st.markdown(summary)
                
                elif action == "Generate Flashcards":
                    with tab2:
                        with st.spinner("Creating flashcards..."):
                            flashcards = generate_flashcards(text)
                            progress.progress(100)
                        st.subheader("Flashcards")
                        for card in flashcards:
                            with st.expander(f"**Q: {card['question']}**"):
                                st.markdown(card["answer"])
                        if st.button("Regenerate Flashcards"):
                            with st.spinner("Regenerating flashcards..."):
                                flashcards = generate_flashcards(text)
                                for card in flashcards:
                                    with st.expander(f"**Q: {card['question']}**"):
                                        st.markdown(card["answer"])
                        if st.button("Export Flashcards as CSV"):
                            csv = "\n".join([f"{c['question']},{answer}" for c, answer in [(card, card['answer'].replace('\n', ' ')) for card in flashcards]])
                            st.download_button("Download CSV", csv, "flashcards.csv")
                
                elif action == "Generate Quiz":
                    with tab3:
                        with st.spinner("Building quiz..."):
                            quiz = generate_quiz(text)
                            progress.progress(100)
                        st.subheader("Quiz Time! 🎉")
                        score = 0
                        answers = []
                        
                        if quiz_mode == "Show Answers Immediately":
                            for i, q in enumerate(quiz):
                                st.write(f"**Question {i+1}: {q['question']}**")
                                for j, choice in enumerate(q['choices'], 1):
                                    st.markdown(f"<div class='quiz-choice'>{j}. {choice}</div>", unsafe_allow_html=True)
                                st.markdown(f"**Correct Answer:** {q['correct']}")
                                st.markdown(f"**Explanation:** {q['explanation']}")
                        
                        else:  # Submit for Final Score
                            for i, q in enumerate(quiz):
                                st.write(f"**Question {i+1}: {q['question']}**")
                                choices = [f"{j}. {choice}" for j, choice in enumerate(q['choices'], 1)]
                                choice = st.radio("Select an answer:", choices, key=f"q{i}")
                                # Extract the choice text (remove number prefix)
                                selected = choice[choice.index(". ") + 2:] if ". " in choice else choice
                                answers.append((selected, q['correct'], q['explanation']))
                            if st.button("Submit Quiz"):
                                score = sum(1 for ans, cor, _ in answers if ans == cor)
                                st.write(f"Your score: {score}/{len(quiz)}")
                                if score == len(quiz):
                                    st.balloons()
                                # Show correct answers and explanations
                                for i, (ans, cor, exp) in enumerate(answers):
                                    st.write(f"**Q{i+1}**: Your answer: {ans} | **Correct**: {cor}")
                                    st.markdown(f"**Explanation:** {exp}")
                
                elif action == "Ask a Question" and user_question:
                    with tab4:
                        with st.spinner("Answering your question..."):
                            answer = answer_question(text, user_question)
                            progress.progress(100)
                        st.subheader("Answer")
                        st.write(answer)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}. Try a different file or check the file format.")
else:
    st.info("Upload a file to get started! Supported: PDF, DOCX, TXT.")