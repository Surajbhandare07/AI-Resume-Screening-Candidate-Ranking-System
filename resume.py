import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Combine job description with resumes
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()

    # Calculate cosine similarity
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()

    return cosine_similarities

# Streamlit app
st.title("📄 AI Resume Screening & Candidate Ranking System")

# Job description input
st.header("📝 Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("📂 Upload Resumes (PDF)")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Check condition before starting ranking
if uploaded_files and job_description:
    if st.button("🚀 Start Ranking"):  # Start button
        st.header("🏆 Ranking Resumes")

        resumes = []
        progress = st.progress(0)  # Progress bar initialization

        # Extract text from PDF files with progress update
        for i, file in enumerate(uploaded_files):
            text = extract_text_from_pdf(file)
            resumes.append(text)
            progress.progress((i + 1) / len(uploaded_files))

        # Rank resumes
        scores = rank_resumes(job_description, resumes)

        # Convert scores to float for sorting
        scores_float = [float(score) for score in scores]

        # Create DataFrame with resume names and scores
        results = pd.DataFrame({
            "Resume": [file.name for file in uploaded_files],
            "Score": scores_float  # Store as float
        })

        # Sort by Score in descending order and reset index
        results = results.sort_values(by="Score", ascending=False).reset_index(drop=True)

        # Assign Rank starting from 1
        results.insert(0, "Rank", range(1, len(results) + 1))

        # Convert Score to percentage format for display
        results["Score"] = results["Score"].apply(lambda x: f"{x * 100:.2f}%")

        # Display results with correct ranking
        st.dataframe(results, height=300)  # Display as table with fixed height

        # Display top candidate
        top_candidate = results.iloc[0]
        st.success(f"🏆 Top Candidate: **{top_candidate['Resume']}** with a score of **{top_candidate['Score']}**")
