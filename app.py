
import streamlit as st
from transformers import pipeline
import pdfplumber
import re

# Tạo pipeline trả lời câu hỏi
qa_pipeline = pipeline("question-answering", framework="tf")

# Hàm trích xuất văn bản từ PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Hàm làm sạch văn bản
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
    return text.lower()

# Giao diện Streamlit
st.title("AI Hỗ trợ Nghiên cứu Văn bản Pháp luật")
st.write("Tải lên văn bản pháp luật (PDF) và đặt câu hỏi để nhận câu trả lời.")

# Tải file PDF
uploaded_file = st.file_uploader("Tải lên file PDF", type=["pdf"])
if uploaded_file is not None:
    # Xử lý file PDF
    raw_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(raw_text)
    st.write("**Nội dung trích xuất từ PDF:**")
    st.text_area("Văn bản pháp luật", value=raw_text, height=200)

    # Nhập câu hỏi
    question = st.text_input("Nhập câu hỏi:")
    if question:
        # Sử dụng AI để trả lời
        result = qa_pipeline(question=question, context=cleaned_text)
        st.write("**Câu trả lời:**", result['answer'])
