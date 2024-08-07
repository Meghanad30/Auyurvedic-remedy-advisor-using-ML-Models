import streamlit as st
from transformers import pipeline
import urllib.request
import PyPDF2
import io


st.title("Ayurvedic Recommeder")

st.sidebar.header("control menu")

file = st.sidebar.file_uploader("Upload file here")
if file is not None:
	pdfdoc_remote = PyPDF2.PdfReader(file)


st.sidebar.write(len(pdfdoc_remote.pages))

pdf_text = ""

for i in range(len(pdfdoc_remote.pages)):
    #st.write(i)
    page = pdfdoc_remote.pages[i]
    page_content = page.extract_text()
    pdf_text += page_content
#st.write(pdf_text)

nlp = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    tokenizer="deepset/roberta-base-squad2",
)

context = pdf_text
question = st.text_input("Please enter your symptoms here")

question_set = {"context": context, "question": question}
results = nlp(question_set)

if st.button("Submit"):
	st.write("Answer: " + results["answer"])
