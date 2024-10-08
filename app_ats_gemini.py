from dotenv import load_dotenv
load_dotenv()

import streamlit as st 
import os
from PIL import Image
import pdf2image
import PyPDF2 as pdf
import google.generativeai as genai
import io
import base64

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

def get_gemini_response(input, pdf_content, prompt):
     model = genai.GenerativeModel('gemini-1.5-flash')
     response = model.generate_content([input,pdf_content[0],prompt])
     return response.text

def input_pdf(uploaded_file):
     if uploaded_file is not None:
     ## convert the pdf to image
        # images = pdf2image.convert_from_bytes(uploaded_file.read())

        # first_page = images[0]

        # #convert to bytes
        # img_byte_arr = io.BytesIO()
        # first_page.save(img_byte_arr, format='JPEG')
        # img_byte_arr = img_byte_arr.getvalue()

        # pdf_parts = [
        #     {
        #         "mime_type":"image/jpeg",
        #         "data": base64.b64encode(img_byte_arr).decode() #encode to base64 format
        #     }
        # ]
        # return pdf_parts

        reader = pdf.PdfReader(uploaded_file)
        text=""
        for page in range(len(reader.pages)):
            page = reader.pages[page]
            text += str(page.extract_text())
        return text
     
     
     else:
        raise FileNotFoundError("No file uploaded")
     
##streamlit app

st.set_page_config(page_title="ATS Resume Expert")
st.header("ATS Tracking System")
input_text = st.text_area("Job Description:", key="input")
uploaded_file = st.file_uploader("Uploaded your resume(PDF)", type=["pdf"])


if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")

submit1 = st.button("Tell me about the resume")
# submit2 = st.button("How Can I Improvise my Skills")
# submit3 = st.button("What are the Keywords That are Missing")
submit4 = st.button("Percentage match")

imput_prompt1 = """
    You are an experienced Technical Human Resource Manager with tech experience in the filed of 
    data science, full stack web development, big data engineering, data analyst, your task is to review the
    provided resume against the job description. 
    Please share your professional evaluation on whether the candidate's profile aligns with the role. 
    Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.

"""

input_prompt4 = """
    You are an skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science,
    full stack web development, big data engineering, data analyst and deep ATS functionality, 
    your task is to evaluate the resume against the provided job description. give me the percentage of
    match if the resume matchesthe job description. First the output should come as percentage and the
    keywords missing and last final thoughts.
    resume:{text}
    description:{jd}

    I want the response in one single string having the structure
    {{"JD Match":"%","MissingKeywords:[]","Profile Summary":""}}
"""

if submit1:
    if uploaded_file is not None:
        pdf_content = input_pdf(uploaded_file)
        response = get_gemini_response(imput_prompt1, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload a  resume PDF file to proceed.")

elif submit4:
    if uploaded_file is not None:
        pdf_content = input_pdf(uploaded_file)
        response = get_gemini_response(input_prompt4, pdf_content, input_text)
        st.subheader("The Response is")
        st.write(response)
    else:
        st.write("Please upload a  resume PDF file to proceed.")
    
