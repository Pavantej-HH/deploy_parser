import fitz  # PyMuPDF
import re
import spacy
from nameparser import HumanName
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from io import BytesIO
import pytesseract
from PIL import Image

# Load custom spaCy model
nlp = spacy.load("outputtrf_v3/outputtrf_v3/model-best")

# Pydantic model for input validation
class ResumeText(BaseModel):
    text: str

# OCR fallback for image-based PDFs
def extract_text_with_ocr(file_bytes):
    text = ""
    doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(BytesIO(pix.tobytes("png")))
        page_text = pytesseract.image_to_string(img)
        text += page_text + "\n"
    doc.close()
    return text

# Try normal text extraction first, then fallback to OCR if text is too short
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    if len(text.strip()) < 100:  # If it's probably image-based
        text = extract_text_with_ocr(file_bytes)
    return text

# Validate phone number
def is_valid_phone(number):
    cleaned = re.sub(r"[^\d]", "", number)
    return 10 <= len(cleaned) <= 13

# Main parsing logic
def extract_info_with_spacy_regex(text):
    doc = nlp(text)
    extracted_info = {
        "name": "",
        "email": "",
        "phone": "",
        "location": "",
        "skills": []
    }

    lines = text.splitlines()

    for ent in doc.ents:
        if ent.label_ == "PERSON" and not extracted_info["name"]:
            extracted_info["name"] = ent.text.strip()
        elif ent.label_ in ["GPE", "LOC"] and not extracted_info["location"]:
            extracted_info["location"] = ent.text.strip()
        elif ent.label_ == "SKILL":
            extracted_info["skills"].append(ent.text.strip())

    # Parse name
    if extracted_info["name"]:
        parsed_name = HumanName(extracted_info["name"])
        extracted_info["First Name"] = parsed_name.first.strip()
        extracted_info["Last Name"] = parsed_name.last.strip()

    # Name fallback using regex
    if not extracted_info["name"]:
        for line in lines[:10]:
            name_match = re.match(r"(?i)^([A-Z][a-z]+)\s+([A-Z][a-z]+)", line.strip())
            if name_match:
                extracted_info["name"] = name_match.group().strip()
                parsed_name = HumanName(extracted_info["name"])
                extracted_info["First Name"] = parsed_name.first.strip()
                extracted_info["Last Name"] = parsed_name.last.strip()
                break

    if not extracted_info["name"]:
        for line in lines:
            match = re.search(r"(?i)^name[:\s]*([A-Z][a-z]+\s[A-Z][a-z]+)", line.strip())
            if match:
                extracted_info["name"] = match.group(1).strip()
                parsed_name = HumanName(extracted_info["name"])
                extracted_info["First Name"] = parsed_name.first.strip()
                extracted_info["Last Name"] = parsed_name.last.strip()
                break

    # Email
    email_match = re.search(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b", text)
    if email_match:
        extracted_info["email"] = email_match.group()

    # Phone using spaCy PHONE entity first
    phone_candidates = [ent.text.strip() for ent in doc.ents if ent.label_ == "PHONE"]
    for num in phone_candidates:
        if is_valid_phone(num):
            extracted_info["phone"] = num
            break

    # Regex fallback if needed
    if not extracted_info["phone"]:
        fallback_match = re.search(r"\b(?:\+91[\s\-]*)?\d{10,13}\b", text)
        if fallback_match and is_valid_phone(fallback_match.group()):
            extracted_info["phone"] = fallback_match.group()

    # LinkedIn
    linkedin_match = re.search(r"(https?://)?(www\.)?linkedin\.com/in/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    if linkedin_match:
        extracted_info["linkedin"] = linkedin_match.group()

    # GitHub
    github_match = re.search(r"(https?://)?(www\.)?github\.com/[a-zA-Z0-9_-]+", text, re.IGNORECASE)
    if github_match:
        extracted_info["github"] = github_match.group()

    return extracted_info

# FastAPI app
app = FastAPI(title="Resume Parser API")

@app.post("/parse_resume", response_model=dict)
async def parse_resume(resume: ResumeText):
    try:
        parsed_data = extract_info_with_spacy_regex(resume.text)
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")

@app.post("/parse_resume_pdf", response_model=dict)
async def parse_resume_pdf(file: UploadFile = File(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="File must be a PDF")
        file_bytes = await file.read()
        text = extract_text_from_pdf(file_bytes)
        parsed_data = extract_info_with_spacy_regex(text)
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")