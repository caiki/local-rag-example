from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import os
import time
import tempfile
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import json
import openai

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DIM = 384  # Depends on embedding model
INDEX = faiss.IndexFlatL2(VECTOR_DIM)
DOC_METADATA = []  # Stores dicts: {"doc_type": str, "embedding": np.array, "field_list": list[str]}
openai.api_key = os.getenv("OPENAI_API_KEY")

# Example document type embedding registration (should be called in setup phase)
from typing import List, Dict

def register_sample_documents(sample_data: List[Dict]):
    """
    sample_data: [
      {
        "text":    <plain-text example of one doc>,
        "type":    <string name of that doc type>,
        "fields":  <list of field-names to extract>
      },
      ...
    ]
    """
    global INDEX, DOC_METADATA
    embeddings = []
    for entry in sample_data:
        text       = entry["text"]
        doc_type   = entry["type"]
        field_list = entry["fields"]
        vec = EMBED_MODEL.encode([text])[0]
        embeddings.append(vec)
        DOC_METADATA.append({
            "doc_type":   doc_type,
            "embedding":  vec,
            "field_list": field_list
        })
    print("DOC_METADATA> ",DOC_METADATA)
    INDEX.add(np.array(embeddings))


def extract_text(file: UploadFile) -> str:
    ext = file.filename.lower().split(".")[-1]
    content = file.file.read()
    if ext in ["jpg", "jpeg", "png"]:
        image = Image.open(tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}", mode="wb")).convert("RGB")
        return pytesseract.image_to_string(image)
    elif ext == "pdf":
        images = convert_from_bytes(content)
        return "\n".join([pytesseract.image_to_string(img) for img in images])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

def classify_document(text: str):
    embedding = EMBED_MODEL.encode([text])[0]
    print("DOC_METADATA> ",DOC_METADATA)
    if len(DOC_METADATA) == 0:
        return "Unknown", 0.0, []
    D, I = INDEX.search(np.array([embedding]), k=1)
    idx = I[0][0]
    confidence = float(1 / (1 + D[0][0]))
    return DOC_METADATA[idx]['doc_type'], confidence, DOC_METADATA[idx]['field_list']

from openai import AzureOpenAI

# Inicializa el cliente con tu endpoint, versión y clave de suscripción
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

def extract_entities(text: str, doc_type: str, field_list: List[str]):
    prompt = f"""
Given the following text extracted from a document of type '{doc_type}',
extract these fields: {field_list}.
Return your response as a valid JSON object with no additional text.
Document Text:
{text[:4000]}
"""
    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=4096,
    )

    try:
        content = response.choices[0].message.content
        return json.loads(content.strip())
    except Exception:
        return {}

@app.post("/extract_entities/")
async def extract_entities_endpoint(file: UploadFile = File(...)):
    start = time.time()
    try:
        text = extract_text(file)
        #print("text> ", text)
        doc_type, confidence, field_list = classify_document(text)
        entities = extract_entities(text, doc_type, field_list)
        processing_time = f"{time.time() - start:.2f}s"
        return JSONResponse(content={
            "document_type": doc_type,
            "confidence": round(confidence, 2),
            "entities": entities,
            "processing_time": processing_time
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 1) Prepare your prototype documents:
samples = [
    {
        "text": (
            "Invoice Number: INV-12345\n"
            "Date: 2024-01-01\n"
            "Total Amount: $450.00\n"
            "Vendor: ABC Corp"
        ),
        "type": "Invoice",
        "fields": ["invoice_number", "date", "total_amount", "vendor_name"]
    },
    {
        "text": (
            "Receipt No.: RCT-67890\n"
            "Date: 2024-03-15\n"
            "Amount: S/100.50\n"
            "Company: Empresa XYZ"
        ),
        "type": "Receipt",
        "fields": ["receipt_number", "date", "amount", "company"]
    },
    # add more prototypes for each doc type you expect…
]

# 2) Index them so classify_document() won’t return “Unknown”:
register_sample_documents(samples)


if __name__ == "__main__":

    # 3) Start your API:
    uvicorn.run("extract_entities:app", host="0.0.0.0", port=8008, reload=True)
