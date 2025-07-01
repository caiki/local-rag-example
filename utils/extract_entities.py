# ----------------------------------------------------------
# ----------------------------------------------------------
# RESTful API for:
#  1. Receiving documents (PDF/image) via HTTP
#  2. Extracting text with OCR (pytesseract / pdf2image)
#  3. Getting semantic embeddings (Sentence-Transformers)
#  4. Classifying document type with FAISS
#  5. Extracting requested entities using Azure OpenAI
#  6. Returning JSON with type, confidence, and extracted fields
# Each line is commented to clarify its purpose.
# ----------------------------------------------------------

# ----------------------- IMPORTS --------------------------
from fastapi import FastAPI, File, UploadFile, HTTPException   # Web framework and data models
from fastapi.responses import JSONResponse                     # Send custom JSON responses
from fastapi.middleware.cors import CORSMiddleware             # Enable CORS for frontend access
from typing import List, Dict                                  # Static typing (Lists and Dictionaries)
import uvicorn                                                 # ASGI server to run FastAPI
import pytesseract                                             # OCR engine based on Tesseract
from PIL import Image                                          # Load and process images
from pdf2image import convert_from_bytes                       # Convert PDF pages to images
import os                                                      # Access environment variables and OS utilities
import time                                                    # Measure processing time
import tempfile                                                # Create secure temporary files
import faiss                                                   # Vector search engine
import numpy as np                                             # Numerical operations
from sentence_transformers import SentenceTransformer          # Embedding model
from openai import AzureOpenAI                                 # Azure OpenAI client
import json                                                    # Parse and generate JSON
import re                                                      # Regular expressions
# ----------------------------------------------------------
# ---------------------- FASTAPI APP -----------------------
app = FastAPI()                                              # Main application instance

# CORS middleware to accept requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                                     # Allowed origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],                                     # Allowed HTTP methods
    allow_headers=["*"],                                     # Allowed headers
)
# ----------------------------------------------------------

# ---------------- GLOBAL CONFIGURATION --------------------
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")        # Lightweight 384-dim embedding model
VECTOR_DIM = 384                                             # Embedding dimension
INDEX = faiss.IndexFlatL2(VECTOR_DIM)                        # FAISS index using L2 similarity
DOC_METADATA: List[Dict] = []                                # Metadata of registered documents
# ----------------------------------------------------------

# ------------- AZURE OPENAI CLIENT (LLM) ------------------
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)
# ----------------------------------------------------------

# ------------ REGISTER SAMPLE DOCUMENTS -------------------
def register_sample_documents(sample_data: List[Dict]):      # Receives a list of sample documents
    """
    Registers embeddings and metadata of sample documents
    so the similarity-based classifier has reference points.
    """
    global INDEX, DOC_METADATA                               # Use global variables

    embeddings = []                                          # Temporary list for embedding vectors

    for entry in sample_data:                                # Iterate over each sample document
        text = entry["text"]                                 # Text content
        doc_type = entry["type"]                             # Document type (Invoice, Receipt, Contract, etc.)
        field_list = entry["fields"]                         # List of fields to extract for this type

        vec = EMBED_MODEL.encode([text])[0]                  # Get embedding (ndarray)
        embeddings.append(vec)                               # Add to embedding batch

        DOC_METADATA.append({                                # Store associated metadata
            "doc_type": doc_type,
            "embedding": vec,
            "field_list": field_list
        })

    print(f"Registered documents: {len(DOC_METADATA)}")      # Logging
    INDEX.add(np.array(embeddings))                          # Add all to the FAISS index
# ----------------------------------------------------------

# ------------------ OCR EXTRACTION FUNCTION ------------------
def extract_text(file: UploadFile) -> str:
    """
    Extracts text from images or PDFs using Tesseract OCR.
    Several configurations (`--psm`) are tested to improve
    quality, and the longest extracted text is returned.
    """
    try:
        ext = file.filename.lower().split(".")[-1]           # Determine file extension
        file.file.seek(0)                                    # Reset file pointer
        content = file.file.read()                           # Read file bytes

        # --- Process IMAGE ---
        if ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp_file:
                tmp_file.write(content)                      # Save image temporarily
                tmp_file.flush()

                image = Image.open(tmp_file.name).convert("RGB")  # Open image in RGB mode

                # Various OCR configurations (psm = page segmentation mode)
                configs = [
                    '--psm 6',                               # Assume a uniform block of text
                    '--psm 4',                               # Column-based analysis
                    '--psm 1',                               # Full page
                ]

                texts: List[str] = []                        # Collected texts

                for config in configs:                       # Run OCR with each config
                    try:
                        text = pytesseract.image_to_string(image, config=config)
                        if text and len(text.strip()) > 50:  # Minimum content threshold
                            texts.append(text)
                    except Exception:                        # Ignore Tesseract errors
                        continue

                os.unlink(tmp_file.name)                     # Delete temporary file

                return max(texts, key=len) if texts else ""  # Return longest text found

        # --- Process PDF ---
        elif ext == "pdf":
            # Convert first 5 pages to images (high DPI for quality)
            images = convert_from_bytes(content, dpi=400, first_page=1, last_page=5)
            all_texts: List[str] = []

            for img in images:                               # OCR page by page
                img = img.convert("RGB")
                page_texts: List[str] = []

                for config in ['--psm 6', '--psm 4', '--psm 1']:
                    try:
                        text = pytesseract.image_to_string(img, config=config)
                        if text and len(text.strip()) > 20:
                            page_texts.append(text)
                    except Exception:
                        continue

                if page_texts:
                    all_texts.append(max(page_texts, key=len))  # Best text from page

            return "\n\n".join(all_texts)                    # Full extracted text

        # --- Unsupported format ---
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {ext}")

    except Exception as e:                                   # Catch general errors
        print(f"Error extracting text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
# ----------------------------------------------------------

# ------ FUNC. TO PREPROCESS TEXT FOR LLM EXTRACTION -------
def preprocess_text_for_extraction(text: str) -> str:
    """
    Normalizes the text (spaces) and labels potentially
    important lines (keywords) to provide extra context to the LLM.
    """
    text = re.sub(r'\s+', ' ', text)                         # Reduce multiple spaces

    lines = text.split('\n')                                 # Split into original lines
    processed_lines: List[str] = []

    for line in lines:
        line = line.strip()
        if len(line) > 2:
            # If line contains generic keywords, mark as IMPORTANT
            if any(
                kw in line.lower()
                for kw in [
                    'invoice', 'factura', 'contract', 'contrato',
                    'receipt', 'recibo', 'number', 'numero',
                    'date', 'fecha', 'total', 'amount', 'vendor',
                    'supplier', 'cliente', 'company', 'empresa'
                ]
            ):
                processed_lines.append(f"IMPORTANT: {line}")
            else:
                processed_lines.append(line)

    return '\n'.join(processed_lines)                        # Return enriched text
# ----------------------------------------------------------

# --------------- DOCUMENT CLASSIFICATION ------------------
def classify_document(text: str):
    """
    Classifies the document based on cosine/L2 similarity
    with registered examples.
    Returns: (type, confidence, field_list)
    """
    if len(DOC_METADATA) == 0:
        return "Unknown", 0.0, []                            # No data → unknown

    embedding = EMBED_MODEL.encode([text])[0]                # Text embedding
    D, I = INDEX.search(np.array([embedding]), k=1)          # k-NN search (k=1)
    idx = int(I[0][0])                                       # Index of the nearest neighbor
    distance = float(D[0][0])                                # L2 distance
    confidence = 1 / (1 + distance)                          # Confidence (inverse of distance)

    return DOC_METADATA[idx]['doc_type'], confidence, DOC_METADATA[idx]['field_list']
# ----------------------------------------------------------

# --------------- EXTRACTION WITH AZURE OPENAI --------------
def extract_entities(text: str, doc_type: str, field_list: List[str]) -> Dict:
    """
    Uses an LLM (GPT) to extract an arbitrary list of fields
    (field_list) from the document text.
    The prompt is **generic**, applicable to any document type.
    """
    processed_text = preprocess_text_for_extraction(text)    # Preprocess text

    # Generic prompt (not limited to invoices)
    prompt = f"""
You are a senior document-intelligence system.

TASK: Extract exactly these fields from the document:
{', '.join(field_list)}

CRITICAL INSTRUCTIONS (generic):
1. Scan the entire document thoroughly.
2. Return **only** a valid JSON object with the requested field names.
3. If a field is clearly present, extract its value (string or number).
4. If multiple candidates exist, choose the most complete / recent.
5. If a date is found, normalise to YYYY-MM-DD when possible.
6. Do not include any explanations, just the pure JSON.
7. Use null when a field is truly not found.

DOCUMENT (pre-processed):
\"\"\"{processed_text}\"\"\"

Respond with JSON only.
"""

    try:
        # Call to GPT model (chat mode)
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise document parser. Output must be valid JSON with no extra text."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,                                 # Deterministic response
            max_tokens=500,                                  # Token limit
        )

        content = response.choices[0].message.content.strip()  # Returned text

        # Clean ```json``` blocks if present
        if content.startswith("```"):
            content = re.sub(r"```(json)?", "", content).strip().rstrip("```").strip()

        entities = json.loads(content)                       # Parse to dict

        # Normalize empty / null values
        validated: Dict[str, str | None] = {}
        for field in field_list:
            val = entities.get(field)
            if isinstance(val, str):
                val = val.strip()
                if val.lower() in {"", "null", "none", "n/a"}:
                    val = None
            validated[field] = val

        return validated                                     # Return cleaned entities

    except (json.JSONDecodeError, KeyError) as e:            # JSON error → use regex fallback
        print(f"JSON error: {e}. Falling back to regex.")
        return extract_entities_fallback(processed_text, field_list)

    except Exception as e:                                   # Other errors
        print(f"LLM extraction error: {e}. Falling back to regex.")
        return extract_entities_fallback(processed_text, field_list)
# ----------------------------------------------------------

# ----------------- REGEX FALLBACK EXTRACTION --------------
def extract_entities_fallback(text: str, field_list: List[str]) -> Dict:
    """
    Quick fallback using generic regular expressions.
    Supports common fields: *number, date, amount, total, vendor/company*.
    """
    entities: Dict[str, str | None] = {fld: None for fld in field_list}
    text_lower = text.lower()

    for field in field_list:
        # Generic number/ID
        if "number" in field or "id" in field:
            match = re.search(r'(invoice|receipt|contract)?\s*#?\s*([a-z0-9\-]{6,})', text_lower, re.I)
            if match:
                entities[field] = match.group(2).upper()

        # Dates
        elif "date" in field:
            match = re.search(r'(\d{4}[/-]\d{2}[/-]\d{2})', text) or \
                    re.search(r'(\d{2}[/-]\d{2}[/-]\d{4})', text)
            if match:
                entities[field] = match.group(1)

        # Amounts / totals
        elif any(kw in field for kw in ["amount", "total", "price"]):
            match = re.search(r'[\$€]\s?(\d+[.,]\d{2})', text) or \
                    re.search(r'(total|amount)\s*:?\s*(\d+[.,]\d{2})', text_lower)
            if match:
                entities[field] = (match.group(1) if match.lastindex == 1 else match.group(2)).replace(',', '.')

        # Company / vendor names
        elif any(kw in field for kw in ["vendor", "company", "supplier", "cliente"]):
            for line in text.split('\n'):
                if any(kw in line.lower() for kw in ["corp", "company", "gmbh", "s.a", "srl", "ltd", "empresa", "vendor", "supplier"]):
                    entities[field] = line.strip()
                    break

    return entities                                          # Final fallback dict
# ----------------------------------------------------------

# -------------------- MAIN ENDPOINT -----------------------
@app.post("/extract_entities/")
async def extract_entities_endpoint(file: UploadFile = File(...)):
    """
    REST Endpoint:
    1. Receives uploaded file
    2. Extracts text via OCR
    3. Classifies document type and fields
    4. Extracts entities
    5. Returns JSON with timing and confidence
    """
    start_time = time.time()                                # Start timer

    # 1) Extract text
    text = extract_text(file)
    if not text or len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Failed to extract text from document")

    # 2) Classify document
    doc_type, confidence, field_list = classify_document(text)

    # 3) Extract requested entities
    entities = extract_entities(text, doc_type, field_list)

    # 4) Calculate total processing time
    total_time = f"{time.time() - start_time:.2f}s"

    # 5) JSON response
    return JSONResponse(
        content={
            "document_type": doc_type,
            "confidence": round(confidence, 2),
            "entities": entities,
            "processing_time": total_time,
        }
    )
# ----------------------------------------------------------

# --------------------- HEALTH CHECK -----------------------
@app.get("/health")
async def health_check():
    """Health monitoring endpoint."""
    return {
        "status": "healthy",
        "registered_types": len(DOC_METADATA),
        "vector_index_size": INDEX.ntotal,
    }
# ----------------------------------------------------------

# --------------- SAMPLE DOCUMENTS (GENERIC) ----------------
sample_documents = [
    {
        # Generic invoice example
        "text": (
            "INVOICE CPB SOFTWARE GERMANY GMBH\n"
            "Invoice Number: 123100401\n"
            "Invoice Date: 2024-03-01\n"
            "Total Amount: $453.53\n"
            "Vendor: CPB SOFTWARE GERMANY GMBH"
        ),
        "type": "Invoice",
        "fields": ["invoice_number", "date", "total_amount", "vendor_name"],  # ← 4 original fields
    },
    {
        # Generic receipt example
        "text": (
            "RECEIPT ABC COMPANY LTD.\n"
            "Receipt Number: RCT-67890\n"
            "Date: 2024-03-15\n"
            "Amount: $100.50\n"
            "Company: ABC COMPANY LTD."
        ),
        "type": "Receipt",
        "fields": ["receipt_number", "date", "amount", "company"],             # 4 fields as well
    },
]
# Register sample examples when starting the module
register_sample_documents(sample_documents)
# ----------------------------------------------------------

# ------------- ENTRY POINT (script mode) -------------
if __name__ == "__main__":                                   # Only when executed directly
    print("🔵 Starting Server to extract entities")
    print(f"Registered types: {len(DOC_METADATA)}")
    uvicorn.run("extract_entities:app", host="0.0.0.0", port=8008, reload=True)  # Hot-reload for development
# ----------------------------------------------------------
