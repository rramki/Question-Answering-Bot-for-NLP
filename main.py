from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
index = None


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile):

    global documents, index

    reader = PdfReader(file.file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    documents = text.split("\n")

    embeddings = model.encode(documents)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return {"message": "PDF uploaded successfully"}


@app.get("/ask")
def ask(question: str):

    query_vector = model.encode([question])
    distances, ids = index.search(np.array(query_vector), k=1)

    answer = documents[ids[0][0]]

    return {"answer": answer}
