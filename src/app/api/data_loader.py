import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()

# Load data
books = pd.read_csv(
    "./../../notebooks/data/processed/books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "./../static/cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load and process documents for ChromaDB
loader = TextLoader(
    "./../../notebooks//data/processed/tagged_description.txt", encoding="utf-8")
raw_documents = loader.load()
text_splitter = CharacterTextSplitter(
    chunk_size=1,
    chunk_overlap=0,
    separator="\n"
)
documents = text_splitter.split_documents(raw_documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma.from_documents(documents, embedding=embeddings)

# Get unique categories and tones for the dropdowns
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
