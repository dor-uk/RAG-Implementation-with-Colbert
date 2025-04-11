import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text() or ""
    return text

def process_pdf_directory(directory_path):
    """Processes all PDFs in the given directory and extracts text."""
    documents = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            documents[filename] = extract_text_from_pdf(file_path)
    return documents

def merge_short_chunks(chunks, min_length=50):
    """Merges small chunks to avoid very short segments."""
    merged_chunks = []
    buffer = ""

    for chunk in chunks:
        if len(chunk) < min_length:
            buffer += " " + chunk  # Append small chunk to buffer
        else:
            if buffer:
                chunk = buffer + " " + chunk  # Merge buffer with the current chunk
                buffer = ""
            merged_chunks.append(chunk)

    # Append any remaining buffer as a chunk
    if buffer:
        merged_chunks.append(buffer)

    return merged_chunks


def re_split_long_chunks(chunks, max_length=400):
    """Reapplies text splitting on chunks that exceed max_length."""
    refined_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,  
        chunk_overlap=100,
        separators=["\n\n", ". ", "! ", "? ", ":", "\n"],  # Keeps sentence coherence
    )

    for chunk in chunks:
        if len(chunk) > max_length:
            new_subchunks = text_splitter.split_text(chunk)
            refined_chunks.extend(new_subchunks)  # Add newly split chunks
        else:
            refined_chunks.append(chunk)

    return refined_chunks

def chunk_documents(documents, chunk_size=324, chunk_overlap=150, min_chunk_length=50, max_length=350):
    """Splits documents into chunks and ensures all chunks fit within max_length."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ". ", "! ", "? ", ":"],
    )
    
    processed_docs = []
    for doc_name, content in documents.items():
        chunks = text_splitter.split_text(content)
        chunks = merge_short_chunks(chunks, min_length=min_chunk_length)  # Merge small chunks
        chunks = re_split_long_chunks(chunks, max_length=max_length)  # Reapply splitting if needed

        for i, chunk in enumerate(chunks):
            processed_docs.append({
                "doc_name": doc_name,
                "chunk_id": i,
                "content": chunk,
                "chunk_size": len(chunk)
            })
    
    return processed_docs

def save_to_json(processed_docs, output_path="chunked_documents.json"):
    """Saves processed chunks into a JSON file."""
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(processed_docs, json_file, ensure_ascii=False, indent=4)
    print(f"Results saved to {output_path}")

# Example usage
directory_path = "./"  # Change to your actual PDF directory
documents = process_pdf_directory(directory_path)
chunked_docs = chunk_documents(documents)
save_to_json(chunked_docs)
