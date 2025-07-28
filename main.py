import os
import json
from config import JSON_CHUNKS
from chunker import load_and_chunk_pdfs
from embedder import get_embeddings
from utils import get_timestamp
from insight import extract_summary_enhanced
from relevance import simple_persona_job_search

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def generate_output(input_data):
    """Generate output matching the specified JSON format"""
    persona = input_data.get("persona", {}).get("role", "Unknown Persona")
    query = input_data.get("job_to_be_done", {}).get("task", "")

    if not query:
        raise ValueError("Missing 'job_to_be_done' task in input.")

    print(f"🔍 Persona: {persona}")
    print(f"🔍 Task: {query}")

    # Load or create chunks from input PDFs
    if not os.path.exists(JSON_CHUNKS):
        print("📚 Chunking PDFs from input directory...")
        docs = load_and_chunk_pdfs(pdf_folder=INPUT_DIR)
    else:
        print("📂 Loading existing chunks...")
        with open(JSON_CHUNKS, "r", encoding="utf-8") as f:
            docs = json.load(f)

    print(f"📄 Loaded {len(docs)} chunks")

    # Add embeddings
    print("🧠 Adding embeddings...")
    docs = get_embeddings(docs)

    # Perform persona-job aware search
    print("🔎 Performing persona and job-aware search...")
    extracted, refined = simple_persona_job_search(
        query=query,
        documents=docs,
        persona=persona,
        job_description=query,
        max_results=10
    )

    print(f"✅ Found {len(extracted)} relevant results")

    # Format output
    unique_files = sorted(set(doc["filename"] for doc in docs))

    extracted_sections = []
    for i, item in enumerate(extracted[:5], 1):  # Top 5 results
        extracted_sections.append({
            "document": item.get("document", ""),
            "section_title": item.get("section_title", ""),
            "importance_rank": i,
            "page_number": item.get("page_number", 0)
        })

    subsection_analysis = []
    for item in refined[:5]:  # Top 5 refined results
        subsection_analysis.append({
            "document": item.get("document", ""),
            "refined_text": item.get("refined_text", item.get("content", "")),
            "page_number": item.get("page_number", 0)
        })

    return {
        "metadata": {
            "input_documents": unique_files,
            "persona": persona,
            "job_to_be_done": query,
            "processing_timestamp": get_timestamp()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

def process_all_inputs():
    """Process all input JSON files in the input/ directory"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]

    if not input_files:
        print("⚠️ No input .json files found in the input/ directory.")
        return

    for filename in input_files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_filename = filename.replace(".json", "_output.json")
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"\n🟢 Processing {input_path} → {output_path}")

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                input_data = json.load(f)

            result = generate_output(input_data)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"✅ Output saved to: {output_path}")
            print("=" * 50)

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            print("=" * 50)

if __name__ == "__main__":
    process_all_inputs()
