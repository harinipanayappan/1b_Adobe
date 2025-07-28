import os
import json
from config import JSON_CHUNKS
from chunker import load_and_chunk_pdfs
from embedder import get_embeddings
from utils import get_timestamp
from insight import extract_summary_enhanced
from relevance import simple_persona_job_search

# Directories (all inputs including PDFs and JSONs are in 'input/')
INPUT_DIR = "input"
OUTPUT_DIR = "output"

def generate_output(input_data, debug_similarities=False):
    persona = input_data.get("persona", {}).get("role", "Unknown Persona")
    query = input_data.get("job_to_be_done", {}).get("task", "")

    if not query:
        raise ValueError("Missing 'job_to_be_done' task in input.")

    print(f"Processing with persona: {persona}")
    print(f"Job to be done: {query}")

    if not os.path.exists(JSON_CHUNKS):
        print("Loading and chunking PDFs from 'input/'...")
        docs = load_and_chunk_pdfs(pdf_folder=INPUT_DIR)
    else:
        print("Loading existing heading-aware chunks...")
        with open(JSON_CHUNKS, "r", encoding="utf-8") as f:
            docs = json.load(f)

    # PDF and heading statistics
    pdf_distribution = {}
    heading_stats = {
        'total_chunks': len(docs),
        'chunks_with_headings': 0,
        'avg_heading_confidence': 0.0,
        'heading_levels': {'H1': 0, 'H2': 0, 'H3': 0}
    }

    confidence_sum = 0.0
    confidence_count = 0

    for doc in docs:
        pdf_name = doc['filename']
        pdf_distribution[pdf_name] = pdf_distribution.get(pdf_name, 0) + 1

        if doc.get('has_heading'):
            heading_stats['chunks_with_headings'] += 1
            level = doc.get('heading_level', 'H3')
            if level in heading_stats['heading_levels']:
                heading_stats['heading_levels'][level] += 1
            confidence = doc.get('heading_confidence', 0.0)
            if confidence > 0:
                confidence_sum += confidence
                confidence_count += 1

    if confidence_count > 0:
        heading_stats['avg_heading_confidence'] = confidence_sum / confidence_count

    print(f"Loaded {len(docs)} chunks from {len(pdf_distribution)} documents")
    print("Heading stats:", heading_stats)

    print("Generating embeddings...")
    docs = get_embeddings(docs)

    print("Searching relevant content based on persona and task...")
    extracted, refined = simple_persona_job_search(
        query=query,
        documents=docs,
        persona=persona,
        job_description=query,
        max_results=10
    )

    print(f"Found {len(extracted)} relevant chunks")

    summary = extract_summary_enhanced(refined) if refined else "No relevant content found for the given query."

    unique_files = sorted(set(doc["filename"] for doc in docs))
    files_with_results = sorted(set(item["document"] for item in extracted))

    classifier_info = {}
    if extracted and 'debug_info' in extracted[0]:
        classifier_info = extracted[0]['debug_info']
        del extracted[0]['debug_info']

    result_heading_stats = {
        'results_with_headings': 0,
        'heading_level_distribution': {},
        'avg_heading_confidence_in_results': 0.0
    }

    heading_conf_sum = 0.0
    heading_conf_count = 0

    for item in extracted:
        if item.get('is_heading_based'):
            result_heading_stats['results_with_headings'] += 1
            level = item.get('heading_level', 'Unknown')
            result_heading_stats['heading_level_distribution'][level] = \
                result_heading_stats['heading_level_distribution'].get(level, 0) + 1
            conf = item.get('heading_confidence', 0.0)
            if conf > 0:
                heading_conf_sum += conf
                heading_conf_count += 1

    if heading_conf_count > 0:
        result_heading_stats['avg_heading_confidence_in_results'] = heading_conf_sum / heading_conf_count

    return {
        "metadata": {
            "input_documents": unique_files,
            "documents_with_results": files_with_results,
            "total_chunks_processed": len(docs),
            "relevant_chunks_found": len(extracted),
            "chunks_per_pdf": pdf_distribution,
            "persona": persona,
            "job_to_be_done": query,
            "processing_timestamp": get_timestamp(),
            "summary": summary,
            "classifier_info": classifier_info,
            "processing_method": "enhanced_persona_job_heading_aware_search",
            "heading_statistics": heading_stats,
            "result_heading_statistics": result_heading_stats
        },
        "extracted_sections": extracted,
        "subsection_analysis": refined
    }

def process_all_inputs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".json"):
            input_path = os.path.join(INPUT_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}_output.json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            print(f"\n🔍 Processing {input_path} → {output_path}")

            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    input_data = json.load(f)
            except Exception as e:
                print(f"❌ Failed to read {input_path}: {e}")
                continue

            try:
                result = generate_output(input_data)
            except Exception as e:
                print(f"❌ Error during processing of {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue

            try:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"✅ Output saved to: {output_path}")
            except Exception as e:
                print(f"❌ Failed to save output: {e}")

if __name__ == "__main__":
    process_all_inputs()