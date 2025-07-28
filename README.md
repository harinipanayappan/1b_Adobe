Persona-Based PDF Insight Extractor
===================================

This project processes a persona and job description from a JSON file and analyzes PDF documents to extract relevant and meaningful content using heading-aware chunking, semantic embeddings, and a relevance classifier.

-----------------------------------
Project Structure
-----------------------------------
.
├── Dockerfile
├── requirements.txt
├── main.py
├── config.py
├── chunker.py
├── embedder.py
├── insight.py
├── relevance.py
├── utils.py
├── input/
│   └── input.json         --> Input file with persona and task
├── data/
│   └── *.pdf              --> PDF documents to be processed
├── output/
│   └── output.json        --> Final results will be saved here

-----------------------------------
Prerequisites
-----------------------------------
- Docker must be installed on your machine.
- Place PDF documents inside the `data/` directory.
- Prepare `input/input.json` with the required persona and task.

-----------------------------------
Step 1: Prepare requirements.txt
-----------------------------------
Make sure `requirements.txt` contains these (and any other) libraries:

langchain
openai
tiktoken
numpy
scikit-learn
pdfplumber

-----------------------------------
Step 2: Build Docker Image
-----------------------------------
Run this command from the project root:

docker build -t persona-insights:latest .

-----------------------------------
Step 3: Run the Docker Container
-----------------------------------

For Linux/macOS:
-----------------
docker run --rm \
  -v "$(pwd)/input:/app/input" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/data:/app/data" \
  --network none \
  persona-insights:latest

For Windows PowerShell:
------------------------
docker run --rm `
  -v "${PWD}\input:/app/input" `
  -v "${PWD}\output:/app/output" `
  -v "${PWD}\data:/app/data" `
  --network none `
  persona-insights:latest

-----------------------------------
Output
-----------------------------------
- Final result saved to: `output/output.json`
- Includes: 
  - Metadata
  - Extracted relevant sections
  - Summary
  - Page numbers

-----------------------------------
Sample input/input.json
-----------------------------------
{
  "persona": {
    "role": "Travel Planner"
  },
  "job_to_be_done": {
    "task": "Plan a trip of 4 days for a group of 10 college friends."
  }
}

-----------------------------------
What This Does
-----------------------------------
✓ Loads PDFs from `data/`
✓ Chunks with heading detection
✓ Embeds content using vector models
✓ Filters content based on persona + job
✓ Returns structured results and summary


