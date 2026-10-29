import os
import json
from pprint import pprint

# Import the Phase 1 and Phase 2 entry points
from ingest import process_lecture_pdf
from router import route_lecture_content

def run_end_to_end_test():
    test_pdf_path = "test_lecture.pdf"
    
    if not os.path.exists(test_pdf_path):
        print(f"[ERROR] Could not find {test_pdf_path}. Please create it first.")
        return

    print("==================================================")
    print(" PHASE 1: INGESTION & EXTRACTION")
    print("==================================================")
    
    try:
        # This will hash the PDF, check SQLite, and call Gemini 2.5 Pro Vision
        ingest_result = process_lecture_pdf(test_pdf_path)
        
        file_hash = ingest_result["file_hash"]
        status = ingest_result["status"]
        extracted_text = ingest_result["extracted_text"]
        
        print(f"\n[SUCCESS] File Hash: {file_hash}")
        print(f"[SUCCESS] Status: {status}")
        
        if status == "PROCESSING":
            print("[WARNING] The file is actively locked by another process.")
            return
            
        if not extracted_text:
            print("[ERROR] No text was extracted.")
            return
            
        print("\n--- Extracted Markdown/LaTeX ---")
        print(extracted_text)
        print("--------------------------------\n")
        
    except Exception as e:
        print(f"\n[PHASE 1 FAILED] {e}")
        return

    # If the file was already routed in a previous test run, we don't need to re-route it
    if ingest_result.get("taxonomy") is not None:
        print("==================================================")
        print(" PHASE 2: ALREADY ROUTED (SERVED FROM CACHE)")
        print("==================================================")
        print(f"Classification: {ingest_result['taxonomy']}")
        print("\nRouting Payload:")
        pprint(json.loads(ingest_result["routing_payload"]))
        return

    print("==================================================")
    print(" PHASE 2: META-AGENT TAXONOMY ROUTING")
    print("==================================================")
    
    try:
        # This will pass the text to Gemini 2.5 Flash and return Pydantic JSON
        routing_result = route_lecture_content(file_hash, extracted_text)
        
        print(f"\n[SUCCESS] Classification: {routing_result.get('classification')}")
        print("\n--- Full Routing Payload ---")
        pprint(routing_result)
        print("--------------------------------\n")
        
    except Exception as e:
        print(f"\n[PHASE 2 FAILED] {e}")
        return

if __name__ == "__main__":
    run_end_to_end_test()