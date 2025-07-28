from datetime import datetime
import json
import os
import re
from typing import Dict, List, Any

def get_timestamp():
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat()

def safe_json_dump(data: Any, filepath: str, ensure_dirs: bool = True) -> bool:
    """Safely dump JSON data with proper encoding and error handling"""
    try:
        if ensure_dirs:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving JSON to {filepath}: {e}")
        return False

def safe_json_load(filepath: str) -> tuple[Any, bool]:
    """Safely load JSON data with error handling"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, True
    except Exception as e:
        print(f"Error loading JSON from {filepath}: {e}")
        return None, False

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0 # type: ignore
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def print_processing_stats(docs: List[Dict]) -> None:
    """Print comprehensive processing statistics"""
    if not docs:
        print("No documents to analyze")
        return
    
    total_docs = len(docs)
    total_chars = sum(len(doc.get('chunk', '')) for doc in docs)
    
    # File distribution
    file_dist = {}
    for doc in docs:
        filename = doc.get('filename', 'Unknown')
        file_dist[filename] = file_dist.get(filename, 0) + 1
    
    # Heading statistics
    heading_stats = {
        'with_headings': 0,
        'levels': {'H1': 0, 'H2': 0, 'H3': 0},
        'confidence_sum': 0.0,
        'confidence_count': 0
    }
    
    for doc in docs:
        if doc.get('has_heading'):
            heading_stats['with_headings'] += 1
            level = doc.get('heading_level', 'H3')
            if level in heading_stats['levels']:
                heading_stats['levels'][level] += 1
            
            confidence = doc.get('heading_confidence', 0.0)
            if confidence > 0:
                heading_stats['confidence_sum'] += confidence
                heading_stats['confidence_count'] += 1
    
    # Print statistics
    print(f"\n📊 PROCESSING STATISTICS")
    print(f"{'='*50}")
    print(f"Total documents processed: {total_docs}")
    print(f"Total characters: {format_file_size(total_chars)}")
    print(f"Average chunk size: {total_chars // total_docs if total_docs > 0 else 0} chars")
    
    print(f"\n📄 FILE DISTRIBUTION:")
    for filename, count in sorted(file_dist.items()):
        percentage = (count / total_docs) * 100
        print(f"  {filename}: {count} chunks ({percentage:.1f}%)")
    
    print(f"\n🏷️  HEADING STATISTICS:")
    heading_percentage = (heading_stats['with_headings'] / total_docs) * 100
    print(f"  Chunks with headings: {heading_stats['with_headings']}/{total_docs} ({heading_percentage:.1f}%)")
    print(f"  Heading levels: H1={heading_stats['levels']['H1']}, H2={heading_stats['levels']['H2']}, H3={heading_stats['levels']['H3']}")
    
    if heading_stats['confidence_count'] > 0:
        avg_confidence = heading_stats['confidence_sum'] / heading_stats['confidence_count']
        print(f"  Average heading confidence: {avg_confidence:.2f}")
    
    print(f"{'='*50}")

def validate_chunk_data(docs: List[Dict]) -> Dict[str, Any]:
    """Validate chunk data quality and return metrics"""
    validation_results = {
        'total_chunks': len(docs),
        'valid_chunks': 0,
        'issues': [],
        'quality_score': 0.0
    }
    
    for i, doc in enumerate(docs):
        chunk_issues = []
        
        # Check required fields
        required_fields = ['filename', 'page', 'chunk', 'chunk_id']
        for field in required_fields:
            if field not in doc or not doc[field]:
                chunk_issues.append(f"Missing {field}")
        
        # Check chunk quality
        chunk_text = doc.get('chunk', '')
        if chunk_text:
            if len(chunk_text) < 20:
                chunk_issues.append("Very short chunk")
            elif len(chunk_text) > 2000:
                chunk_issues.append("Very long chunk")
            
            # Check for proper text formatting
            if chunk_text.count('\n') / len(chunk_text) > 0.1:
                chunk_issues.append("Excessive line breaks")
        
        # Check heading data consistency
        if doc.get('has_heading'):
            if not doc.get('heading_text'):
                chunk_issues.append("Has heading flag but no heading text")
            if not doc.get('heading_level'):
                chunk_issues.append("Has heading but no level specified")
        
        if chunk_issues:
            validation_results['issues'].append({
                'chunk_index': i,
                'chunk_id': doc.get('chunk_id', 'Unknown'),
                'issues': chunk_issues
            })
        else:
            validation_results['valid_chunks'] += 1
    
    # Calculate quality score
    if validation_results['total_chunks'] > 0:
        validation_results['quality_score'] = validation_results['valid_chunks'] / validation_results['total_chunks']
    
    return validation_results

def print_validation_report(validation_results: Dict[str, Any]) -> None:
    """Print validation report"""
    print(f"\n🔍 VALIDATION REPORT")
    print(f"{'='*50}")
    print(f"Total chunks: {validation_results['total_chunks']}")
    print(f"Valid chunks: {validation_results['valid_chunks']}")
    print(f"Quality score: {validation_results['quality_score']:.2f}")
    
    if validation_results['issues']:
        print(f"\n⚠️  ISSUES FOUND ({len(validation_results['issues'])}):")
        for issue in validation_results['issues'][:10]:  # Show first 10 issues
            print(f"  Chunk {issue['chunk_id']}: {', '.join(issue['issues'])}")
        
        if len(validation_results['issues']) > 10:
            print(f"  ... and {len(validation_results['issues']) - 10} more issues")
    else:
        print("\n✅ No validation issues found!")
    
    print(f"{'='*50}")

def cleanup_temp_files(directories: List[str]) -> None:
    """Clean up temporary files and directories"""
    for directory in directories:
        if os.path.exists(directory):
            try:
                import shutil
                shutil.rmtree(directory)
                print(f"Cleaned up temporary directory: {directory}")
            except Exception as e:
                print(f"Warning: Could not clean up {directory}: {e}")

def ensure_output_quality(output_data: Dict) -> Dict:
    """Ensure output data meets quality standards"""
    # Ensure all text fields are properly formatted
    if 'extracted_sections' in output_data:
        for section in output_data['extracted_sections']:
            if 'section_title' in section:
                section['section_title'] = clean_output_text(section['section_title'])
            if 'heading_text' in section:
                section['heading_text'] = clean_output_text(section['heading_text'])
            if 'display_heading' in section:
                section['display_heading'] = clean_output_text(section['display_heading'])
    
    if 'subsection_analysis' in output_data:
        for analysis in output_data['subsection_analysis']:
            if 'refined_text' in analysis:
                analysis['refined_text'] = clean_output_text(analysis['refined_text'])
    
    # Ensure summary is clean
    if 'metadata' in output_data and 'summary' in output_data['metadata']:
        output_data['metadata']['summary'] = clean_output_text(output_data['metadata']['summary'])
    
    return output_data

def clean_output_text(text: str) -> str:
    """Clean text for final output"""
    if not text:
        return ""
    
    # Ensure proper Unicode normalization
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    
    # Replace problematic characters
    replacements = {
        '\x00': '',  # Null character
        '\x08': '',  # Backspace
        '\x0C': '',  # Form feed
    }
    
    for bad_char, replacement in replacements.items():
        text = text.replace(bad_char, replacement)
    
    # Remove control characters
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', '', text)
    
    # For headings, be more aggressive with special character removal
    text = re.sub(r'[^\w\s\-.,():"]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()