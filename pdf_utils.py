"""
PDF Utility Functions for Duplicate Detection
"""

def get_pdf_minhash(file_path):
    """
    Calculate MinHash for PDF file content
    
    Args:
        file_path (str): Path to PDF file
        
    Returns:
        MinHash: MinHash object for the PDF
    """
    try:
        from PyPDF2 import PdfReader
        from datasketch import MinHash
        
        # Read PDF
        reader = PdfReader(file_path)
        
        # Extract text from all pages
        text = ""
        for page in reader.pages:
             page_text = page.extract_text()
             if page_text:
                text += page_text
        
        # Create MinHash
        m = MinHash(num_perm=128)
        
        # Add words to MinHash
        words = text.lower().split()
        for word in words:
            m.update(word.encode('utf-8'))
        
        return m
        
    except ImportError:
        print("Missing library: Install with 'pip install PyPDF2 datasketch'")
        # Return empty MinHash if library not available
        try:
            from datasketch import MinHash
            return MinHash(num_perm=128)
        except:
            return None
    except Exception as e:
        print(f"Error processing PDF: {e}")
        try:
            from datasketch import MinHash
            return MinHash(num_perm=128)
        except:
            return None


def get_pdf_similarity(file1_path, file2_path):
    """
    Calculate similarity between two PDF files
    
    Args:
        file1_path (str): Path to first PDF file
        file2_path (str): Path to second PDF file
        
    Returns:
        float: Similarity score (0-100)
    """
    try:
        hash1 = get_pdf_minhash(file1_path)
        hash2 = get_pdf_minhash(file2_path)
        
        if hash1 and hash2:
            similarity = hash1.jaccard(hash2) * 100
            return similarity
        return 0.0
        
    except Exception as e:
        print(f"Error calculating PDF similarity: {e}")
        return 0.0


def extract_text_from_pdf(file_path):
    """
    Extract text content from PDF file
    
    Args:
        file_path (str): Path to PDF file
        
    Returns:
        str: Extracted text
    """
    try:
        from PyPDF2 import PdfReader
        
        reader = PdfReader(file_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text()
        
        return text
        
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""