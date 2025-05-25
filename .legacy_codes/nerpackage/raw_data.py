import warnings
import docx

def extract_text(doc_path):
    """
    Extracts text from a .docx file.

    Args:
        doc_path (str): The path to the .docx file.

    Returns:
        str: The extracted text from the .docx file.
    """
    try:
        doc = docx.Document(doc_path)
    except Exception as e:
        warnings.warn(f"Failed to open the document: {e}", UserWarning)
        return ""

    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)

    return "\n".join(full_text)


docx_path = r'../dsai/raw_data/ruznamce_raw_text.docx'
ruznamce_text = extract_text(docx_path)

warnings.warn("This is a pending deprecation warning", DeprecationWarning)