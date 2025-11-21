import io
from fastapi import UploadFile
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def summarize_pdf(pdf_input) -> str:
    """
    Summarize a PDF document using Google Gemini API.
    
    Args:
        pdf_input: Either a file path (str) or UploadFile object
    
    Returns:
        str: Summary text or error message
    """
    uploaded_file = None
    file_to_upload = None
    close_file_after = False
    
    try:
        # Prepare file based on input type
        if isinstance(pdf_input, str):
            # File path provided
            file_to_upload = open(pdf_input, "rb")
            close_file_after = True
        elif hasattr(pdf_input, "read"):
            # UploadFile or file-like object
            file_bytes = await pdf_input.read()
            file_to_upload = io.BytesIO(file_bytes)
            file_to_upload.seek(0)
            close_file_after = True
        else:
            return "Error: pdf_input must be a file path or a file-like object"
        
        # Upload file to Gemini
        print("Uploading PDF to Gemini...")
        uploaded_file = client.files.upload(
            file=file_to_upload,
            config=types.UploadFileConfig(mime_type="application/pdf")
        )
        print(f"File uploaded: {uploaded_file.name}")
        
        # Generate summary
        print("Generating summary...")
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[uploaded_file, "Summarize this document in 3 concise bullet points."]
        )
        
        # Extract text from response
        # The response object has a 'text' property directly
        if hasattr(response, 'text'):
            summary_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            # Fallback: extract from candidates
            summary_text = response.candidates[0].content.parts[0].text
        else:
            return "Error: Unable to extract text from Gemini response"
        
        print("Summary generated successfully")
        return summary_text.strip()
        
    except AttributeError as e:
        # Debug information for attribute errors
        print(f"AttributeError: {e}")
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        
        # Try alternative extraction methods
        try:
            if hasattr(response, 'candidates'):
                summary_text = response.candidates[0].content.parts[0].text
                return summary_text.strip()
        except:
            pass
        
        return f"Error: Could not extract summary from response - {e}"
        
    except Exception as e:
        import traceback
        print(f"Summarization error: {e}")
        traceback.print_exc()
        return f"An API error occurred during summarization: {e}"
        
    finally:
        # Cleanup uploaded file from Gemini
        if uploaded_file:
            try:
                print(f"Deleting uploaded file: {uploaded_file.name}")
                client.files.delete(name=uploaded_file.name)
            except Exception as e:
                print(f"Warning: Could not delete uploaded file: {e}")
        
        # Close local file handle
        if file_to_upload and close_file_after:
            try:
                file_to_upload.close()
            except:
                pass


# Test function for debugging
async def test_summarize(pdf_path: str):
    """Test function to debug summarization with a local PDF."""
    result = await summarize_pdf(pdf_path)
    print("\n=== Summary Result ===")
    print(result)
    print("======================\n")
    return result


if __name__ == "__main__":
    import asyncio
    
    # Test with a local PDF file
    test_pdf = "test.pdf"  # Replace with your test PDF path
    if os.path.exists(test_pdf):
        asyncio.run(test_summarize(test_pdf))
    else:
        print(f"Test PDF not found: {test_pdf}")