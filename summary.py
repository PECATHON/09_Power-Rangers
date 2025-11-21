import io
from fastapi import FastAPI, UploadFile, File
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Initialize FastAPI
app = FastAPI()

# Load Gemini API key
load_dotenv()
client = genai.Client()


# --- PDF Summarization Function ---
async def summarize_pdf(pdf_input) -> str:
    uploaded_file = None

    try:
        # 1️⃣ If input is a file path (str)
        if isinstance(pdf_input, str):
            file_to_upload = open(pdf_input, "rb")
            file_name = os.path.basename(pdf_input)
            close_file_after = True

        # 2️⃣ If input is FastAPI UploadFile (async file)
        elif hasattr(pdf_input, "read"):
            file_bytes = await pdf_input.read()
            file_to_upload = io.BytesIO(file_bytes)
            file_name = getattr(pdf_input, "filename", "uploaded_file.pdf")
            close_file_after = False

        else:
            return "Error: pdf_input must be a file path or a file-like object"

        # Upload file to Gemini
        uploaded_file = client.files.upload(
            file=file_to_upload,
            config=types.UploadFileConfig(mime_type="application/pdf")
        )

        # Generate summary using the new method: client.generate()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            # content is a list of messages or file references
            messages=[
                {"role": "user", "content": [
                    uploaded_file,
                    "Summarize this document in 3 concise bullet points."
                ]}
            ]
        )

        return response.output_text  # output_text contains the generated summary

    except Exception as e:
        return f"An API error occurred during summarization: {e}"

    finally:
        if uploaded_file:
            client.files.delete(name=uploaded_file.name)
        if 'close_file_after' in locals() and close_file_after:
            file_to_upload.close()


# --- FastAPI Endpoint ---
@app.post("/summary")
async def summary(file: UploadFile = File(...)):
    summary_value = await summarize_pdf(file)
    print(summary_value)
    return {"summary": summary_value}
