import PyPDF2

def extract_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

# Extract text from the PDF
pdf_text = extract_pdf_text('Project Problem Statement Employee Sentiment Analysis.pdf')
print(pdf_text)
