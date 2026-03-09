import fitz

pdf_path = r"C:\Users\pramo\Downloads\ONER.pdf"
try:
    doc = fitz.open(pdf_path)
    with open("oner_full_text.txt", "w", encoding="utf-8") as f:
        for page in doc:
            f.write(page.get_text())
except Exception as e:
    print(e)
