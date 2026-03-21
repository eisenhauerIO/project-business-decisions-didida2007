import pdfplumber

p = pdfplumber.open('AngristLavy1999.pdf')
print(f"Total pages: {len(p.pages)}")
for i in range(9, 16):
    print(f"\n{'='*60}")
    print(f"PAGE {i+1}")
    print('='*60)
    text = p.pages[i].extract_text()
    if text:
        print(text)
