
import os
from fpdf import FPDF
import re

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        # self.cell(30, 10, 'Title', 1, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

def create_pdf(md_file, pdf_file):
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)
    
    with open(md_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        if not line:
            pdf.ln(5)
            continue
            
        # Headers
        if line.startswith('# '):
            pdf.set_font('Arial', 'B', 16)
            pdf.multi_cell(0, 10, line[2:])
            pdf.set_font('Arial', '', 12)
        elif line.startswith('## '):
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, line[3:], 0, 1)
            pdf.set_font('Arial', '', 12)
        elif line.startswith('### '):
            pdf.ln(2)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, line[4:], 0, 1)
            pdf.set_font('Arial', '', 12)
            
        # Lists
        elif line.startswith('* ') or line.startswith('- '):
            pdf.set_x(20)
            # Remove bold formatting for simplicity in lists
            content = line[2:].replace('**', '') 
            pdf.multi_cell(0, 5, chr(149) + ' ' + content)
            
        # Numbered lists
        elif re.match(r'^\d+\.', line):
            pdf.set_x(20)
            content = re.sub(r'^\d+\.\s*', '', line).replace('**', '')
            number = line.split('.')[0] + '.'
            pdf.multi_cell(0, 5, number + ' ' + content)
            
        # Metadata / Separator
        elif line.startswith('---'):
            pdf.line(10, pdf.get_y(), 200, pdf.get_y())
            pdf.ln(5)
            
        # Regular text
        else:
            # Handle bold text within paragraph (simple replacement)
            # FPDF doesn't support rich text easily in standard cell/multicell without HTMLMixin
            # We will just strip ** for cleanliness or leave them. 
            # Let's strip them to look cleaner.
            clean_line = line.replace('**', '')
            pdf.multi_cell(0, 5, clean_line)
            pdf.ln(1)

    pdf.output(pdf_file)
    print(f"PDF generated: {pdf_file}")

if __name__ == "__main__":
    source_dir = "/workspace/vertical-gan-power-electronics"
    md_path = os.path.join(source_dir, "RESEARCH_SAMPLE.md")
    pdf_path = os.path.join(source_dir, "RESEARCH_SAMPLE.pdf")
    
    if os.path.exists(md_path):
        create_pdf(md_path, pdf_path)
    else:
        print(f"File not found: {md_path}")
