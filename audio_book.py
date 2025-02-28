import os
import PyPDF2
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

class PDFToAudio:
    def __init__(self, model_name="Zyphra/Zonos-v0.1-transformer", device="cuda"):
        self.device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.model = Zonos.from_pretrained(model_name, device=self.device)
        
    def pdf_to_text(self, pdf_path, start_page=None, end_page=None):
        """Extract text from PDF file with optional page range."""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            # Validate and adjust page range
            start_page = max(1, start_page if start_page else 1)
            end_page = min(total_pages, end_page if end_page else total_pages)
            
            if start_page > end_page:
                start_page, end_page = end_page, start_page
                
            text = ""
            for page_num in tqdm(range(start_page-1, end_page), desc="Extracting text"):
                text += pdf_reader.pages[page_num].extract_text()
        return text

    def create_speaker_embedding(self, reference_audio="assets/exampleaudio.mp3"):
        """Create speaker embedding from reference audio."""
        wav, sampling_rate = torchaudio.load(reference_audio)
        # Ensure wav has correct dimensionality (2D)
        if wav.ndim < 2:
            wav = wav.unsqueeze(0)  # Add channel dimension if missing
        elif wav.ndim > 2:
            wav = wav.mean(0, keepdim=True)  # Average channels if too many
        return self.model.make_speaker_embedding(wav, sampling_rate)

    def smart_text_split(self, text, max_chars=200):
        """Split text intelligently at sentence boundaries and punctuation marks."""
        # First, try to split at sentence endings
        sentence_delimiters = '.!?'
        chunks = []
        current_chunk = ""
        
        # Remove extra whitespace and normalize line endings
        text = ' '.join(text.split())
        
        words = text.split()
        for word in words:
            test_chunk = current_chunk + (' ' if current_chunk else '') + word
            
            # Check if adding this word would exceed max length
            if len(test_chunk) > max_chars and current_chunk:
                # If current chunk ends with a sentence delimiter, add it as is
                if current_chunk[-1] in sentence_delimiters:
                    chunks.append(current_chunk)
                    current_chunk = word
                else:
                    # Look for other punctuation marks to split at
                    punct_positions = [pos for pos, char in enumerate(current_chunk) 
                                     if char in ',:;']
                    
                    if punct_positions:
                        # Split at the last punctuation mark
                        split_pos = punct_positions[-1] + 1
                        chunks.append(current_chunk[:split_pos].strip())
                        current_chunk = current_chunk[split_pos:].strip() + \
                                      (' ' if current_chunk[split_pos:].strip() else '') + word
                    else:
                        # If no punctuation found, just split at max_chars
                        chunks.append(current_chunk)
                        current_chunk = word
            else:
                current_chunk = test_chunk
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def text_to_speech(self, text, output_path, speaker_embedding, language="en-us"):
        """Convert text to speech using Zonos model."""
        # Split text into smaller chunks intelligently
        text_chunks = self.smart_text_split(text)
        
        all_wavs = []
        for chunk in tqdm(text_chunks, desc="Converting to speech"):
            if not chunk.strip():  # Skip empty chunks
                continue
            
            # Create conditioning dictionary for the chunk
            cond_dict = make_cond_dict(
                text=chunk,
                speaker=speaker_embedding,
                language=language
            )
            conditioning = self.model.prepare_conditioning(cond_dict)
            
            # Generate audio codes
            codes = self.model.generate(conditioning)
            
            # Decode to waveform
            wav = self.model.autoencoder.decode(codes).cpu()
            # Ensure wav has correct dimensionality
            if wav.ndim == 2 and wav.size(0) > 1:
                wav = wav[0:1]  # Take only the first channel
            all_wavs.append(wav[0])

        # Concatenate all wave chunks
        if all_wavs:
            final_wav = torch.cat(all_wavs, dim=-1)
            torchaudio.save(output_path, final_wav.unsqueeze(0), self.model.autoencoder.sampling_rate)
            return True
        return False

def get_pdf_info(pdf_path):
    """Get information about a PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        return len(pdf_reader.pages)

def select_pdf(pdf_files):
    """Let user select a PDF file from the list."""
    if len(pdf_files) == 1:
        return pdf_files[0]
    
    print("\nAvailable PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        total_pages = get_pdf_info(pdf)
        print(f"{i}. {pdf.name} ({total_pages} pages)")
    
    while True:
        try:
            choice = int(input("\nSelect a PDF file (enter number): "))
            if 1 <= choice <= len(pdf_files):
                return pdf_files[choice - 1]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def get_page_range(total_pages):
    """Get page range from user."""
    print(f"\nTotal pages in PDF: {total_pages}")
    
    while True:
        try:
            start = input(f"Enter start page (1-{total_pages}, press Enter for first page): ").strip()
            start = int(start) if start else 1
            
            end = input(f"Enter end page (1-{total_pages}, press Enter for last page): ").strip()
            end = int(end) if end else total_pages
            
            if 1 <= start <= total_pages and 1 <= end <= total_pages:
                return start, end
            print("Invalid page range. Please try again.")
        except ValueError:
            print("Please enter valid numbers.")

def process_pdfs():
    """Process PDFs in the input folder with user interaction."""
    input_dir = Path("input")
    output_dir = Path("output")
    
    # Create directories if they don't exist
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Get list of PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the input directory.")
        return
    
    # Initialize the converter
    converter = PDFToAudio()
    
    # Create speaker embedding from reference audio
    speaker_embedding = converter.create_speaker_embedding()
    
    # Let user select PDF file
    selected_pdf = select_pdf(pdf_files)
    print(f"\nSelected: {selected_pdf.name}")
    
    # Get total pages and page range
    total_pages = get_pdf_info(selected_pdf)
    start_page, end_page = get_page_range(total_pages)
    
    # Convert PDF to text with selected page range
    print(f"\nProcessing pages {start_page} to {end_page} from {selected_pdf.name}")
    text = converter.pdf_to_text(selected_pdf, start_page, end_page)
    
    if not text.strip():
        print(f"No text could be extracted from the selected pages of {selected_pdf.name}")
        return
    
    # Generate output path
    output_path = output_dir / f"{selected_pdf.stem}_pages_{start_page}-{end_page}.wav"
    
    # Convert text to speech
    try:
        success = converter.text_to_speech(
            text=text,
            output_path=str(output_path),
            speaker_embedding=speaker_embedding
        )
        if success:
            print(f"\nSuccessfully created audio file: {output_path.name}")
        else:
            print(f"\nFailed to create audio - no valid text chunks found")
    except Exception as e:
        print(f"\nError processing file: {str(e)}")

if __name__ == "__main__":
    print("PDF to Audio Converter using Zonos")
    print("Place your PDF files in the 'input' folder")
    process_pdfs()
