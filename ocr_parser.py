import easyocr
import re
from datetime import datetime
from typing import Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF for PDF processing

class OCRParser:
    """OCR Parser for extracting text and invoice fields from documents"""
    
    def __init__(self):
        """Initialize EasyOCR reader"""
        try:
            self.reader = easyocr.Reader(['en'])
        except Exception as e:
            print(f"Warning: EasyOCR initialization failed: {e}")
            self.reader = None
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from PDF or image file
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text as string
        """
        try:
            if file_path.lower().endswith('.pdf'):
                return self._extract_text_from_pdf(file_path)
            else:
                return self._extract_text_from_image(file_path)
        except Exception as e:
            raise Exception(f"Text extraction failed: {str(e)}")
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            doc.close()
            
            # If PDF text extraction yields poor results, convert to image and use OCR
            if len(text.strip()) < 50:
                return self._extract_text_from_pdf_images(pdf_path)
            
            return text
            
        except Exception as e:
            # Fallback to image-based OCR
            return self._extract_text_from_pdf_images(pdf_path)
    
    def _extract_text_from_pdf_images(self, pdf_path: str) -> str:
        """Extract text from PDF by converting pages to images"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(min(3, len(doc))):  # Process first 3 pages max
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Convert to numpy array for EasyOCR
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Extract text using OCR
                if self.reader:
                    results = self.reader.readtext(img)
                    page_text = " ".join([result[1] for result in results])
                    text += page_text + " "
            
            doc.close()
            return text
            
        except Exception as e:
            raise Exception(f"PDF OCR extraction failed: {str(e)}")
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image file using EasyOCR"""
        try:
            if not self.reader:
                raise Exception("EasyOCR not initialized")
            
            # Read and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Could not read image file")
            
            # Preprocessing for better OCR results
            img = self._preprocess_image(img)
            
            # Extract text
            results = self.reader.readtext(img)
            text = " ".join([result[1] for result in results])
            
            return text
            
        except Exception as e:
            raise Exception(f"Image OCR extraction failed: {str(e)}")
    
    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply threshold to get better contrast
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def parse_invoice_fields(self, text: str) -> Dict[str, Any]:
        """
        Parse invoice fields from extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Dictionary with parsed invoice fields
        """
        # Clean text
        text = self._clean_text(text)
        
        # Extract fields
        fields = {
            "invoice_number": self._extract_invoice_number(text),
            "vendor": self._extract_vendor_name(text),
            "date": self._extract_invoice_date(text),
            "amount": self._extract_amount(text)
        }
        
        return fields
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with parsing
        text = re.sub(r'[^\w\s\-\.\,\:\$\(\)\/]', ' ', text)
        
        return text.strip()
    
    def _extract_invoice_number(self, text: str) -> Optional[str]:
        """Extract invoice number from text"""
        patterns = [
            r'invoice\s*(?:no|number|#)\s*:?\s*([A-Z0-9\-]+)',
            r'inv\s*(?:no|#)\s*:?\s*([A-Z0-9\-]+)',
            r'bill\s*(?:no|number|#)\s*:?\s*([A-Z0-9\-]+)',
            r'(?:^|\s)([A-Z]{2,4}[-]?\d{3,8})(?:\s|$)',
            r'#\s*([A-Z0-9\-]{4,15})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_vendor_name(self, text: str) -> Optional[str]:
        """Extract vendor name from text"""
        patterns = [
            r'(?:from|vendor|supplier|company)\s*:?\s*([A-Za-z\s&\.,]{3,50})',
            r'bill\s*to\s*:?\s*([A-Za-z\s&\.,]{3,50})',
            r'^([A-Z][A-Za-z\s&\.,]{10,40})(?:\n|\r)',
            r'([A-Z][A-Za-z\s&\.]{5,30})\s*(?:ltd|inc|corp|llc|pvt)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                vendor = match.group(1).strip()
                # Clean vendor name
                vendor = re.sub(r'\s+', ' ', vendor)
                vendor = vendor.strip('.,')
                if len(vendor) > 3:
                    return vendor
        
        return None
    
    def _extract_invoice_date(self, text: str) -> Optional[str]:
        """Extract invoice date from text"""
        patterns = [
            r'(?:invoice\s*date|date|dated)\s*:?\s*(\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4})',
            r'(\d{1,2}[-\/]\d{1,2}[-\/]\d{2,4})',
            r'(\d{2,4}[-\/]\d{1,2}[-\/]\d{1,2})',
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4})',
            r'((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{2,4})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Try to parse and standardize date
                    date_str = self._standardize_date(match)
                    if date_str:
                        return date_str
                except:
                    continue
        
        return None
    
    def _standardize_date(self, date_str: str) -> Optional[str]:
        """Standardize date format to YYYY-MM-DD"""
        date_formats = [
            '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%m-%d-%Y', '%Y-%m-%d',
            '%d/%m/%y', '%m/%d/%y', '%y/%m/%d',
            '%d-%m-%y', '%m-%d-%y', '%y-%m-%d',
            '%d %b %Y', '%b %d, %Y', '%d %B %Y', '%B %d, %Y'
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def _extract_amount(self, text: str) -> Optional[float]:
        """Extract invoice amount from text"""
        patterns = [
            r'(?:total|amount|sum|grand\s*total)\s*:?\s*\$?\s*([\d,]+\.?\d*)',
            r'(?:total|amount)\s*(?:due|payable)\s*:?\s*\$?\s*([\d,]+\.?\d*)',
            r'\$\s*([\d,]+\.?\d*)',
            r'(?:^|\s)([\d,]{3,}\.?\d{0,2})(?:\s|$)',
            r'(?:rs|inr|usd|eur|gbp)\.?\s*([\d,]+\.?\d*)'
        ]
        
        amounts = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean amount string
                    amount_str = re.sub(r'[,$]', '', match)
                    amount = float(amount_str)
                    
                    # Filter reasonable amounts (between $1 and $1M)
                    if 1 <= amount <= 1000000:
                        amounts.append(amount)
                except ValueError:
                    continue
        
        # Return the largest amount found (likely the total)
        return max(amounts) if amounts else None
