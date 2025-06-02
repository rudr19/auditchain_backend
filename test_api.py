"""
Test script for the AuditChain Invoice Fraud Detection API
Run this to test all endpoints
"""

import requests
import json
import pandas as pd
from io import StringIO
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print("\n🔍 Testing model info endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Model info test failed: {e}")
        return False

def create_sample_csv():
    """Create a sample CSV for training"""
    data = {
        'amount': [1500.00, 2500.50, 75000.00, 45.99, 3200.00, 125000.00, 800.25, 5.00],
        'vendor': ['ABC Corp', 'XYZ Limited', 'Suspicious Vendor', 'Tiny Co', 'Normal Business', 'Big Spender Inc', 'Regular Store', 'Micro Biz'],
        'date': ['2024-01-15', '2024-01-20', '2024-01-25', '2024-01-30', '2024-02-05', '2024-02-10', '2024-02-15', '2024-02-20'],
        'invoice_number': ['INV001', 'INV002', 'INV003', 'INV004', 'INV005', 'INV006', 'INV007', 'INV008']
    }
    
    df = pd.DataFrame(data)
    csv_string = df.to_csv(index=False)
    
    # Save to file for testing
    with open('test_training_data.csv', 'w') as f:
        f.write(csv_string)
    
    return 'test_training_data.csv'

def test_model_training():
    """Test the model training endpoint"""
    print("\n🔍 Testing model training endpoint...")
    
    try:
        # Create sample CSV
        csv_file = create_sample_csv()
        
        # Upload CSV for training
        with open(csv_file, 'rb') as f:
            files = {'file': ('training_data.csv', f, 'text/csv')}
            response = requests.post(f"{BASE_URL}/train", files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Clean up
        import os
        os.remove(csv_file)
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Model training test failed: {e}")
        return False

def create_sample_invoice_image():
    """Create a simple invoice image for testing"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Create a simple invoice image
        img = Image.new('RGB', (600, 800), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            title_font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
        
        # Add invoice content
        draw.text((50, 50), "INVOICE", fill='black', font=title_font)
        draw.text((50, 100), "Invoice Number: INV-2024-001", fill='black', font=font)
        draw.text((50, 130), "Date: 2024-03-15", fill='black', font=font)
        draw.text((50, 160), "Vendor: Test Company Ltd", fill='black', font=font)
        draw.text((50, 200), "Amount: $2,500.00", fill='black', font=font)
        draw.text((50, 250), "Description: Professional Services", fill='black', font=font)
        draw.text((50, 300), "Thank you for your business!", fill='black', font=font)
        
        # Save image
        img.save('test_invoice.png')
        return 'test_invoice.png'
        
    except ImportError:
        print("⚠️ PIL not available, creating text file instead...")
        # Create a text file as fallback
        with open('test_invoice.txt', 'w') as f:
            f.write("""
            INVOICE
            Invoice Number: INV-2024-001
            Date: 2024-03-15
            Vendor: Test Company Ltd
            Amount: $2,500.00
            Description: Professional Services
            Thank you for your business!
            """)
        return 'test_invoice.txt'

def test_upload_endpoint():
    """Test the upload endpoint"""
    print("\n🔍 Testing upload endpoint...")
    
    try:
        # Create sample invoice
        invoice_file = create_sample_invoice_image()
        
        # Upload invoice
        with open(invoice_file, 'rb') as f:
            files = {'file': (invoice_file, f, 'image/png')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("📊 Fraud Detection Results:")
            print(f"  Invoice Number: {result.get('invoice_number', 'Not found')}")
            print(f"  Vendor: {result.get('vendor', 'Not found')}")
            print(f"  Date: {result.get('date', 'Not found')}")
            print(f"  Amount: ${result.get('amount', 0):,.2f}")
            print(f"  Anomaly Score: {result.get('anomaly_score', 0):.3f}")
            print(f"  Is Fraud: {result.get('is_fraud', False)}")
            print(f"  Confidence: {result.get('confidence', 'Unknown')}")
            print(f"  Reason: {result.get('reason', 'No reason provided')}")
        else:
            print(f"Response: {response.text}")
        
        # Clean up
        import os
        os.remove(invoice_file)
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Upload test failed: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    print("🚀 Starting AuditChain API Tests...\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Model Training", test_model_training),
        ("Invoice Upload", test_upload_endpoint)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results[test_name] = result
        
        if result:
            print(f"✅ {test_name} - PASSED")
        else:
            print(f"❌ {test_name} - FAILED")
        
        # Add delay between tests
        time.sleep(1)
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the API server and try again.")

if __name__ == "__main__":
    run_all_tests()
