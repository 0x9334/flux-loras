#!/usr/bin/env python3
"""
Simple test script for the Flux Dynamic LoRA server
"""

import requests
import json
import base64
from PIL import Image
import io

# Server configuration
SERVER_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{SERVER_URL}/health-check")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_lora_cfgs():
    """Test the LoRA configurations endpoint"""
    print("\nTesting LoRA configurations...")
    try:
        response = requests.get(f"{SERVER_URL}/lora-cfgs")
        print(f"Status Code: {response.status_code}")
        print(f"Available LoRAs: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_generate_simple():
    """Test the generate endpoint with a simple request"""
    print("\nTesting image generation (simple)...")
    
    payload = {
        "prompt": "A beautiful sunset over mountains",
        "guidance_scale": 3.5,
        "num_inference_steps": 20,
        "seed": 42,
        "height": 512,
        "width": 512,
        "lora_cfg": {
            "frazetta_lora": True,
            "naked_woman_lora": False,
            "saggy_breasts_lora": False
        }
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/generate", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "image" in result:
                # Decode and save the image
                image_data = base64.b64decode(result["image"])
                image = Image.open(io.BytesIO(image_data))
                image.save("test_output.png")
                print("✅ Image generated successfully and saved as 'test_output.png'")
                print(f"Image size: {image.size}")
            else:
                print("❌ No image in response")
                print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False
x
def test_generate_v1():
    """Test the v1 generate endpoint"""
    print("\nTesting v1 image generation...")
    
    payload = {
        "prompt": "A majestic dragon flying over a medieval castle",
        "guidance_scale": 4.0,
        "num_inference_steps": 25,
        "seed": 123,
        "height": 768,
        "width": 768,
        "lora_cfg": {
            "frazetta_lora": False,
            "naked_woman_lora": False,
            "saggy_breasts_lora": False
        }
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/v1/generate", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "image" in result:
                # Decode and save the image
                image_data = base64.b64decode(result["image"])
                image = Image.open(io.BytesIO(image_data))
                image.save("test_output_v1.png")
                print("✅ V1 Image generated successfully and saved as 'test_output_v1.png'")
                print(f"Image size: {image.size}")
            else:
                print("❌ No image in response")
                print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_generate_with_all_loras():
    """Test generation with all LoRAs enabled"""
    print("\nTesting image generation with all LoRAs...")
    
    payload = {
        "prompt": "A fantasy warrior",
        "guidance_scale": 3.5,
        "num_inference_steps": 20,
        "seed": 999,
        "height": 512,
        "width": 512,
        "lora_cfg": {
            "frazetta_lora": True,
            "naked_woman_lora": True,
            "saggy_breasts_lora": True
        }
    }
    
    try:
        response = requests.post(f"{SERVER_URL}/generate", json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if "image" in result:
                # Decode and save the image
                image_data = base64.b64decode(result["image"])
                image = Image.open(io.BytesIO(image_data))
                image.save("test_output_all_loras.png")
                print("✅ Image with all LoRAs generated successfully and saved as 'test_output_all_loras.png'")
                print(f"Image size: {image.size}")
            else:
                print("❌ No image in response")
                print(f"Response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Flux Dynamic LoRA Server Tests")
    print("=" * 50)
    
    # Test basic endpoints
    health_ok = test_health_check()
    lora_ok = test_lora_cfgs()
    
    if not health_ok:
        print("❌ Server health check failed. Make sure the server is running!")
        return
    
    # Test generation endpoints
    simple_ok = test_generate_simple()
    v1_ok = test_generate_v1()
    all_loras_ok = test_generate_with_all_loras()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"LoRA Configs: {'✅ PASS' if lora_ok else '❌ FAIL'}")
    print(f"Simple Generation: {'✅ PASS' if simple_ok else '❌ FAIL'}")
    print(f"V1 Generation: {'✅ PASS' if v1_ok else '❌ FAIL'}")
    print(f"All LoRAs Generation: {'✅ PASS' if all_loras_ok else '❌ FAIL'}")
    
    if all([health_ok, lora_ok, simple_ok, v1_ok, all_loras_ok]):
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()