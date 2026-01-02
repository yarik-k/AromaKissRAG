#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def install_requirements():
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")
        return False

def check_files():
    print("Checking required files...")
    
    required_files = ["messages_simple_list.json", "aromakiss_rag_bot.py"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
            print(f"Missing: {file}")
        else:
            print(f"Found: {file}")
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    
    return True

def test_imports():
    print("🧪 Testing imports...")
    
    modules = [
        "openai",
        "sentence_transformers", 
        "sklearn",
        "numpy",
        "torch",
        "transformers"
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"{module}")
        except ImportError as e:
            print(f"{module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        return False
    
    return True

def quick_test():
    print("🚀 Quick system test...")
    
    try:
        from aromakiss_rag_bot import AromaKissRAG
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("OPENAI_API_KEY not set. Using placeholder for initialization test.")
        
        print("Initializing RAG system...")
        rag_bot = AromaKissRAG(api_key)
        
        print(f"Loaded {len(rag_bot.posts)} posts")
        print(f"Created embeddings shape: {rag_bot.embeddings.shape}")
        
        print("Testing post retrieval...")
        similar_posts = rag_bot._retrieve_similar_posts("свечи", 3)
        print(f"Retrieved {len(similar_posts)} similar posts")
        
        print("System test passed!")
        return True
        
    except Exception as e:
        print(f"System test failed: {e}")
        return False

def main():
    print("RAG Bot Setup")
    print("=" * 40)
    
    if not check_files():
        print("\nSetup failed: Missing required files")
        return
    
    if not install_requirements():
        print("\nSetup failed: Could not install requirements")
        return
    
    if not test_imports():
        print("\nSetup failed: Import errors")
        return
    
    if not quick_test():
        print("\nSetup failed: System test failed")
        return
    
    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main() 