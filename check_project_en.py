"""
Script to check project readiness for submission
Checks presence of all required files and components
"""

import os
from pathlib import Path


def check_file(filepath, critical=True):
    """Check if file exists"""
    exists = os.path.exists(filepath)
    status = "?" if exists else ("?" if critical else "??")
    filename = Path(filepath).name
    print(f"{status} {filename}")
    return exists


def check_directory(dirpath, critical=True):
    """Check if directory exists"""
    exists = os.path.exists(dirpath) and os.path.isdir(dirpath)
    status = "?" if exists else ("?" if critical else "??")
    dirname = Path(dirpath).name
    print(f"{status} {dirname}/")
    return exists


def check_project():
    """Check project readiness"""
    
    print("="*60)
    print("PROJECT READINESS CHECK")
    print("="*60)
    print()
    
    all_good = True
    
    # Main scripts
    print("Main scripts:")
    all_good &= check_file("01_prepare_data.py")
    all_good &= check_file("02_train_tokenizer.py")
    all_good &= check_file("03_train.py")
    all_good &= check_file("model.py")
    all_good &= check_file("config.py")
    all_good &= check_file("utils.py")
    print()
    
    # Demo and testing
    print("Demo and testing:")
    all_good &= check_file("demo_inference.ipynb")
    all_good &= check_file("quick_test.py")
    check_file("app.py", critical=False)  # Optional
    print()
    
    # Documentation
    print("Documentation:")
    all_good &= check_file("README.md")
    check_file("QUICKSTART.md", critical=False)
    check_file("ARCHITECTURE.md", critical=False)
    check_file("EXAMPLES.md", critical=False)
    check_file("SUBMISSION.md", critical=False)
    print()
    
    # Configuration
    print("Configuration:")
    all_good &= check_file("requirements.txt")
    all_good &= check_file(".gitignore")
    check_file("run_pipeline.sh", critical=False)
    print()
    
    # Generated data
    print("Generated data:")
    has_data = check_directory("data/processed", critical=False)
    has_tokenizer = check_directory("tokenizer", critical=False)
    has_model = check_directory("models", critical=False)
    
    if has_data:
        check_file("data/processed/train.txt", critical=False)
        check_file("data/processed/val.txt", critical=False)
    
    if has_tokenizer:
        check_file("tokenizer/tokenizer.json", critical=False)
        check_file("tokenizer/config.json", critical=False)
    
    if has_model:
        check_file("models/best_model.pt", critical=False)
        check_file("models/final_model.pt", critical=False)
    
    print()
    
    # Summary
    print("="*60)
    print("CHECK SUMMARY")
    print("="*60)
    print()
    
    if all_good:
        print("? All critical files are present!")
        print()
    else:
        print("? Some critical files are missing!")
        print("   Check files marked with ? above.")
        print()
    
    # Check trained model
    if has_model and os.path.exists("models/best_model.pt"):
        print("? Model is trained and saved")
        
        model_size = os.path.getsize("models/best_model.pt") / (1024 * 1024)
        print(f"   Model size: {model_size:.1f} MB")
        
        if model_size < 10:
            print("   ??  Model seems too small. Make sure training completed.")
        
        print()
    else:
        print("??  Model not trained yet")
        print("   Run: python 03_train.py")
        print()
    
    # Check tokenizer
    if has_tokenizer and os.path.exists("tokenizer/tokenizer.json"):
        print("? Tokenizer is trained")
        print()
    else:
        print("??  Tokenizer not trained yet")
        print("   Run: python 02_train_tokenizer.py")
        print()
    
    # Check data
    if has_data and os.path.exists("data/processed/train.txt"):
        print("? Data is prepared")
        
        data_size = os.path.getsize("data/processed/train.txt") / (1024 * 1024)
        print(f"   train.txt size: {data_size:.1f} MB")
        
        if data_size < 1:
            print("   ??  Data seems too small. Recommend at least 10-100 MB.")
        
        print()
    else:
        print("??  Data not prepared yet")
        print("   Run: python 01_prepare_data.py")
        print()
    
    # Submission checklist
    print("="*60)
    print("SUBMISSION CHECKLIST")
    print("="*60)
    print()
    
    checklist = [
        ("Code uploaded to GitHub", False),
        ("README.md filled", os.path.exists("README.md")),
        ("Model trained", has_model and os.path.exists("models/best_model.pt")),
        ("Model uploaded to HF/GDrive/GH", False),
        ("demo_inference.ipynb works", os.path.exists("demo_inference.ipynb")),
        ("Examples with different prompts", os.path.exists("EXAMPLES.md")),
        ("requirements.txt updated", os.path.exists("requirements.txt")),
        (".gitignore configured", os.path.exists(".gitignore")),
    ]
    
    for task, done in checklist:
        status = "?" if done else "?"
        print(f"{status} {task}")
    
    print()
    
    # Next steps
    print("="*60)
    print("NEXT STEPS")
    print("="*60)
    print()
    
    steps_needed = []
    
    if not (has_data and os.path.exists("data/processed/train.txt")):
        steps_needed.append("1. Prepare data: python 01_prepare_data.py")
    
    if not (has_tokenizer and os.path.exists("tokenizer/tokenizer.json")):
        steps_needed.append("2. Train tokenizer: python 02_train_tokenizer.py")
    
    if not (has_model and os.path.exists("models/best_model.pt")):
        steps_needed.append("3. Train model: python 03_train.py")
    
    if has_model and os.path.exists("models/best_model.pt"):
        steps_needed.append("4. Test model: python quick_test.py")
        steps_needed.append("5. Open demo_inference.ipynb and run")
        steps_needed.append("6. Upload model to Hugging Face / Google Drive")
        steps_needed.append("7. Create GitHub repo and upload code")
        steps_needed.append("8. Update README.md with model link")
        steps_needed.append("9. Submit to instructor!")
    
    if steps_needed:
        for step in steps_needed:
            print(f"? {step}")
    else:
        print("?? Everything ready for submission!")
        print()
        print("Final steps:")
        print("1. Upload model to HF/GDrive/GH Release")
        print("2. Create GitHub repository")
        print("3. Update README with model link")
        print("4. Submit to instructor!")
    
    print()
    print("="*60)
    
    return all_good


if __name__ == "__main__":
    check_project()


