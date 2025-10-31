#!/usr/bin/env python3
"""
Setup Script for Car Brand Recognition Project
This script helps set up the environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True

def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return False
    
    # Install packages
    commands = [
        ("python3 -m pip install --upgrade pip", "Upgrading pip"),
        ("python3 -m pip install -r requirements.txt", "Installing project dependencies")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def create_directories():
    """Create necessary project directories."""
    print("\n📁 Creating project directories...")
    
    directories = [
        "dataset/train",
        "dataset/test", 
        "dataset/validation",
        "models",
        "notebooks",
        "src",
        "webapp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    return True

def setup_git_repo():
    """Initialize git repository if not already initialized."""
    print("\n🔧 Setting up git repository...")
    
    if Path(".git").exists():
        print("   ✅ Git repository already initialized")
        return True
    
    commands = [
        ("git init", "Initializing git repository"),
        ("git add .", "Adding files to git"),
        ("git commit -m 'Initial commit: Car brand recognition project setup'", "Creating initial commit")
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            print(f"   ⚠️  {description} failed (this is optional)")
    
    return True

def create_gitignore():
    """Create .gitignore file for the project."""
    print("\n📝 Creating .gitignore file...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# Dataset
dataset/train/*
dataset/test/*
dataset/validation/*
!dataset/README.md

# Models
models/*.h5
models/*.pkl
models/*.joblib

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log

# Temporary files
*.tmp
*.temp
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("   ✅ .gitignore created")
    return True

def verify_installation():
    """Verify that key packages are installed correctly."""
    print("\n🔍 Verifying installation...")
    
    test_imports = [
        ("tensorflow", "TensorFlow"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-learn")
    ]
    
    failed_imports = []
    
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"   ✅ {name} imported successfully")
        except ImportError:
            print(f"   ❌ {name} import failed")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n⚠️  Some packages failed to import: {', '.join(failed_imports)}")
        print("   Try running: pip install -r requirements.txt")
        return False
    
    print("\n✅ All packages verified successfully!")
    return True

def main():
    """Main setup function."""
    print("🚗 Car Brand Recognition Project Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create .gitignore
    create_gitignore()
    
    # Setup git (optional)
    setup_git_repo()
    
    # Verify installation
    if not verify_installation():
        print("⚠️  Some packages may not be working correctly")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Download dataset: python src/download_data.py")
    print("2. Explore data: jupyter notebook notebooks/01_problem_understanding.ipynb")
    print("3. Prepare data: python src/data_preparation.py")
    print("4. Train model: python src/train_model.py")
    print("5. Deploy app: streamlit run webapp/app.py")
    print("\nHappy coding! 🚀")

if __name__ == "__main__":
    main()