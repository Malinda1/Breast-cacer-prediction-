#!/usr/bin/env python3
"""
Script to install all requirements for the Breast Cancer Prediction project.
This script replicates the functionality of the requirement_install.ipynb notebook.
"""

import subprocess
import sys
import os
from pathlib import Path

def print_banner():
    """Print a nice banner."""
    print("🚀" + "="*60 + "🚀")
    print("   Breast Cancer Prediction - Requirements Installer")
    print("🚀" + "="*60 + "🚀")

def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible!")
        return True
    else:
        print("❌ Python version must be 3.8 or higher!")
        return False

def install_requirements():
    """Install all requirements from requirements.txt."""
    print("\n📦 Installing requirements...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    print(f"📁 Using requirements file: {requirements_file.absolute()}")
    
    try:
        # Upgrade pip first
        print("⬆️  Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True, text=True)
        
        # Install requirements
        print("📥 Installing packages...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                               check=True, capture_output=True, text=True)
        
        print("✅ All requirements installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def verify_installation():
    """Verify that key packages are installed."""
    print("\n🔍 Verifying installation...")
    
    key_packages = [
        "numpy", "pandas", "sklearn", "tensorflow", 
        "torch", "transformers", "matplotlib", "seaborn"
    ]
    
    failed_packages = []
    
    for package in key_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Some packages failed to import: {failed_packages}")
        return False
    else:
        print("\n🎉 All key packages are working!")
        return True

def main():
    """Main function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some packages may not be working correctly.")
        print("You may need to install them manually.")
    
    print("\n" + "="*60)
    print("🎉 Installation completed successfully!")
    print("You can now start working on your Breast Cancer Prediction project.")
    print("="*60)

if __name__ == "__main__":
    main() 