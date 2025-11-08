import subprocess
import sys
import importlib.util
import os

def is_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return False
        return True
    except ImportError:
        return False

def install_package(package):
    """Install a package using pip"""
    try:
        print(f"🔍 Checking {package}...")
        
        # Extract package name without version
        base_package = package.split('==')[0].split('>')[0].split('<')[0]
        
        if is_package_installed(base_package):
            print(f"✅ {base_package} is already installed")
            return True
        
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
        
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False
    except Exception as e:
        print(f"❌ Error installing {package}: {e}")
        return False

def main():
    print("🚀 CANCER DETECTION SYSTEM - DEPENDENCY INSTALLER")
    print("=" * 50)
    
    # List of required packages
    requirements = [
        "flask==2.3.3",
        "tensorflow==2.13.0",
        "numpy==1.24.3",
        "joblib==1.3.2",
        "flask-sqlalchemy==3.0.5",
        "flask-login==0.6.3",
        "werkzeug==2.3.7",
        "reportlab==4.0.4",
        "pillow==10.0.0",
        "requests==2.31.0"
    ]
    
    print("📋 Required packages:")
    for req in requirements:
        print(f"   - {req}")
    
    print("\n🔄 Starting installation...")
    print("-" * 50)
    
    success_count = 0
    failed_packages = []
    
    for package in requirements:
        if install_package(package):
            success_count += 1
        else:
            failed_packages.append(package)
    
    print("-" * 50)
    print("📊 INSTALLATION SUMMARY:")
    print(f"✅ Successfully installed/verified: {success_count}/{len(requirements)} packages")
    
    if failed_packages:
        print(f"❌ Failed to install: {len(failed_packages)} packages")
        for failed in failed_packages:
            print(f"   - {failed}")
        print("\n💡 Try installing failed packages manually:")
        for failed in failed_packages:
            print(f"   pip install {failed}")
    else:
        print("🎉 All dependencies are ready!")
    
    print("\n🔧 Additional System Checks:")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"   Python Version: {python_version}")
    
    # Check if running in VS Code
    try:
        import vscode
        print("   ✅ Running in VS Code environment")
    except ImportError:
        print("   ℹ️  Not in VS Code (optional)")
    
    print("\n🚀 You can now run: python app.py")

if __name__ == "__main__":
    main()