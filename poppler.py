import os
import sys
import zipfile
import urllib.request
import shutil

def download_and_extract_poppler(dest_dir="C:\\poppler"):
    url = "https://github.com/oschwartz10612/poppler-windows/releases/download/v23.11.0-0/Release-23.11.0-0.zip"
    zip_path = os.path.join(dest_dir, "poppler.zip")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    print("Downloading Poppler...")
    urllib.request.urlretrieve(url, zip_path)
    print("Extracting Poppler...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    os.remove(zip_path)
    # Find the bin directory
    for root, dirs, files in os.walk(dest_dir):
        if 'pdfinfo.exe' in files:
            bin_dir = root
            break
    else:
        raise Exception("Poppler bin directory not found after extraction.")
    print(f"Poppler bin directory: {bin_dir}")
    # Add to PATH for current process
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]
    print("Poppler installed and PATH updated for this session.")
    print("Testing pdfinfo...")
    os.system('pdfinfo -v')

if __name__ == "__main__":
    if os.name == 'nt':
        download_and_extract_poppler()
    else:
        print("This script is intended for Windows. Use install_poppler.sh for Linux/macOS.")
