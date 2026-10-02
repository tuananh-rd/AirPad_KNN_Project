import os
import sys
import urllib.request
import zipfile
import time

URL = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
TARGET_DIR = os.path.join("data", "hagrid_v2")
ZIP_PATH = os.path.join(TARGET_DIR, "annotations.zip")
EXTRACT_DIR = os.path.join(TARGET_DIR, "annotations")

os.makedirs(TARGET_DIR, exist_ok=True)

if os.path.exists(EXTRACT_DIR) and len(os.listdir(EXTRACT_DIR)) > 0:
    print(f"Target directory '{EXTRACT_DIR}' already contains files:")
    print(os.listdir(EXTRACT_DIR)[:10])
    sys.exit(0)

if not os.path.exists(ZIP_PATH) or os.path.getsize(ZIP_PATH) < 700_000_000:
    print(f"Downloading {URL} to {ZIP_PATH} ...", flush=True)
    start_time = time.time()
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if downloaded > total_size:
            downloaded = total_size
        pct = (downloaded / total_size) * 100 if total_size > 0 else 0
        mb_down = downloaded / (1024 * 1024)
        mb_tot = total_size / (1024 * 1024)
        if block_num % 1000 == 0 or downloaded == total_size:
            elapsed = time.time() - start_time
            speed = mb_down / elapsed if elapsed > 0 else 0
            print(f"  Downloaded: {mb_down:.1f} MB / {mb_tot:.1f} MB ({pct:.1f}%) - {speed:.1f} MB/s", flush=True)

    urllib.request.urlretrieve(URL, ZIP_PATH, reporthook)
    print(f"Download completed in {time.time() - start_time:.1f}s. Size: {os.path.getsize(ZIP_PATH)} bytes", flush=True)
else:
    print(f"ZIP file already exists: {ZIP_PATH} ({os.path.getsize(ZIP_PATH)} bytes)", flush=True)

print(f"Extracting {ZIP_PATH} to {EXTRACT_DIR} ...", flush=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
start_extract = time.time()
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print(f"Extraction completed in {time.time() - start_extract:.1f}s.", flush=True)
extracted_files = []
for root, dirs, files in os.walk(EXTRACT_DIR):
    for f in files:
        extracted_files.append(os.path.relpath(os.path.join(root, f), EXTRACT_DIR))

print(f"Total extracted files: {len(extracted_files)}")
print("Sample files:", extracted_files[:10])
