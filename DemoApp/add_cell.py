import json

path = "/Users/jeffreyvillamil/Desktop/mapua/thesis/thesisApp/tests_3seed.ipynb"

with open(path, "r") as f:
    nb = json.load(f)

new_cell = {
  "cell_type": "code",
  "execution_count": None,
  "id": "UnzipDatasetCell",
  "metadata": {},
  "outputs": [],
  "source": [
    "# Unzip structured_ewaste_pipeline.zip\n",
    "import zipfile\n",
    "import os\n",
    "\n",
    "zip_path = \"/content/drive/MyDrive/Thesis/Model_Checkpoints/structured_ewaste_pipeline.zip\"\n",
    "extract_dir = \"/content\"\n",
    "\n",
    "if not os.path.exists(\"/content/structured_ewaste_pipeline\"):\n",
    "    print(f\"Extracting {zip_path}...\")\n",
    "    with zipfile.ZipFile(zip_path, 'r') as zip_ref:\n",
    "        zip_ref.extractall(extract_dir)\n",
    "    print(\"Extraction complete.\")\n",
    "else:\n",
    "    print(\"Dataset already extracted in /content/structured_ewaste_pipeline\")"
  ]
}

# Find the SETUP cell
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and any('SETUP' in line for line in cell['source']):
        insert_idx = i + 1
        break

if insert_idx != -1:
    nb['cells'].insert(insert_idx, new_cell)
    with open(path, "w") as f:
        json.dump(nb, f, indent=2)
    print("Successfully added cell.")
else:
    print("Could not find SETUP cell.")
