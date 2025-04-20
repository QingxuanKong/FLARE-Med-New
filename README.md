# MedRAG Pipeline Setup Guide & Script

This document provides a step-by-step guide to configure the MedRAG pipeline and its associated components like MIRAGE. It includes system setup, data handling, indexing, generation, and evaluation.

**Assumptions:**

- You are running on a AWS Ubuntu 22.04 instance.
- You have `sudo` privileges.
- The project is structured with a root directory (e.g., `~/project/`) containing `MedRAG` and `MIRAGE` subdirectories. Adjust paths throughout this guide if your structure differs.
- You have sufficient disk space for datasets (PubMed and Wikipedia is very large) and indices.

---

## 1. System Prerequisites & Setup

```bash
sudo apt update
sudo apt install -y lftp git python3.10 python3.10-venv python3.10-dev build-essential unzip awscli openjdk-17-jdk
```

## 2. Setting up Python virtual environment

```bash
# Navigate to your main project directory (IMPORTANT: ADJUST IF NEEDED)
cd ~/project/

# Create the virtual environment in the current directory
python3.10 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and install base Python packages
pip install --upgrade pip
pip install -r MedRAG/requirements.txt
```

## 3. MedRAG Configuration

```bash
# Navigate to the MedRAG directory (IMPORTANT: ADJUST PATH IF NEEDED)
cd MedRAG
```

### 3.1. Download Corpus Data

```bash
# PubMed Baseline
mkdir -p src/corpus/pubmed/baseline
cd src/corpus/pubmed/baseline
# Use lftp to download all PubMed baseline XML files
lftp -c "open ftp.ncbi.nlm.nih.gov; cd pubmed/baseline; mget *.xml.gz"
cd ../../../.. # Back to MedRAG root

# StatPearls
mkdir -p src/corpus/statpearls
cd src/corpus/statpearls
wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz
tar -xzf statpearls_NBK430685.tar.gz
rm statpearls_NBK430685.tar.gz
cd ../../.. # Back to MedRAG root

# Textbooks
mkdir -p src/corpus/textbooks
# MANUAL ACTION REQUIRED: Download textbook data from
https://drive.google.com/file/d/1ImYUSLk9JbgHXOemfvyiDiirluZHPeQw/view

# Wikipedia (Placeholder)
# Data will be downloaded and chunked in the next step when running src/data/wikipedia.py
```

### 3.2. Construct Data Chunks

The chunk construction of pubmed and wikipedia takes a while.

```bash
python src/data/pubmed.py
python src/data/statpearls.py
python src/data/textbooks.py
python src/data/wikipedia.py
```

### 3.3. Index Corpus using Pyserini

```bash
# Setup Java Environment
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 >> ~/.bashrc
export PATH=$JAVA_HOME/bin:$PATH >> ~/.bashrc
source ~/.bashrc

# Checking Java version
java -version

# Run Indexing (Background Processes)

echo ">>> Starting indexing processes in the background. Monitor log files."

# Ensure virtual environment is active

source ../.venv/bin/activate

# Index PubMed
INDEX_INPUT_PUBMED="src/corpus/pubmed/chunk"
INDEX_OUTPUT_PUBMED="src/corpus/pubmed/index/bm25"
python -m pyserini.index.lucene \
    --collection JsonCollection \
    --input "$INDEX_INPUT_PUBMED" \
 --index "$INDEX_OUTPUT_PUBMED" \
 --generator DefaultLuceneDocumentGenerator \
 --threads 5 \
 --storePositions --storeDocvectors --storeRaw > pubmed_index.log 2>&1 &

# Replace with statpearls, textbooks and wikipedia to index the rest corpus
```

### 3.4. Test MedRAG Run

```bash
python test.py
```

## 4. MIRAGE Configuration and Execution

```bash
# Navigate to the MIRAGE directory (IMPORTANT: ADJUST PATH IF NEEDED)
cd MIRAGE
```

### 4.1. (Optional) Download Pre-Retrieved Snippets

```bash
wget -O retrieved_snippets_10k.zip https://virginia.box.com/shared/static/cxq17th6eisl2pn04vp0x723zczlvlzc.zip
unzip retrieved_snippets_10k.zip -d retrieved_snippets_10k
```

### 4.2. Generate Answers

```bash
# Logging into Hugging Face Hub..."
huggingface-cli login

# Adapt log file name if needed based on config
python src/generate.py --config config.json > log_generate_bioasq.txt 2>&1

# Config the json file to generate answer of different datasets
```

### 4.3. Evaluate Accuracy

```bash
# Adapt log file name if needed based on config
python src/evaluation.py --config config.json > log_evaluate_bioasq.txt 2>&1

# Config the json file to evaluate answer of different datasets
```

## 5. Optional: AWS S3 Data Management Commands

```bash
# --- Configure AWS CLI (Run once manually if needed) ---
aws configure

# --- Example: Upload MedRAG corpus data TO S3 ---
aws s3 cp ~/project/MedRAG/src/corpus s3://medrag-isla/corpus --recursive

# --- Example: Check S3 contents ---
aws s3 ls s3://medrag-isla/corpus/ --recursive --human-readable --summarize

# --- Example: Download/Sync MedRAG corpus FROM S3 ---
aws s3 sync s3://medrag-isla/corpus/ ~/project/MedRAG/src/corpus/
```
