#!/usr/bin/env python3
import os
import subprocess
import sys

def check_lfs_status():
    """Check if Git LFS is installed and configured properly"""
    try:
        subprocess.run(["git", "lfs", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Git LFS is installed.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Git LFS is not installed or not properly configured.")
        print("Please install Git LFS: https://git-lfs.github.com/")
        return False

def pull_lfs_content(corpus_dir):
    """Pull Git LFS content for the specified corpus directory"""
    os.chdir(corpus_dir)
    print(f"Pulling Git LFS content in {corpus_dir}...")
    
    try:
        # Initialize LFS for this repository if not already done
        subprocess.run(["git", "lfs", "install"], check=True)
        
        # Pull LFS content
        result = subprocess.run(["git", "lfs", "pull"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result.stdout.decode())
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error pulling LFS content: {e}")
        print(f"Error output: {e.stderr.decode() if e.stderr else 'No error output'}")
        return False

def check_jsonl_files(chunk_dir):
    """Check if JSONL files are actual content or LFS pointers"""
    is_lfs_pointer = False
    
    for filename in os.listdir(chunk_dir):
        if filename.endswith('.jsonl'):
            file_path = os.path.join(chunk_dir, filename)
            with open(file_path, 'r') as f:
                content = f.read().strip()
                if content.startswith('version https://git-lfs.github.com/spec'):
                    print(f"File {filename} is a Git LFS pointer, not actual content")
                    is_lfs_pointer = True
                    break
    
    return is_lfs_pointer

def main():
    # Get the absolute path to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the textbooks corpus
    corpus_dir = os.path.join(script_dir, "src", "data", "corpus", "textbooks")
    chunk_dir = os.path.join(corpus_dir, "chunk")
    
    if not os.path.exists(corpus_dir):
        print(f"Error: Corpus directory {corpus_dir} does not exist.")
        return False
        
    # Check if Git LFS is installed
    if not check_lfs_status():
        return False
        
    # Check if JSONL files are LFS pointers
    if check_jsonl_files(chunk_dir):
        # Pull LFS content
        return pull_lfs_content(corpus_dir)
    else:
        print("JSONL files seem to have actual content, no need to pull LFS data.")
        return True

if __name__ == "__main__":
    success = main()
    if success:
        print("Setup completed successfully!")
    else:
        print("Setup failed. Please check the error messages above.")
        sys.exit(1) 