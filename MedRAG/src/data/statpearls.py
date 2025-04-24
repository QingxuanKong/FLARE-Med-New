import os
import json
import tqdm
import xml.etree.ElementTree as ET

# Get the absolute path to the MedRAG directory
script_dir = os.path.dirname(os.path.abspath(__file__))
medrag_dir = os.path.dirname(os.path.dirname(script_dir))
corpus_dir = os.path.join(os.path.dirname(script_dir), "corpus")

def ends_with_ending_punctuation(s):
    ending_punctuation = ('.', '?', '!')
    return any(s.endswith(char) for char in ending_punctuation)

def concat(title, content):
    if ends_with_ending_punctuation(title.strip()):
        return title.strip() + " " + content.strip()
    else:
        return title.strip() + ". " + content.strip()

def extract_text(element):
    text = (element.text or "").strip()

    for child in element:
        text += (" " if len(text) else "") + extract_text(child)
        if child.tail and len(child.tail.strip()) > 0:
            text += (" " if len(text) else "") + child.tail.strip()
    return text.strip()

def is_subtitle(element):
    if element.tag != 'p':
        return False
    if len(list(element)) != 1:
        return False
    if list(element)[0].tag != 'bold':
        return False
    if list(element)[0].tail and len(list(element)[0].tail.strip()) > 0:
        return False
    return True

def extract(fpath):
    fname = os.path.basename(fpath).replace(".nxml", "")
    tree = ET.parse(fpath)
    title = tree.getroot().find(".//title").text
    sections = tree.getroot().findall(".//sec")
    saved_text = []
    j = 0
    last_text = None
    for sec in sections:
        sec_title = sec.find('./title').text.strip()
        sub_title = ""
        prefix = " -- ".join([title, sec_title])
        last_text = None
        last_json = None
        last_node = None
        for ch in sec:
            if is_subtitle(ch):
                last_text = None
                last_json = None
                sub_title = extract_text(ch)
                prefix = " -- ".join(prefix.split(" -- ")[:2] + [sub_title])
            elif ch.tag == 'p':
                curr_text = extract_text(ch)
                if len(curr_text) < 200 and last_text is not None and len(last_text + curr_text) < 1000:
                    last_text = " ".join([last_json['content'], curr_text])
                    last_json = {"id": last_json['id'], "title": last_json['title'], "content": last_text}
                    last_json["contents"] = concat(last_json["title"], last_json["content"])
                    saved_text[-1] = json.dumps(last_json)
                else:
                    last_text = curr_text
                    last_json = {"id": '_'.join([fname, str(j)]), "title": prefix, "content": curr_text}
                    last_json["contents"] = concat(last_json["title"], last_json["content"])
                    saved_text.append(json.dumps(last_json))
                    j += 1
            elif ch.tag == 'list':
                list_text = [extract_text(c) for c in ch]
                if last_text is not None and len(" ".join(list_text) + last_text) < 1000:
                    last_text = " ".join([last_json["content"]] + list_text)
                    last_json = {"id": last_json['id'], "title": last_json['title'], "content": last_text}
                    last_json["contents"] = concat(last_json["title"], last_json["content"])
                    saved_text[-1] = json.dumps(last_json)
                elif len(" ".join(list_text)) < 1000:
                    last_text = " ".join(list_text)
                    last_json = {"id": '_'.join([fname, str(j)]), "title": prefix, "content": last_text}
                    last_json["contents"] = concat(last_json["title"], last_json["content"])
                    saved_text.append(json.dumps(last_json))
                    j += 1
                else:
                    last_text = None
                    last_json = None                    
                    for c in list_text:
                        saved_text.append(json.dumps({"id": '_'.join([fname, str(j)]), "title": prefix, "content": c, "contents": concat(prefix, c)}))
                        j += 1
                if last_node is not None and is_subtitle(last_node):
                    sub_title = ""
                    prefix = " -- ".join([title, sec_title])
            last_node = ch
    return saved_text

if __name__ == "__main__":
    input_dir = os.path.join(corpus_dir, "statpearls", "statpearls_NBK430685")
    output_dir = os.path.join(corpus_dir, "statpearls", "chunk")
    
    print(f"Looking for files in: {input_dir}")
    print(f"Input directory exists: {os.path.exists(input_dir)}")
    
    if not os.path.exists(input_dir):
        print("Error: Input directory does not exist")
        print(f"Current script path: {script_dir}")
        print(f"MedRAG directory: {medrag_dir}")
        print(f"Corpus directory: {corpus_dir}")
        exit(1)
    
    fnames = sorted([fname for fname in os.listdir(input_dir) if fname.endswith("nxml")])
    print(f"Found {len(fnames)} NXML files")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    for fname in tqdm.tqdm(fnames):
        fpath = os.path.join(input_dir, fname)
        saved_text = extract(fpath)
        if len(saved_text) > 0:
            output_file = os.path.join(output_dir, fname.replace(".nxml", ".jsonl"))
            with open(output_file, 'w') as f:
                f.write('\n'.join(saved_text))
    
    print(f"Processing complete. Check {output_dir} for output files.")