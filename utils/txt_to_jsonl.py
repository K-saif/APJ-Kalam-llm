import json

input_file = "C:/Users/khans/Documents/APJ-Abdul-Kalam/wings_of_fire_converted.txt"   # one paragraph per line
output_file = "C:/Users/khans/Documents/APJ-Abdul-Kalam/kalam_cpt.jsonl"

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "a", encoding="utf-8") as f_out:
    
    for line in f_in:
        paragraph = line.strip()
        if paragraph:  # skip empty lines
            json_line = {"text": paragraph}
            f_out.write(json.dumps(json_line, ensure_ascii=False) + "\n")