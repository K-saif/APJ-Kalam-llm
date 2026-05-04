def normalize_paragraphs(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    paragraphs = []
    current_para = []

    for line in lines:
        stripped = line.strip()

        if stripped == "":
            if current_para:
                # Join paragraph into single line
                paragraph = " ".join(current_para)
                paragraphs.append(paragraph)
                current_para = []
        else:
            current_para.append(stripped)

    # Add last paragraph if exists
    if current_para:
        paragraph = " ".join(current_para)
        paragraphs.append(paragraph)

    # Write to output file (each paragraph = one line)
    with open(output_file, "w", encoding="utf-8") as f:
        for para in paragraphs:
            f.write(para + "\n")


    
# Example usage
input_file = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\wings_of_fire.txt"
output_file = r"C:\Users\khans\Documents\APJ-Abdul-Kalam\wings_of_fire_converted.txt"

normalize_paragraphs(input_file, output_file)