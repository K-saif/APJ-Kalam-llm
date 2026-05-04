import time
import pyperclip
from pynput.keyboard import Controller, Key

# Configuration
FILE_PATH = r"c:\Users\khans\Documents\APJ-Abdul-Kalam\wings_of_fire_converted.txt"
INTERVAL_SECONDS = 30
STARTUP_DELAY = 10 

PREFIX = """You are generating high-quality question-answer pairs for fine-tuning a language model to mimic the speaking and thinking style of Dr. A. P. J. Abdul Kalam, based on his book "Wings of Fire".

You will be given a paragraph from the book.

Your task:
- Generate 5 to 10 meaningful question-answer pairs based ONLY on the given paragraph.
- The questions should feel like someone is genuinely asking Dr. Kalam about his life, experiences, thoughts, or lessons.
- The answers should be written in the FIRST PERSON, as if Dr. Kalam himself is responding.

Style Guidelines for Answers:
- Keep answers VERY CONCISE (STRICT: 1 sentence, max 15–20 words)
- Reflect humility and clarity of thought
- Include light reflection, but avoid long philosophical explanations
- Use simple and natural language
- Stay grounded in the paragraph content

Style Guidelines for Questions:
- Natural, conversational, and curious tone
- Focus on "why", "how", "what did you feel", "what did you learn"
- Avoid generic or robotic questions

Output Format (STRICT JSON):
[
  {
    "question": "....",
    "answer": "...."
  }
]

Important:
- Each answer MUST be only ONE sentence (no multi-line answers)
- Do NOT exceed 20 words per answer
- Do NOT copy sentences directly from the paragraph
- Do NOT hallucinate events outside the paragraph
- Ensure answers still feel like Kalam’s voice: simple, wise, and grounded
- Maintain diversity in questions

Now here is the paragraph:

"""

def main():
    print("=" * 60)
    print("CLIPBOARD PASTER SCRIPT (INSTANT)")
    print("=" * 60)
    
    keyboard = Controller()
    
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        print(f"Total lines: {len(lines)}")
        print(f"Starting in {STARTUP_DELAY} seconds...")
        for i in range(STARTUP_DELAY, 0, -1):
            print(f"Preparing... {i}", end='\r')
            time.sleep(1)

        for line_index, line in enumerate(lines, 1):
            text = line.strip()
            if not text: continue  # Skip truly empty lines if preferred

            # 1. Combine Prefix and Line
            full_payload = f"{PREFIX}\n{text}"

            # 2. Copy to Clipboard
            pyperclip.copy(full_payload)

            # 3. Paste using Ctrl+V (or Cmd+V for Mac)
            with keyboard.pressed(Key.ctrl):
                keyboard.press('v')
                keyboard.release('v')

            # 4. Hit Enter
            time.sleep(0.5) # Short pause to ensure paste finishes before Enter
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)

            print(f"[{line_index}/{len(lines)}] Pasted successfully.")

            # 5. Wait for the interval
            if line_index < len(lines):
                time.sleep(INTERVAL_SECONDS)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()