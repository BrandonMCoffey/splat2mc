import os
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(CURRENT_DIR, "spritesheet.png")
ROWS = 3

def create_spritesheet():
    files = [f for f in os.listdir(CURRENT_DIR) if f.lower().endswith('.png') and f != "spritesheet.png"]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    total_frames = len(files)
    if total_frames % ROWS != 0:
        print(f"Warning: Total frames ({total_frames}) is not divisible by {ROWS} rows.")
    
    # Auto-detect column count based on file count
    cols = total_frames // ROWS
    if cols == 0:
        print("No images found!")
        return

    print(f"Auto-detected {cols} columns based on {total_frames} total frames.")

    with Image.open(os.path.join(CURRENT_DIR, files[0])) as img:
        w, h = img.size

    spritesheet = Image.new('RGBA', (w * cols, h * ROWS), (0, 0, 0, 0))

    for index, filename in enumerate(files):
        # Calculate grid position
        col = index % cols
        row = index // cols
        
        with Image.open(os.path.join(CURRENT_DIR, filename)) as frame:
            spritesheet.paste(frame.resize((w, h)), (col * w, row * h))
            if index % 5 == 0: # Print every 5th for cleanliness
                print(f"Pasting frame {index}...")

    spritesheet.save(OUTPUT_FILE)
    print(f"\nSuccess! Saved {cols}x{ROWS} spritesheet.")
    input("Press Enter to exit...")

if __name__ == "__main__":
    create_spritesheet()