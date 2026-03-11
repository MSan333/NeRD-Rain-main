cd ~/pyproject/nerd/Datasets/Rain200L/test/input/
python3 << 'EOF'
import os
from PIL import Image

inp_dir = '.'
inp_files = sorted(os.listdir(inp_dir))
bad_files = []

print("Checking images...")
for f in inp_files:
    if f.endswith('.png') or f.endswith('.jpg'):
        try:
            img = Image.open(os.path.join(inp_dir, f))
            img.load()
        except (OSError, IOError) as e:
            print(f'Bad file: {f} - {e}')
            bad_files.append(f)

print(f'\nFound {len(bad_files)} corrupted files:')
for f in bad_files:
    print(f'  {f}')

if bad_files:
    print('\nDeleting corrupted files...')
    for f in bad_files:
        os.remove(os.path.join(inp_dir, f))
        print(f'Deleted: {f}')
else:
    print('\nNo corrupted files found.')
EOF
