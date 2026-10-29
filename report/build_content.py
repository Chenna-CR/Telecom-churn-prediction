import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import content_a
import content_b, content_c
from PIL import Image
B = content_a.B
for b in B:
    if b['t'] == 'fig':
        w, h = Image.open(b['file']).size
        b['px_w'] = int(b['w'] * 96); b['px_h'] = int(b['w'] * 96 * h / w)
json.dump(B, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'content.json'), 'w'), ensure_ascii=False, indent=0)
from collections import Counter
print(Counter(b['t'] for b in B)); 
words = sum(len(b.get('text', '').split()) for b in B) + sum(len(' '.join(b['items']).split()) for b in B if b['t'] in ('bullets','refs'))
print('approx words', words)
