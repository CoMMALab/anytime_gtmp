import pickle
import sys
from collections.abc import Mapping, Sequence

def print_tree(obj, indent=0, max_depth=3):
    prefix = ' ' * indent
    if max_depth < 0:
        print(f"{prefix}... (max depth reached)")
        return
    print(f"{prefix}{type(obj).__name__}")
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            print(f"{prefix}├─ Key: {repr(k)}")
            print_tree(v, indent + 4, max_depth - 1)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        print(f"{prefix}├─ Length: {len(obj)}")
        for i, item in enumerate(obj[:5]):  # Show up to 5 items
            print(f"{prefix}├─ [{i}]")
            print_tree(item, indent + 4, max_depth - 1)
        if len(obj) > 5:
            print(f"{prefix}└─ ... ({len(obj)-5} more items)")
    elif hasattr(obj, '__dict__'):
        for attr, value in vars(obj).items():
            print(f"{prefix}├─ Attr: {attr}")
            print_tree(value, indent + 4, max_depth - 1)
    else:
        print(f"{prefix}├─ Value: {repr(obj)}")

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "problems.pkl"
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    print_tree(data)
