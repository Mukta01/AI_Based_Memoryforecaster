import importlib
import sys

deps = ["psutil", "pandas", "numpy", "sklearn", "torch", "matplotlib", "joblib"]
missing = []

for d in deps:
    try:
        importlib.import_module(d)
        print(f"✅ {d} is available")
    except ImportError:
        print(f"❌ {d} is missing")
        missing.append(d)

if missing:
    print(f"\nMissing dependencies: {', '.join(missing)}")
    print("Please run: python3 -m pip install " + " ".join(missing))
    sys.exit(1)
else:
    print("\nAll dependencies are ready!")
