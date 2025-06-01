# check_main.py
import main
import inspect

print("Checking main.py module...")
print(f"Module file location: {main.__file__}")
print("\nChecking run function signature:")
print(f"Signature: {inspect.signature(main.run)}")

# Check the actual source code
print("\nFirst 500 characters of run function:")
source = inspect.getsource(main.run)
print(source[:500])