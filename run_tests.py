"""
Simple script to run pytest on all tests in the project.
"""
import subprocess
import sys

def main():
    print("Running tests for seg-ana package...")
    
    # Run pytest with verbose output
    result = subprocess.run(
        ["pytest", "-v", "tests/"],
        capture_output=True,
        text=True
    )
    
    # Print output
    print(result.stdout)
    
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    
    # Return exit code
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
