#!/usr/bin/env python3
"""
Test script for the evaluate.py script.
"""

import os
import sys
import subprocess

def test_evaluate():
    """Test the evaluate script with a sample test ID."""
    
    # Test with the provided test ID
    test_id = "20250815_150016"
    
    print(f"Testing evaluate.py with test ID: {test_id}")
    
    # Check if the test directory exists
    test_dir = f"output/blendergym/{test_id}"
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} does not exist.")
        return False
    
    print(f"Found test directory: {test_dir}")
    
    # List contents of the test directory
    contents = os.listdir(test_dir)
    print(f"Directory contents: {contents[:10]}...")  # Show first 10 items
    
    # Run the evaluate script
    try:
        cmd = [sys.executable, "evaluate.py", test_id]
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Evaluation completed successfully!")
            print("Output:")
            print(result.stdout)
            
            # Check if evaluation results were created
            eval_dir = f"output/blendergym/{test_id}/evaluation"
            if os.path.exists(eval_dir):
                eval_files = os.listdir(eval_dir)
                print(f"Evaluation files created: {eval_files}")
            else:
                print("Warning: Evaluation directory was not created")
                
        else:
            print("❌ Evaluation failed!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_evaluate()
    sys.exit(0 if success else 1) 