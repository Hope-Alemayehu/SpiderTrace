#!/usr/bin/env python3
"""
Simple test script for SpiderTrace functionality
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spidertrace.circuit import Gate
from spidertrace.engine import propagate_errors
from spidertrace.error import PauliError


def test_hadamard_x_to_z():
    """Test X error through Hadamard becomes Z"""
    print("=== Test: X error through Hadamard ===")
    
    circuit = [Gate("H", (0,))]
    errors = [PauliError(0, "X")]
    
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial: X on qubit 0")
    for i, step in enumerate(trace):
        print(f"After H: {dict(step.errors_after)}")
    
    expected = {0: "Z"}
    actual = trace[0].errors_after
    assert actual == expected, f"Expected {expected}, got {actual}"
    print("PASS: Test passed!")
    return trace


def test_hadamard_z_to_x():
    """Test Z error through Hadamard becomes X"""
    print("\n=== Test: Z error through Hadamard ===")
    
    circuit = [Gate("H", (0,))]
    errors = [PauliError(0, "Z")]
    
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial: Z on qubit 0")
    for i, step in enumerate(trace):
        print(f"After H: {dict(step.errors_after)}")
    
    expected = {0: "X"}
    actual = trace[0].errors_after
    assert actual == expected, f"Expected {expected}, got {actual}"
    print("PASS: Test passed!")
    return trace


def test_cnot_x_spread():
    """Test X on control spreads to target"""
    print("\n=== Test: X on control spreads to target ===")
    
    circuit = [Gate("CNOT", (0, 1))]
    errors = [PauliError(0, "X")]
    
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial: X on qubit 0 (control)")
    for i, step in enumerate(trace):
        print(f"After CNOT: {dict(step.errors_after)}")
    
    expected = {0: "X", 1: "X"}
    actual = trace[0].errors_after
    assert actual == expected, f"Expected {expected}, got {actual}"
    print("PASS: Test passed!")
    return trace


def test_cnot_z_spread():
    """Test Z on target spreads to control"""
    print("\n=== Test: Z on target spreads to control ===")
    
    circuit = [Gate("CNOT", (0, 1))]
    errors = [PauliError(1, "Z")]
    
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial: Z on qubit 1 (target)")
    for i, step in enumerate(trace):
        print(f"After CNOT: {dict(step.errors_after)}")
    
    expected = {0: "Z", 1: "Z"}
    actual = trace[0].errors_after
    assert actual == expected, f"Expected {expected}, got {actual}"
    print("PASS: Test passed!")
    return trace


def test_cnot_xz_cancellation():
    """Test X on control and Z on target create Y errors"""
    print("\n=== Test: X on control + Z on target ===")
    
    circuit = [Gate("CNOT", (0, 1))]
    errors = [PauliError(0, "X"), PauliError(1, "Z")]
    
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial: X on qubit 0, Z on qubit 1")
    for i, step in enumerate(trace):
        print(f"After CNOT: {dict(step.errors_after)}")
    
    expected = {0: "Y", 1: "Y"}
    actual = trace[0].errors_after
    assert actual == expected, f"Expected {expected}, got {actual}"
    print("PASS: Test passed!")
    return trace


def test_multi_gate_circuit():
    """Test multi-gate circuit: H then CNOT"""
    print("\n=== Test: Multi-gate circuit (H then CNOT) ===")
    
    circuit = [
        Gate("H", (0,)),
        Gate("CNOT", (0, 1))
    ]
    errors = [PauliError(0, "X")]
    
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial: X on qubit 0")
    for i, step in enumerate(trace):
        print(f"Step {i+1} ({step.gate.name}): {dict(step.errors_after)}")
    
    # After H: X -> Z on qubit 0
    # After CNOT: Z on 0 stays (no spreading since Z is on control, not target)
    expected_step1 = {0: "Z"}
    expected_step2 = {0: "Z"}
    
    assert trace[0].errors_after == expected_step1, f"Step 1: Expected {expected_step1}, got {trace[0].errors_after}"
    assert trace[1].errors_after == expected_step2, f"Step 2: Expected {expected_step2}, got {trace[1].errors_after}"
    print("PASS: Test passed!")
    return trace


def main():
    """Run all tests"""
    print("SpiderTrace Test Suite")
    print("=" * 50)
    
    try:
        test_hadamard_x_to_z()
        test_hadamard_z_to_x()
        test_cnot_x_spread()
        test_cnot_z_spread()
        test_cnot_xz_cancellation()
        test_multi_gate_circuit()
        
        print("\n" + "=" * 50)
        print("SUCCESS: All tests passed! SpiderTrace is working correctly.")
        
    except Exception as e:
        print(f"\nFAIL: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
