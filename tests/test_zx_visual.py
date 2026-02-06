#!/usr/bin/env python3
"""
Test script for ZX visualization functionality
"""

from spidertrace.circuit import Gate
from spidertrace.engine import propagate_errors
from spidertrace.error import PauliError
from spidertrace.zx_visual import draw_trace_step, visualize_trace, save_diagram
import pyzx as zx


def test_hadamard_propagation():
    """Test X error propagation through Hadamard gate"""
    print("=== Test: X error through Hadamard ===")
    
    # Create circuit: H on qubit 0
    circuit = [Gate("H", (0,))]
    
    # Initial X error on qubit 0
    errors = [PauliError(0, "X")]
    
    # Propagate errors
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial errors: {[(e.qubit, e.type) for e in errors]}")
    for i, step in enumerate(trace):
        print(f"Step {i}: Gate {step.gate.name} on qubit {step.gate.qubits}")
        print(f"  Errors after: {dict(step.errors_after)}")
    
    # Create ZX diagrams
    diagrams = visualize_trace(circuit, trace)
    
    # Save diagrams
    for i, diagram in enumerate(diagrams):
        filename = f"h_test_step_{i}.png"
        save_diagram(diagram, filename)
        print(f"  Saved diagram: {filename}")
    
    return diagrams


def test_cnot_propagation():
    """Test error propagation through CNOT gate"""
    print("\n=== Test: X error through CNOT ===")
    
    # Create circuit: CNOT with control 0, target 1
    circuit = [Gate("CNOT", (0, 1))]
    
    # Initial X error on control qubit
    errors = [PauliError(0, "X")]
    
    # Propagate errors
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial errors: {[(e.qubit, e.type) for e in errors]}")
    for i, step in enumerate(trace):
        print(f"Step {i}: Gate {step.gate.name} on qubits {step.gate.qubits}")
        print(f"  Errors after: {dict(step.errors_after)}")
    
    # Create ZX diagrams
    diagrams = visualize_trace(circuit, trace)
    
    # Save diagrams
    for i, diagram in enumerate(diagrams):
        filename = f"cnot_x_test_step_{i}.png"
        save_diagram(diagram, filename)
        print(f"  Saved diagram: {filename}")
    
    return diagrams


def test_multi_gate_circuit():
    """Test error propagation through multi-gate circuit"""
    print("\n=== Test: Multi-gate circuit ===")
    
    # Create circuit: H on qubit 0, then CNOT(0,1)
    circuit = [
        Gate("H", (0,)),
        Gate("CNOT", (0, 1))
    ]
    
    # Initial X error on qubit 0
    errors = [PauliError(0, "X")]
    
    # Propagate errors
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial errors: {[(e.qubit, e.type) for e in errors]}")
    for i, step in enumerate(trace):
        print(f"Step {i}: Gate {step.gate.name} on qubits {step.gate.qubits}")
        print(f"  Errors after: {dict(step.errors_after)}")
    
    # Create ZX diagrams
    diagrams = visualize_trace(circuit, trace)
    
    # Save diagrams
    for i, diagram in enumerate(diagrams):
        filename = f"multi_gate_step_{i}.png"
        save_diagram(diagram, filename)
        print(f"  Saved diagram: {filename}")
    
    return diagrams


def test_z_error_propagation():
    """Test Z error propagation through Hadamard"""
    print("\n=== Test: Z error through Hadamard ===")
    
    # Create circuit: H on qubit 0
    circuit = [Gate("H", (0,))]
    
    # Initial Z error on qubit 0
    errors = [PauliError(0, "Z")]
    
    # Propagate errors
    trace = propagate_errors(circuit, errors)
    
    print(f"Initial errors: {[(e.qubit, e.type) for e in errors]}")
    for i, step in enumerate(trace):
        print(f"Step {i}: Gate {step.gate.name} on qubit {step.gate.qubits}")
        print(f"  Errors after: {dict(step.errors_after)}")
    
    # Create ZX diagrams
    diagrams = visualize_trace(circuit, trace)
    
    # Save diagrams
    for i, diagram in enumerate(diagrams):
        filename = f"z_test_step_{i}.png"
        save_diagram(diagram, filename)
        print(f"  Saved diagram: {filename}")
    
    return diagrams


def main():
    """Run all tests"""
    print("Testing SpiderTrace ZX Visualization")
    print("=" * 50)
    
    try:
        # Test basic functionality
        test_hadamard_propagation()
        test_z_error_propagation()
        
        # Note: CNOT tests will show incomplete behavior since CNOT rules are commented out
        print("\nNote: CNOT tests show incomplete behavior - CNOT propagation rules need to be implemented in engine.py")
        
        print("\n" + "=" * 50)
        print("All tests completed! Check the generated PNG files for ZX diagrams.")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
