#!/usr/bin/env python3
"""
Test your own custom circuits with SpiderTrace
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spidertrace.circuit import Gate
from spidertrace.engine import propagate_errors
from spidertrace.error import PauliError
from spidertrace.zx_visual import save_complete_visualization

try:
    import pytest
except ImportError:  # running as a script without pytest installed
    pass
else:
    pytestmark = pytest.mark.skip(
        reason="interactive script, not a test: it reads gates and errors with input(); "
               "run it directly as a script from the repo root with PYTHONPATH=.")


def create_custom_circuit():
    """Interactive circuit builder"""
    
    print("=" * 60)
    print("SPIDERTRACE: CUSTOM CIRCUIT BUILDER")
    print("=" * 60)
    
    circuit = []
    errors = []
    
    print("\n📋 Build your quantum circuit:")
    print("Available gates: H (Hadamard), CNOT")
    print("Format: H qubit_number OR CNOT control target")
    print("Type 'done' when finished building circuit")
    
    while True:
        gate_input = input(f"\nGate {len(circuit)+1} (or 'done'): ").strip().upper()
        
        if gate_input == 'DONE':
            break
        
        if gate_input == 'H':
            try:
                qubit = int(input("  Enter qubit number for H gate: "))
                circuit.append(Gate("H", (qubit,)))
                print(f"  ✓ Added H on qubit {qubit}")
            except ValueError:
                print("  ❌ Invalid qubit number")
        
        elif gate_input == 'CNOT':
            try:
                control = int(input("  Enter control qubit: "))
                target = int(input("  Enter target qubit: "))
                circuit.append(Gate("CNOT", (control, target)))
                print(f"  ✓ Added CNOT (control:{control}, target:{target})")
            except ValueError:
                print("  ❌ Invalid qubit numbers")
        
        else:
            print("  ❌ Invalid gate. Use 'H', 'CNOT', or 'done'")
    
    print(f"\n✅ Circuit built with {len(circuit)} gates")
    
    # Add errors
    print("\n🎯 Add initial Pauli errors:")
    print("Available errors: X, Z")
    print("Format: error_type qubit_number")
    print("Type 'done' when finished adding errors")
    
    while True:
        error_input = input(f"\nError {len(errors)+1} (or 'done'): ").strip().upper()
        
        if error_input == 'DONE':
            break
        
        if error_input in ['X', 'Z']:
            try:
                qubit = int(input(f"  Enter qubit number for {error_input} error: "))
                errors.append(PauliError(qubit, error_input))
                print(f"  ✓ Added {error_input} error on qubit {qubit}")
            except ValueError:
                print("  ❌ Invalid qubit number")
        else:
            print("  ❌ Invalid error type. Use 'X', 'Z', or 'done'")
    
    print(f"\n✅ Added {len(errors)} initial errors")
    
    return circuit, errors


def test_custom_circuit():
    """Test the custom circuit"""
    
    circuit, errors = create_custom_circuit()
    
    if not circuit:
        print("\n❌ No circuit built. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("TESTING YOUR CUSTOM CIRCUIT")
    print("=" * 60)
    
    # Show circuit
    print("\n📋 Your circuit:")
    for i, gate in enumerate(circuit):
        if gate.name == "H":
            print(f"  Step {i+1}: H on qubit {gate.qubits[0]}")
        elif gate.name == "CNOT":
            print(f"  Step {i+1}: CNOT (control: {gate.qubits[0]}, target: {gate.qubits[1]})")
    
    # Show initial errors
    print("\n🎯 Initial errors:")
    if errors:
        for error in errors:
            print(f"  {error.type} on qubit {error.qubit}")
    else:
        print("  No initial errors")
    
    # Propagate errors
    trace = propagate_errors(circuit, errors)
    
    # Show propagation
    print("\n🔄 Error propagation:")
    current_errors = {e.qubit: e.type for e in errors}
    
    print(f"\n  Before any gates:")
    max_qubit = max(max(gate.qubits) for gate in circuit) if circuit else 0
    for q in range(max_qubit + 1):
        error = current_errors.get(q, "I")
        print(f"    Qubit {q}: {error}")
    
    for i, step in enumerate(trace):
        gate = step.gate
        new_errors = step.errors_after
        
        print(f"\n  After {gate.name} on qubit(s) {gate.qubits}:")
        for q in range(max_qubit + 1):
            old_error = current_errors.get(q, "I")
            new_error = new_errors.get(q, "I")
            
            if old_error != new_error:
                print(f"    Qubit {q}: {old_error} → {new_error} ⚡")
            else:
                print(f"    Qubit {q}: {new_error}")
        
        current_errors = new_errors.copy()
    
    # Show final result
    print(f"\n🏁 Final errors:")
    for q in range(max_qubit + 1):
        error = current_errors.get(q, "I")
        print(f"  Qubit {q}: {error}")
    
    # Generate ZX diagrams
    print("\n" + "=" * 60)
    print("GENERATING ZX DIAGRAMS")
    print("=" * 60)
    
    try:
        save_complete_visualization(circuit, errors, trace, "custom_circuit")
        print("\n✅ ZX diagrams generated!")
        print("📁 Check for files named 'custom_circuit_step_*.png'")
    except Exception as e:
        print(f"\n❌ Error generating diagrams: {e}")
    
    print("\n" + "=" * 60)
    print("CUSTOM CIRCUIT TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    test_custom_circuit()
