#core propagation rules

from typing import Dict
from spidertrace.circuit import Gate

class TraceStep:
    def __init__(self, gate, errors_after):
        self.gate = gate
        self.errors_after = errors_after
   
   
def propagate_errors(circuit_sequence, errors):
    """
    circuit sequence: list of Gate objects
    errors: list of PauliError orbjects
    returns: list of TraceStep Objects
    """
    trace = []
    current_errors = {e.qubit: e.type for e in errors}
    
    for gate in circuit_sequence:
        #apply gate_specific rewrite rules to current errors
        new_errors = apply_gate_rules(gate, current_errors)
        trace.append(TraceStep(gate, new_errors.copy()))
        current_errors = new_errors
        
    return trace
    
def apply_gate_rules(gate: Gate, errors: Dict[int,str])-> Dict[int,str]:
    """yes
    Applies Pauli propagation rules for a single gate.
    
    gate: Gate object
    errors: dict mapping qubit index -> Pauli type ("X" or "Z")
    
    Returns a new dict with updated
    """
    new_errors = errors.copy()
    
    if gate.name == "H":
        q = gate.qubits[0]
        
        if q in new_errors:
            if new_errors[q] == "X":
                new_errors[q] = "Z"
            elif new_errors[q] == "Z":
                new_errors[q] = "X"
            elif new_errors[q] == "Y":
                pass
    
    if gate.name == "CNOT":
        c, t = gate.qubits[0], gate.qubits[1] 
        
        # Check for simultaneous X on control and Z on target
        x_on_control = c in new_errors and new_errors[c] == "X"
        z_on_target = t in new_errors and new_errors[t] == "Z"
        
        if x_on_control and z_on_target:
            # X on control + Z on target = Y on both
            new_errors[c] = "Y"
            new_errors[t] = "Y"
        else:
            # Handle individual cases
            # X on control spreads to target
            if x_on_control:
                if t in new_errors:
                    if new_errors[t] == "X":
                        # X * X = I (cancel out)
                        del new_errors[t]
                    elif new_errors[t] == "Z":
                        # X * Z = Y
                        new_errors[t] = "Y"
                    else:
                        # No error on target, add X
                        new_errors[t] = "X"
                else:
                    new_errors[t] = "X"
            
            # Z on target spreads to control
            if z_on_target:
                if c in new_errors:
                    if new_errors[c] == "Z":
                        # Z * Z = I (cancel out)
                        del new_errors[c]
                    elif new_errors[c] == "X":
                        # Z * X = Y
                        new_errors[c] = "Y"
                    else:
                        # No error on control, add Z
                        new_errors[c] = "Z"
                else:
                    new_errors[c] = "Z"
    
    return new_errors