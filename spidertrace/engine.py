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

        # Full CNOT conjugation via the symplectic (x, z) representation
        # (I=00, X=10, Z=01, Y=11). This covers every input Pauli -- including
        # Y on the control or target, which the earlier X/Z-only special-casing
        # silently dropped. The X-control/Z-target rules below are the standard
        # CNOT symplectic update; the previously handled cases are a subset:
        #     x_t' = x_t XOR x_c   (X on control spreads X to target)
        #     z_c' = z_c XOR z_t   (Z on target spreads Z to control)
        #     x_c, z_t unchanged
        _to_xz = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
        _to_p = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}

        xc, zc = _to_xz[new_errors.get(c, "I")]
        xt, zt = _to_xz[new_errors.get(t, "I")]

        new_c = _to_p[(xc, zc ^ zt)]
        new_t = _to_p[(xt ^ xc, zt)]

        for q, p in ((c, new_c), (t, new_t)):
            if p == "I":
                new_errors.pop(q, None)
            else:
                new_errors[q] = p

    if gate.name == "CZ":
        c, t = gate.qubits[0], gate.qubits[1]

        p_c = new_errors.get(c, "I")
        p_t = new_errors.get(t, "I")

        # Full CZ conjugation table: CZ (p_c ⊗ p_t) CZ, ignoring global phase.
        # Derived by composing individual single-qubit rules and multiplying Paulis.
        cz_table = {
            ("I", "I"): ("I", "I"),
            ("X", "I"): ("X", "Z"),
            ("I", "X"): ("Z", "X"),
            ("Z", "I"): ("Z", "I"),
            ("I", "Z"): ("I", "Z"),
            ("Y", "I"): ("Y", "Z"),
            ("I", "Y"): ("Z", "Y"),
            ("X", "X"): ("Y", "Y"),
            ("X", "Z"): ("X", "I"),
            ("Z", "X"): ("I", "X"),
            ("X", "Y"): ("Y", "X"),
            ("Y", "X"): ("X", "Y"),
            ("Z", "Z"): ("Z", "Z"),
            ("Z", "Y"): ("I", "Y"),
            ("Y", "Z"): ("Y", "I"),
            ("Y", "Y"): ("X", "X"),
        }

        q_c, q_t = cz_table[(p_c, p_t)]

        if q_c == "I":
            new_errors.pop(c, None)
        else:
            new_errors[c] = q_c

        if q_t == "I":
            new_errors.pop(t, None)
        else:
            new_errors[t] = q_t

    return new_errors