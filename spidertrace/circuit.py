# defines gates and circuits as sequence

from dataclasses import dataclass

@dataclass
class Gate:
    name: str   #"H" or "CNOT" only
    qubits: tuple[int]  #tuple of qubit indices; length=1 for H and 2 for CNOT
    

    """
    Example
    circuit_sequence = [
    Gate("H", (0,)),
    Gate("CNOT", (0,1)),
    ]
    """