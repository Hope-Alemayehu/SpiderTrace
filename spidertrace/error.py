# define the pauli error and propagation rules

from dataclasses import dataclass

@dataclass
class PauliError:
    qubit: int
    type: str  #could be "X" or "Z"
    
    """Example
    errors = [PauliError(0, "X"), PauliError(1, "Z")]
    """
