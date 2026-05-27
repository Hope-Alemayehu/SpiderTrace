# SpiderTrace

SpiderTrace is a small tool that traces how Pauli X and Z errors propagate through stabilizer circuits by rewriting ZX diagrams.

## Features
-  **Error Propagation Engine**: Accurately tracks Pauli errors through quantum circuits
-  **ZX Diagram Visualization**: Generate visual representations of circuits and error propagation
-  **Interactive Testing**: Build and test your own custom circuits
-  **Comprehensive Output**: See before/after states and step-by-step propagation

## Constraints 
SpiderTrace supports:
- Clifford group only
- Gates: H, CNOT, CZ
- Errors: single-qubit X, Z, or Y (Y errors arise naturally from propagation)
- Representation: ZX diagram
- Goal: trace Pauli error propagation

SpiderTrace will not:
- Simulate quantum states
- Support T / arbitrary rotations
- Do decoding or correction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hope-Alemayehu/SpiderTrace.git
cd SpiderTrace
```

2. Install dependencies:
```bash
pip install pyzx
```

## Quick Start

### Test the Built-in Examples
```bash
# Run basic tests
python test_simple.py

# Generate ZX diagrams for all propagation steps
python display_all_zx.py
```

### Build and Test Your Own Circuits
```bash
python test_custom.py
```
Follow the interactive prompts to:

1. Build your circuit with H, CNOT, and CZ gates
2. Add initial X or Z errors
3. See the complete error propagation
4. Generate ZX diagrams

## Usage Examples

### Python API
```python
from spidertrace.circuit import Gate
from spidertrace.engine import propagate_errors
from spidertrace.error import PauliError

# Create a circuit
circuit = [
    Gate("H", (0,)),
    Gate("CNOT", (0, 1)),
    Gate("CZ", (0, 1))
]

# Add initial errors
errors = [PauliError(0, "X")]

# Propagate errors
trace = propagate_errors(circuit, errors)

# View results
for step in trace:
    print(f"After {step.gate.name}: {step.errors_after}")
```

### ZX Diagram Generation
```python
from spidertrace.zx_visual import save_complete_visualization

# Generate complete visualization
save_complete_visualization(circuit, errors, trace, "my_circuit")
```

## Formal Definition
(as formal as it gets)

- Error is a symbolic Pauli operator attached to a wire/spider

- Propagation is conjugation by gates

- Rewrite rules are the algebraic identities you can apply to transform diagrams while preserving the overall unitary or effect.

Hadamard gate

```text
Through H:
H : X ↔ Z 

is the same as:
X → H X H = Z
Z → H Z H = X
```

CNOT gate — for CNOT with control `c` and target `t`:

```text
X_c => X_c X_t
Z_t => Z_c Z_t
X_t => X_t
Z_c => Z_c
```

CZ gate — for CZ with qubits `c` and `t`:

```text
X_c => X_c Z_t
X_t => Z_c X_t
Z_c => Z_c
Z_t => Z_t
```
Meaning: CZ is symmetric — X on either qubit picks up a Z on the other qubit.

## Error Propagation Rules

### Hadamard Gate (H)
- **X error** → **Z error** (Pauli swap)
- **Z error** → **X error** (Pauli swap)
- **Y error** → **Y error** (unchanged)

### CNOT Gate
For CNOT with control `c` and target `t`:

| Input Error | Output Error |
|-------------|--------------|
| X on control | X on control + X on target |
| Z on target | Z on control + Z on target |
| X on control + Z on target | Y on control + Y on target |
| X on target | X on target (unchanged) |
| Z on control | Z on control (unchanged) |

### CZ Gate

For CZ with qubits `c` and `t` (symmetric gate):

| Input Error | Output Error |
|-------------|--------------|
| X on c | X on c + Z on t |
| X on t | Z on c + X on t |
| Z on c | Z on c (unchanged) |
| Z on t | Z on t (unchanged) |
| X on c + X on t | Y on c + Y on t |
| Y on c | Y on c + Z on t |
| Y on t | Z on c + Y on t |

In ZX calculus, CZ is represented as two Z-spiders connected by a Hadamard edge.

## Project Structure

```
SpiderTrace/
├── spidertrace/
│   ├── __init__.py          # Package exports
│   ├── circuit.py           # Gate definitions (H, CNOT, CZ)
│   ├── engine.py            # Error propagation engine
│   ├── error.py             # Pauli error definitions
│   ├── zx_visual.py         # ZX diagram generation
│   ├── display_all_zx.py    # Display ZX diagrams
│   └── utils.py             # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_simple.py       # Unit tests
│   ├── test_engine.py       # Engine tests
│   ├── test_custom.py       # Custom circuit tests
│   └── test_zx_visual.py    # ZX visualization tests
├── test_simple.py           # Run all tests
├── test_custom.py           # Interactive circuit builder
├── pyproject.toml           # Python package configuration
└── README.md                # This file
```

## Examples

### Example 1: Single Hadamard
```python
circuit = [Gate("H", (0,))]
errors = [PauliError(0, "X")]

# Result: X → Z on qubit 0
```

### Example 2: CNOT Spreading
```python
circuit = [Gate("CNOT", (0, 1))]
errors = [PauliError(0, "X")]

# Result: X on qubit 0 → X on both qubits 0 and 1
```

### Example 3: CZ Phase Kickback

```python
circuit = [Gate("CZ", (0, 1))]
errors = [PauliError(0, "X")]

# Result: X on qubit 0 → X on qubit 0 + Z on qubit 1
```

### Example 4: Multi-Gate Circuit
```python
circuit = [
    Gate("H", (0,)),
    Gate("CNOT", (0, 1))
]
errors = [PauliError(0, "X")]

# Step 1: X → Z on qubit 0 (after H)
# Step 2: Z spreads to qubit 1 (after CNOT)
# Final: Z on both qubits
```

## Testing

Run the test suite to verify functionality:
```bash
python test_simple.py
```

This tests:
- Hadamard error propagation (X ↔ Z)
- CNOT error spreading
- CZ error propagation and phase kickback
- Error cancellation and Y error creation
- Multi-gate circuits

## ZX Diagram Output

The tool generates PNG files showing:
1. **Clean circuit** - No errors
2. **Initial errors** - Errors injected at circuit start
3. **After each gate** - Step-by-step propagation
4. **Final state** - Complete error distribution

## Contributing

Feel free to extend SpiderTrace with:

- Additional Clifford gates (S, SWAP, etc.)
- More error types
- Enhanced visualization
- Performance optimizations


## Acknowledgments

Built using [PyZX](https://github.com/Quantomatic/pyzx) for ZX diagram visualization.
