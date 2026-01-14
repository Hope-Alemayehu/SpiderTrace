# SpiderTrace

SpiderTRace is a small tool that traces how Pauli X and Z errors propagate through stablizre circuits by rewritting ZX diagrams.

## Constraints 
SpiderTrace will support:
- Clifford group only
- Gates: H, CNOT
- Errors: single X or Z
- Representation: ZX diagram
- Goal: trace Pauli error Propagation

S*5derTrace will not:
- simulate quantum states
- Support T / arbitrary rotations
- Do decoding or correction

## Formal Definition
(as formal as it gets)

- Error is a symbolic pauli operators attached to a wire/spider

- Propagation is conjugation by gates

- Rewrite rules are the algebric identities you can apply to transform diagrams while preserving the overall unitary or effect.

Hadmard gate

```bash
 Through H:
 H : X ↔ Z 

 is the same as:
 X → H X H = Z
 Z → H Z H = X

```

CNOT gate
For Cnot with control c and target t

```bash
 X_c => X_c X_t
 Z_t => Z_c Z_t
 X_t => X_t
 Z_c => Z_c
```
Meaning: 
1. X on control spreads to target
2. Z on target spreads to control
3. X on target and Z on control stay the same
