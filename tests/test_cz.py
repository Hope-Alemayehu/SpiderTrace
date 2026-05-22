#!/usr/bin/env python3
"""
Unit tests for CZ gate error propagation.

CZ conjugation rules (verified algebraically):
  CZ (X_c ⊗ I)  CZ = X_c ⊗ Z_t
  CZ (I  ⊗ X_t) CZ = Z_c ⊗ X_t
  CZ (Z_c ⊗ I)  CZ = Z_c ⊗ I
  CZ (I  ⊗ Z_t) CZ = I  ⊗ Z_t
  CZ (Y_c ⊗ I)  CZ = Y_c ⊗ Z_t
  CZ (I  ⊗ Y_t) CZ = Z_c ⊗ Y_t
  CZ (X_c ⊗ X_t) CZ = Y_c ⊗ Y_t  (NOT Z⊗Z; see algebra below)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spidertrace.circuit import Gate
from spidertrace.engine import propagate_errors
from spidertrace.error import PauliError


def _run(errors_list):
    circuit = [Gate("CZ", (0, 1))]
    errors = [PauliError(q, p) for q, p in errors_list]
    trace = propagate_errors(circuit, errors)
    return trace[0].errors_after


def test_cz_x_on_control():
    """X on control -> X_c Z_t"""
    result = _run([(0, "X")])
    assert result == {0: "X", 1: "Z"}, f"Expected {{0:'X', 1:'Z'}}, got {result}"
    print("PASS: X on control -> X_c Z_t")


def test_cz_x_on_target():
    """X on target -> Z_c X_t"""
    result = _run([(1, "X")])
    assert result == {0: "Z", 1: "X"}, f"Expected {{0:'Z', 1:'X'}}, got {result}"
    print("PASS: X on target -> Z_c X_t")


def test_cz_z_on_control():
    """Z on control -> unchanged"""
    result = _run([(0, "Z")])
    assert result == {0: "Z"}, f"Expected {{0:'Z'}}, got {result}"
    print("PASS: Z on control -> unchanged")


def test_cz_z_on_target():
    """Z on target -> unchanged"""
    result = _run([(1, "Z")])
    assert result == {1: "Z"}, f"Expected {{1:'Z'}}, got {result}"
    print("PASS: Z on target -> unchanged")


def test_cz_y_on_control():
    """Y on control -> Y_c Z_t"""
    result = _run([(0, "Y")])
    assert result == {0: "Y", 1: "Z"}, f"Expected {{0:'Y', 1:'Z'}}, got {result}"
    print("PASS: Y on control -> Y_c Z_t")


def test_cz_y_on_target():
    """Y on target -> Z_c Y_t"""
    result = _run([(1, "Y")])
    assert result == {0: "Z", 1: "Y"}, f"Expected {{0:'Z', 1:'Y'}}, got {result}"
    print("PASS: Y on target -> Z_c Y_t")


def test_cz_simultaneous_x_both():
    """
    X on both qubits simultaneously.

    Algebra: CZ(X_c ⊗ X_t)CZ = (X_c⊗Z_t)(Z_c⊗X_t)
             = (X·Z)_c ⊗ (Z·X)_t
             = (-iY_c) ⊗ (iY_t)
             = Y_c ⊗ Y_t  (global phase -1 ignored in stabilizer formalism)
    """
    result = _run([(0, "X"), (1, "X")])
    assert result == {0: "Y", 1: "Y"}, f"Expected {{0:'Y', 1:'Y'}}, got {result}"
    print("PASS: simultaneous X on both -> Y_c Y_t")


def test_cz_no_errors():
    """No errors -> no errors"""
    result = _run([])
    assert result == {}, f"Expected {{}}, got {result}"
    print("PASS: no errors -> no errors")


def test_cz_symmetry():
    """CZ is symmetric: swapping control and target gives same result"""
    circuit = [Gate("CZ", (1, 0))]  # roles swapped
    trace = propagate_errors(circuit, [PauliError(1, "X")])
    result = trace[0].errors_after
    assert result == {1: "X", 0: "Z"}, f"Expected {{1:'X', 0:'Z'}}, got {result}"
    print("PASS: CZ symmetry (X on qubit 1 as 'control' -> X_1 Z_0)")


def main():
    print("CZ Gate Test Suite")
    print("=" * 50)
    try:
        test_cz_x_on_control()
        test_cz_x_on_target()
        test_cz_z_on_control()
        test_cz_z_on_target()
        test_cz_y_on_control()
        test_cz_y_on_target()
        test_cz_simultaneous_x_both()
        test_cz_no_errors()
        test_cz_symmetry()
        print("\n" + "=" * 50)
        print("SUCCESS: All CZ tests passed!")
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
