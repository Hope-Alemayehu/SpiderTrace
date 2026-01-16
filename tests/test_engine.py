from spidertrace.circuit import Gate
from spidertrace.engine import propagate_errors
from spidertrace.error import PauliError

print("TEST_ENGINE MODULE LOADED")

def print_trace(trace):
    for step in trace:
        print(f"{step.gate} -> {step.errors_after}")
        
#test 1: X through H
circuit = [Gate("H",(0,))]
errors = [PauliError(0,"X")]

trace = propagate_errors(circuit, errors)

for step in trace:
    print(step.gate, step.errors_after)