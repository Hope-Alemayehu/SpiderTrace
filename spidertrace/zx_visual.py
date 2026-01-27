#mapping circuit + erros => Zx diagrams
import pyzx as zx 
from spidertrace.circuit import Gate
from spidertrace.engine import TraceStep


def draw_trace_step(circuit, trace_step):
    