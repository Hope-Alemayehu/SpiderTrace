#mapping circuit + erros => Zx diagrams
import pyzx as zx 
from spidertrace.circuit import Gate
from spidertrace.engine import TraceStep


def draw_circuit_only(circuit):
    """
    Creates a ZX diagram showing the circuit without any errors.
    
    Args:
        circuit: List of Gate objects representing the full circuit
    
    Returns:
        pyzx Graph object that can be displayed
    """
    g = zx.Graph()
    
    # Determine number of qubits from circuit
    max_qubit = max(max(gate.qubits) for gate in circuit) if circuit else 0
    num_qubits = max_qubit + 1
    
    # Create input and output boundaries
    inputs = []
    outputs = []
    
    for q in range(num_qubits):
        inputs.append(g.add_vertex(zx.VertexType.BOUNDARY, qubit=q, row=0))
        outputs.append(g.add_vertex(zx.VertexType.BOUNDARY, qubit=q, row=len(circuit) * 2 + 1))
    
    # Add gates
    current_row = 1
    
    for i, gate in enumerate(circuit):
        gate_row = current_row
        
        if gate.name == "H":
            q = gate.qubits[0]
            # Add Hadamard gate (Z-spider with H-phase)
            h_vertex = g.add_vertex(zx.VertexType.Z, qubit=q, row=gate_row)
            g.set_phase(h_vertex, 1)  # H phase
            
            # Connect to previous vertex or input
            if i == 0:
                g.add_edge((inputs[q], h_vertex))
            else:
                # Find previous vertex on this qubit
                prev_vertices = [v for v in g.vertices() 
                               if g.qubit(v) == q and g.row(v) < gate_row]
                if prev_vertices:
                    prev = max(prev_vertices, key=lambda v: g.row(v))
                    g.add_edge((prev, h_vertex))
            
            # Connect to output if this is the last gate
            if i == len(circuit) - 1:
                g.add_edge((h_vertex, outputs[q]))
                
        elif gate.name == "CNOT":
            control, target = gate.qubits
            
            # Add CNOT as Z-spider (control) and X-spider (target)
            control_vertex = g.add_vertex(zx.VertexType.Z, qubit=control, row=gate_row)
            target_vertex = g.add_vertex(zx.VertexType.X, qubit=target, row=gate_row)
            
            # Connect control and target
            g.add_edge((control_vertex, target_vertex))
            
            # Connect to previous vertices or inputs
            for q, vertex in [(control, control_vertex), (target, target_vertex)]:
                if i == 0:
                    g.add_edge((inputs[q], vertex))
                else:
                    prev_vertices = [v for v in g.vertices() 
                                   if g.qubit(v) == q and g.row(v) < gate_row]
                    if prev_vertices:
                        prev = max(prev_vertices, key=lambda v: g.row(v))
                        g.add_edge((prev, vertex))
            
            # Connect to outputs if this is the last gate
            if i == len(circuit) - 1:
                g.add_edge((control_vertex, outputs[control]))
                g.add_edge((target_vertex, outputs[target]))
        
        current_row += 2
    
    return g


def draw_initial_errors(circuit, errors):
    """
    Creates a ZX diagram showing the circuit with initial errors.
    
    Args:
        circuit: List of Gate objects representing the full circuit
        errors: List of PauliError objects representing initial errors
    
    Returns:
        pyzx Graph object that can be displayed
    """
    g = draw_circuit_only(circuit)
    
    # Add error annotations
    max_qubit = max(max(gate.qubits) for gate in circuit) if circuit else 0
    num_qubits = max_qubit + 1
    
    # Add errors at the beginning (row 0.5)
    for error in errors:
        qubit = error.qubit
        error_type = error.type
        
        if error_type == "X":
            error_vertex = g.add_vertex(zx.VertexType.X, qubit=qubit, row=0.5)
            g.set_phase(error_vertex, 1)  # X/Z error
        elif error_type == "Z":
            error_vertex = g.add_vertex(zx.VertexType.Z, qubit=qubit, row=0.5)
            g.set_phase(error_vertex, 0)  # Z error
        
        # Connect error vertex to input
        inputs = [v for v in g.vertices() if g.type(v) == zx.VertexType.BOUNDARY and g.row(v) == 0]
        input_vertex = inputs[qubit]
        g.add_edge((input_vertex, error_vertex))
        
        # Connect error to the first gate on this qubit
        gate_vertices = [v for v in g.vertices() 
                         if g.qubit(v) == qubit and g.row(v) > 0.5 
                         and g.type(v) != zx.VertexType.BOUNDARY]
        if gate_vertices:
            first_gate = min(gate_vertices, key=lambda v: g.row(v))
            g.add_edge((error_vertex, first_gate))
    
    return g


def draw_trace_step(circuit, trace_step, step_index):
    """
    Creates a ZX diagram showing the circuit with errors at a specific trace step.
    
    Args:
        circuit: List of Gate objects representing the full circuit
        trace_step: TraceStep object containing gate and errors_after
        step_index: Index of the current step in the trace
    
    Returns:
        pyzx Graph object that can be displayed
    """
    g = zx.Graph()
    
    # Determine number of qubits from circuit
    max_qubit = max(max(gate.qubits) for gate in circuit) if circuit else 0
    num_qubits = max_qubit + 1
    
    # Create input and output boundaries
    inputs = []
    outputs = []
    
    for q in range(num_qubits):
        inputs.append(g.add_vertex(zx.VertexType.BOUNDARY, qubit=q, row=0))
        outputs.append(g.add_vertex(zx.VertexType.BOUNDARY, qubit=q, row=len(circuit) * 2 + 2))
    
    # Add gates up to current step
    current_row = 1
    
    for i, gate in enumerate(circuit):
        gate_row = current_row
        
        if gate.name == "H":
            q = gate.qubits[0]
            # Add Hadamard gate (Z-spider with H-phase)
            h_vertex = g.add_vertex(zx.VertexType.Z, qubit=q, row=gate_row)
            g.set_phase(h_vertex, 1)  # H phase
            
            # Connect to previous vertex or input
            if i == 0:
                g.add_edge((inputs[q], h_vertex))
            else:
                # Find previous vertex on this qubit
                prev_vertices = [v for v in g.vertices() 
                               if g.qubit(v) == q and g.row(v) < gate_row]
                if prev_vertices:
                    prev = max(prev_vertices, key=lambda v: g.row(v))
                    g.add_edge((prev, h_vertex))
            
            # Connect to output if this is the last gate
            if i == len(circuit) - 1:
                g.add_edge((h_vertex, outputs[q]))
                
        elif gate.name == "CNOT":
            control, target = gate.qubits
            
            # Add CNOT as Z-spider (control) and X-spider (target)
            control_vertex = g.add_vertex(zx.VertexType.Z, qubit=control, row=gate_row)
            target_vertex = g.add_vertex(zx.VertexType.X, qubit=target, row=gate_row)
            
            # Connect control and target
            g.add_edge((control_vertex, target_vertex))
            
            # Connect to previous vertices or inputs
            for q, vertex in [(control, control_vertex), (target, target_vertex)]:
                if i == 0:
                    g.add_edge((inputs[q], vertex))
                else:
                    prev_vertices = [v for v in g.vertices() 
                                   if g.qubit(v) == q and g.row(v) < gate_row]
                    if prev_vertices:
                        prev = max(prev_vertices, key=lambda v: g.row(v))
                        g.add_edge((prev, vertex))
            
            # Connect to outputs if this is the last gate
            if i == len(circuit) - 1:
                g.add_edge((control_vertex, outputs[control]))
                g.add_edge((target_vertex, outputs[target]))
        
        current_row += 2
    
    # Add error annotations after the current step
    error_row = len(circuit) * 2 + 1
    for qubit, error_type in trace_step.errors_after.items():
        if error_type == "X":
            error_vertex = g.add_vertex(zx.VertexType.X, qubit=qubit, row=error_row)
            g.set_phase(error_vertex, 1)  # X/Z error
        elif error_type == "Z":
            error_vertex = g.add_vertex(zx.VertexType.Z, qubit=qubit, row=error_row)
            g.set_phase(error_vertex, 0)  # Z error
        elif error_type == "Y":
            # Y error can be represented as both X and Z on same vertex
            error_vertex = g.add_vertex(zx.VertexType.Z, qubit=qubit, row=error_row)
            g.set_phase(error_vertex, 2)  # Y phase
        
        # Connect error vertex to the circuit
        circuit_vertices = [v for v in g.vertices() 
                           if g.qubit(v) == qubit and g.row(v) < error_row 
                           and g.type(v) != zx.VertexType.BOUNDARY]
        if circuit_vertices:
            last_circuit_vertex = max(circuit_vertices, key=lambda v: g.row(v))
            g.add_edge((last_circuit_vertex, error_vertex))
            g.add_edge((error_vertex, outputs[qubit]))
    
    return g


def visualize_complete_trace(circuit, initial_errors, trace):
    """
    Creates a complete visualization showing:
    1. Circuit without errors
    2. Circuit with initial errors
    3. Circuit after each propagation step
    
    Args:
        circuit: List of Gate objects
        initial_errors: List of PauliError objects
        trace: List of TraceStep objects from propagate_errors
    
    Returns:
        List of pyzx Graph objects with titles
    """
    diagrams = []
    
    # 1. Circuit without errors
    circuit_only = draw_circuit_only(circuit)
    diagrams.append((circuit_only, "Circuit (No Errors)"))
    
    # 2. Circuit with initial errors
    with_initial_errors = draw_initial_errors(circuit, initial_errors)
    diagrams.append((with_initial_errors, "Circuit with Initial Errors"))
    
    # 3. Circuit after each propagation step
    for i, step in enumerate(trace):
        title = f"After {step.gate.name} on qubit(s) {step.gate.qubits}"
        diagram = draw_trace_step(circuit, step, i)
        diagrams.append((diagram, title))
    
    return diagrams


def visualize_trace(circuit, trace):
    """
    Creates ZX diagrams for each step in the trace.
    
    Args:
        circuit: List of Gate objects
        trace: List of TraceStep objects from propagate_errors
    
    Returns:
        List of pyzx Graph objects
    """
    diagrams = []
    for i, step in enumerate(trace):
        diagram = draw_trace_step(circuit, step, i)
        diagrams.append(diagram)
    return diagrams


def save_diagram(diagram, filename):
    """
    Save a ZX diagram to file.
    
    Args:
        diagram: pyzx Graph object
        filename: Output filename (should end in .png, .svg, or .pdf)
    """
    zx.draw(diagram, filename)


def save_complete_visualization(circuit, initial_errors, trace, base_filename):
    """
    Save complete visualization with all steps.
    
    Args:
        circuit: List of Gate objects
        initial_errors: List of PauliError objects  
        trace: List of TraceStep objects
        base_filename: Base filename for saved images
    """
    diagrams = visualize_complete_trace(circuit, initial_errors, trace)
    
    for i, (diagram, title) in enumerate(diagrams):
        filename = f"{base_filename}_step_{i}_{title.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.png"
        save_diagram(diagram, filename)
        print(f"Saved: {filename}")
    
    return diagrams