#!/usr/bin/env python3
"""
Display all ZX diagrams together in one window
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyzx as zx
from spidertrace.circuit import Gate
from spidertrace.engine import propagate_errors
from spidertrace.error import PauliError
from spidertrace.zx_visual import draw_circuit_only, draw_initial_errors, draw_trace_step


def display_all_zx_diagrams():
    """Generate and display all ZX diagrams together"""
    
    # Example circuit: H on qubit 0, then CNOT(0,1)
    circuit = [
        Gate("H", (0,)),
        Gate("CNOT", (0, 1))
    ]
    
    # Initial X error on qubit 0
    errors = [PauliError(0, "X")]
    
    # Get propagation trace
    trace = propagate_errors(circuit, errors)
    
    print("=" * 80)
    print("SPIDERTRACE: ALL ZX DIAGRAMS IN ONE DISPLAY")
    print("=" * 80)
    
    print("\n📋 CIRCUIT: H on qubit 0, then CNOT(0,1)")
    print("🎯 INITIAL ERROR: X on qubit 0")
    
    # Generate all diagrams
    diagrams = []
    titles = []
    
    # 1. Circuit without errors
    print("\n" + "="*60)
    print("DIAGRAM 1: CIRCUIT WITHOUT ERRORS")
    print("="*60)
    diagram1 = draw_circuit_only(circuit)
    diagrams.append(diagram1)
    titles.append("1. Clean Circuit (No Errors)")
    print("✓ Generated: Clean circuit structure")
    
    # 2. Circuit with initial errors
    print("\n" + "="*60)
    print("DIAGRAM 2: CIRCUIT WITH INITIAL ERRORS")
    print("="*60)
    diagram2 = draw_initial_errors(circuit, errors)
    diagrams.append(diagram2)
    titles.append("2. Initial X Error on Qubit 0")
    print("✓ Generated: X error injected at beginning")
    
    # 3. After H gate
    print("\n" + "="*60)
    print("DIAGRAM 3: AFTER HADAMARD GATE")
    print("="*60)
    diagram3 = draw_trace_step(circuit, trace[0], 0)
    diagrams.append(diagram3)
    titles.append("3. After H: X → Z Transformation")
    print("✓ Generated: X error becomes Z error")
    
    # 4. After CNOT gate
    print("\n" + "="*60)
    print("DIAGRAM 4: AFTER CNOT GATE")
    print("="*60)
    diagram4 = draw_trace_step(circuit, trace[1], 1)
    diagrams.append(diagram4)
    titles.append("4. After CNOT: Z Spreads to Both Qubits")
    print("✓ Generated: Z error spreads to qubit 1")
    
    # Display all diagrams together
    print("\n" + "="*80)
    print("DISPLAYING ALL ZX DIAGRAMS TOGETHER")
    print("="*80)
    
    try:
        # Try to display using pyzx's built-in display
        for i, (diagram, title) in enumerate(zip(diagrams, titles)):
            print(f"\n{title}:")
            print("-" * len(title))
            
            # Convert to basic circuit representation for text display
            try:
                # Get a simple representation
                zx.draw(diagram, f"temp_diagram_{i}.png")
                print(f"→ Saved as temp_diagram_{i}.png")
                print("→ Open this file to see the ZX diagram")
            except:
                # Fallback to basic info
                vertices = list(diagram.vertices())
                print(f"→ Diagram has {len(vertices)} vertices")
                print("→ Use pyzx.draw() to visualize")
                
    except Exception as e:
        print(f"Display error: {e}")
        print("Falling back to file saving...")
    
    # Save all diagrams for manual viewing
    print("\n" + "="*80)
    print("SAVING ALL DIAGRAMS FOR VIEWING")
    print("="*80)
    
    filenames = []
    for i, (diagram, title) in enumerate(zip(diagrams, titles)):
        filename = f"zx_display_{i+1}_{title.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '').replace(',', '')}.png"
        zx.draw(diagram, filename)
        filenames.append(filename)
        print(f"✓ Saved: {filename}")
    
    print("\n" + "="*80)
    print("COMPLETE ZX VISUALIZATION READY!")
    print("="*80)
    print("All 4 diagrams have been generated and saved:")
    print()
    for i, (filename, title) in enumerate(zip(filenames, titles)):
        print(f"  {i+1}. {filename}")
        print(f"     → {title}")
    
    print(f"\n🎯 Open all {len(filenames)} files to see the complete error propagation!")
    print("📊 Each diagram shows a different stage of the process:")
    print("   1. Clean circuit structure")
    print("   2. Initial error injection") 
    print("   3. Error transformation after H gate")
    print("   4. Error spreading after CNOT gate")
    
    print("\n" + "=" * 80)
    
    return filenames


if __name__ == "__main__":
    try:
        filenames = display_all_zx_diagrams()
        print(f"\n🎉 Success! Generated {len(filenames)} ZX diagrams for viewing.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
