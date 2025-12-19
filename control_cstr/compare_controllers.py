#!/usr/bin/env python3
"""
Comparison plot script for NMPC, CT, T2D2, and T3D3 controllers.

This script loads the saved control loop data from all four controllers
and creates comparison plots showing states and inputs on the same graph.
"""

import os
import sys
from pathlib import Path
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def load_controller_results(results_dir: Path, controller_name: str):
    """Load saved results for a controller.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing the results files
    controller_name : str
        Name of the controller (NMPC, CT, T2D2, T3D3)
    
    Returns
    -------
    dict or None
        Dictionary with time, states, inputs, etc., or None if file not found
    """
    results_file = results_dir / f'{controller_name}_results.pkl'
    if not results_file.exists():
        print(f"Warning: {results_file} not found. Skipping {controller_name}.")
        return None
    
    return joblib.load(results_file.as_posix())


def plot_comparison(results_dict: dict, figures_dir: Path, sim_setup: dict = None):
    """Create comparison plots for all controllers.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping controller names to their results
    figures_dir : Path
        Directory to save the comparison plots
    sim_setup : dict, optional
        Simulation setup containing input bounds
    """
    # Define consistent colors and line styles for each controller
    styles = {
        'NMPC': {'color': 'black', 'linestyle': '-', 'linewidth': 2, 'label': 'NMPC'},
        'CT': {'color': '#1f77b4', 'linestyle': ':', 'linewidth': 2, 'label': 'C'},
        'T2D2': {'color': '#d62728', 'linestyle': '-.', 'linewidth': 2, 'label': r'$\bar{z}(t-T_\mathrm{s})$'},
        'T3D3': {'color': 'orange', 'linestyle': '--', 'linewidth': 2, 'label': r'$\hat{z}(t)$'},
    }
    
    # State names with units
    state_names = ['CA1', 'T1', 'CA2', 'T2', 'CB1', 'CB2', 'CU1', 'CU2']
    state_units = ['mol/m³', 'K', 'mol/m³', 'K', 'mol/m³', 'mol/m³', 'mol/m³', 'mol/m³']
    
    # Input names with units
    input_names = ['F', 'L', 'Tc1', 'Tc2']
    input_units = ['m³/s', 'm³/s', 'K', 'K']
    
    # Get reference trajectory from any controller (they should all have the same)
    reference_ns = None
    for name, data in results_dict.items():
        if data is not None and data.get('reference_ns') is not None:
            reference_ns = data['reference_ns']
            break
    
    # ==================== STATES COMPARISON ====================
    fig_states, axs_states = plt.subplots(8, 1, figsize=(14, 20), sharex=True)
    
    for i, state_name in enumerate(state_names):
        ax = axs_states[i]
        
        # Plot reference if available
        if reference_ns is not None:
            time_ref = np.arange(reference_ns.shape[1])
            ax.plot(time_ref, reference_ns[i, :], 'g--', linewidth=3, 
                   label='Reference')
        
        # Plot each controller's trajectory
        for name in ['NMPC', 'CT', 'T2D2', 'T3D3']:
            if name in results_dict and results_dict[name] is not None:
                data = results_dict[name]
                time = data['time']
                states = data['states']
                
                ax.plot(time, states[i, :], 
                       color=styles[name]['color'],
                       linestyle=styles[name]['linestyle'],
                       linewidth=styles[name]['linewidth'],
                       label=styles[name]['label'])
        
        ax.set_ylabel(f'{state_name} [{state_units[i]}]', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    axs_states[-1].set_xlabel('Time step', fontsize=11)
    fig_states.suptitle('Controller Comparison: States vs References', 
                       fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    states_filename_pdf = figures_dir / 'comparison_states.pdf'
    states_filename_svg = figures_dir / 'comparison_states.svg'
    fig_states.savefig(states_filename_pdf.as_posix(), dpi=200, bbox_inches='tight')
    fig_states.savefig(states_filename_svg.as_posix(), bbox_inches='tight')
    print(f"Saved states comparison to {states_filename_pdf}")
    print(f"Saved states comparison to {states_filename_svg}")
    
    # ==================== INPUTS COMPARISON ====================
    fig_inputs, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Get input bounds from sim_setup if available
    u_min_ns = None
    u_max_ns = None
    if sim_setup is not None:
        # Try to get non-scaled bounds
        u_min_ns = sim_setup.get('u_min_ns')
        u_max_ns = sim_setup.get('u_max_ns')
        
        # If not available, try to inverse transform from scaled bounds
        if u_min_ns is None and 'u_min' in sim_setup:
            try:
                import joblib
                scalerU = joblib.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'scalerU.pkl'))
                u_min_ns = scalerU.inverse_transform(sim_setup['u_min'].reshape(1, -1))[0]
                u_max_ns = scalerU.inverse_transform(sim_setup['u_max'].reshape(1, -1))[0]
            except:
                pass
    
    # Determine time range for plotting bounds
    max_time = 0
    for name, data in results_dict.items():
        if data is not None:
            max_time = max(max_time, data['time'][-1])
    
    # Formatter for scientific notation (multiply values by 1e4 and show as ×10⁻⁴)
    def scientific_formatter_1e4(x, pos):
        return f'{x*1e4:.1f}'
    
    for i, input_name in enumerate(input_names):
        # Plot input bounds if available
        if u_min_ns is not None and u_max_ns is not None:
            time_bounds = [0, max_time]
            # axs[i].plot(time_bounds, [u_min_ns[i], u_min_ns[i]], 
            #            color='purple', linestyle='--', linewidth=2, 
            #            label='Bounds', alpha=0.7)
            axs[i].plot(time_bounds, [u_max_ns[i], u_max_ns[i]], 
                       color='purple', linestyle='--', linewidth=2,label='Bounds', alpha=0.7)
        
        # Plot each controller's input trajectory
        for name in ['NMPC', 'CT', 'T2D2', 'T3D3']:
            if name in results_dict and results_dict[name] is not None:
                data = results_dict[name]
                inputs = data['inputs']
                
                # Handle different input array shapes (some have T+1, some have T)
                if inputs.shape[1] == data['time'].shape[0]:
                    time_input = data['time']
                else:
                    time_input = data['time'][:-1]
                
                axs[i].plot(time_input, inputs[i, :], 
                           color=styles[name]['color'],
                           linestyle=styles[name]['linestyle'],
                           linewidth=styles[name]['linewidth'],
                           label=styles[name]['label'])
        
        # Set y-axis label with units
        if input_name in ['F', 'L']:
            # Use scientific notation for flow rates
            axs[i].set_ylabel(f'{input_name} [×10⁻⁴ {input_units[i]}]', fontsize=11, fontweight='bold')
            axs[i].yaxis.set_major_formatter(FuncFormatter(scientific_formatter_1e4))
        else:
            axs[i].set_ylabel(f'{input_name} [{input_units[i]}]', fontsize=11, fontweight='bold')
        
        axs[i].grid(True, alpha=0.3)
        axs[i].legend(loc='best', fontsize=9, framealpha=0.9)
    
    axs[-1].set_xlabel('Time step', fontsize=11)
    fig_inputs.suptitle('Controller Comparison: Control Inputs', 
                       fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    inputs_filename_pdf = figures_dir / 'comparison_inputs.pdf'
    inputs_filename_svg = figures_dir / 'comparison_inputs.svg'
    fig_inputs.savefig(inputs_filename_pdf.as_posix(), dpi=200, bbox_inches='tight')
    fig_inputs.savefig(inputs_filename_svg.as_posix(), bbox_inches='tight')
    print(f"Saved inputs comparison to {inputs_filename_pdf}")
    print(f"Saved inputs comparison to {inputs_filename_svg}")
    
    # ==================== OBJECTIVE VALUES TABLE ====================
    print("\n" + "="*70)
    print("CONTROLLER PERFORMANCE COMPARISON")
    print("="*70)
    print(f"{'Controller':<12} {'Total Objective':<18} {'State Term':<18} {'Input Term':<18}")
    print("-"*70)
    
    for name in ['NMPC', 'CT', 'T2D2', 'T3D3']:
        if name in results_dict and results_dict[name] is not None:
            data = results_dict[name]
            obj = data.get('objective', 'N/A')
            state = data.get('state_term', 'N/A')
            inp = data.get('input_term', 'N/A')
            
            if isinstance(obj, (int, float)):
                print(f"{name:<12} {obj:<18.2f} {state:<18.2f} {inp:<18.2f}")
            else:
                print(f"{name:<12} {obj:<18} {state:<18} {inp:<18}")
    print("="*70 + "\n")


def main():
    """Main execution function."""
    # Get project root (parent of control_cstr)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Directories
    results_dir = script_dir
    figures_dir = project_root / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load simulation setup to get input bounds
    sim_setup_path = script_dir / 'sim_setup.pkl'
    sim_setup = None
    if sim_setup_path.exists():
        sim_setup = joblib.load(sim_setup_path.as_posix())
    
    # Load results from all controllers
    controller_names = ['NMPC', 'CT', 'T2D2', 'T3D3']
    results_dict = {}
    
    for name in controller_names:
        results_dict[name] = load_controller_results(results_dir, name)
    
    # Check if we have at least one controller's results
    if not any(v is not None for v in results_dict.values()):
        print("Error: No controller results found. Please run the individual")
        print("controller scripts (NMPC.py, CT.py, T2D2.py, T3D3.py) first.")
        return
    
    # Create comparison plots
    plot_comparison(results_dict, figures_dir, sim_setup)
    
    print("\nComparison plots generated successfully!")


if __name__ == '__main__':
    main()

