# CE 4011 Assignment 2 - 2D Frame Analysis

This repository contains a robust **Matrix Stiffness Method** solver implemented in pure Python for 2D structural frame analysis. It is designed to handle linear elastic analysis of structures, explicitly computing nodal displacements, member end forces (in local axes), and support reactions. 

A core component of this assignment is the custom-built matrix engine (`matrix_lib`), demonstrating efficient equation solving algorithms (LDLT Crout Factorisation) using sparse/banded architectures (e.g., Skyline Matrix, Banded Matrix) entirely without external heavy dependencies like NumPy.

## Requirements

- **Python 3.x**
- No external libraries (like `numpy` or `scipy`) are required to run the main analysis engines.

## 🚀 How to Run the Program

You can use the program in two primary ways: verifying standard homework problems or analyzing custom structures.

### 1. Running Verification Suites (Tests)
To quickly verify that the application works as expected (and that standard structures check out globally in equilibrium), run the following test scripts from your terminal or IDE:

```bash
# Test 1: Cantilever Frame
python verify_cantilever.py

# Test 2: Complex Portal Frame
python verify_portal.py

# Test 3: Mathematical Verification of Custom Matrix Solvers
python verify_banded.py
```

### 2. Using the Main Application Analyzer
To solve a custom frame structure, you can supply a `.txt` structure parameter file. The application accepts the file via command-line arguments and will output the results either to the terminal or into a new save file.

**Usage syntax:**
```bash
python main.py <input_file.txt> [output_file.txt]
```

**Examples:**
```bash
# Print results directly to the console
python main.py test_cantilever.txt

# Analyze and print results, AND save output to a text file
python main.py test_portal.txt portal_results.txt
```

## 📂 File Structure

- `main.py` : Terminal interface to run analyses.
- `frame_analysis.py` : Core script implementing stiffness generation, assembly, and boundary conditions.
- `read_input.py` : Parses the defined text input formats containing nodes, elements, boundary conditions, and loads.
- `write_output.py` : Formats and outputs the final member internal forces, nodal displacements, and reactions.
- `equilibrium_check.py` : Executes a static global equilibrium check ($\Sigma F_x=0$, $\Sigma F_y=0$) assessing overall model stability.
- `/matrix_lib/`:
    * `banded_matrix.py` : Stores symmetric banded matrices in compact form with LDLT solve algorithms.
    * `skyline_matrix.py` : Handles sparse matrix profiling for varying-bandwidth matrices, reducing RAM usage.
    * `dense_matrix.py` : Complete NxN array solver for cross-verification.
- `uml_hires.puml` / `uml_source.puml` : PlantUML Architecture Diagrams.
