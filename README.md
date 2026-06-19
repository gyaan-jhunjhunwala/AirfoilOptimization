# ✈️ Aerodynamic Airfoil Optimization Dashboard
This is an interactive Streamlit web application designed to compute, analyze, and optimize NACA 4-digit airfoil geometries using NeuralFoil.
The dashboard integrates aerodynamic backends with mathematical optimization algorithms (SLSQP) to find ideal airfoil parameters for specific flight conditions in real-time.

**Live App:** https://airfoil-optimization.streamlit.app/

## Features
* **NEW! Aircraft Presets:** Domain-specific targets that automatically set realistic slider ranges and design parameters.  
* **Dynamic Geometry Generation:** Live rendering of NACA 4-digit airfoils using cosine spacing for enhanced resolution at the leading and trailing edges.
* **Performance Sweeps:** Automated angle of attack sweeps to analyze lift-to-drag ratios across custom ranges.
* **Mathematical Optimization:** Integrated scipy.optimize pipeline to maximize the lift-to-drag ratio based on user-defined constraints (thickness, camber, and flight envelope restrictions).
* **Polished UI/UX:** Clean, responsive dark-themed dashboard built entirely with Streamlit and custom CSS layout components.

## Installation & Setup
To run this dashboard locally, ensure you have Python 3.8+ installed, then follow these steps:
1. **Clone the repository:**
```
git clone https://github.com/gyaan-jhunjhunwala/AirfoilOptimization.git
cd AirfoilOptimization
```
2. **Create and activate a virtual environment:**
```
# On Windows
python -m venv venv
venv\Scripts\activate
```
```
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
3. **Install the required dependencies:**
```
pip install -r requirements.txt
```
4. **Launch the Streamlit app:**
```
streamlit run app.py
```
