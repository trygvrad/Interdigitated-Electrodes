# Interdigitated-Electrodes

Interdigitated-Electrodes is a python module for calculating the capacitance and electric field surrounding interdigitated electrodes. The code is demonstrated in an interactive web app hosted at https://interdigitated-electrodes.streamlit.app/. 

Examples of use are supplied in jupyter notebooks:

* Example_simple.ipynb - Simple example
* Example_CV_curve.ipynb - Processing of capacitance-voltage curves
* Example_hysteresis_loop(PE).ipynb. - Processing of ferroelectric hysteresis loops
* Examples_different_geometries.ipynb -Advanced geometries
* Example_field_lines.ipynb - Calculation of electric field lines

The main code is contained in:
* infinite_fourier.py
* pair_conformal.py

While the following files are used for validation:
* infinite_conformal.py
* tests.ipynb
* tests_pair.ipynb

A detailed description of the model is available in  https://iopscience.iop.org/article/10.1088/1361-665X/abb4b9/meta
