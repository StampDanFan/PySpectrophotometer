# PySpectrophotometer
Python/Pygame Application for Spectrum Analysis for a Spectrophotometer.

Main Menu
- Use Left and Right Arrow Keys to switch between analyzed graphs
- Press Enter to see the name of the image.
- Press Space to toggle between intensity and absorbance.
- Press a to see the calibration graph.

Drag and Drop Images onto the window to analyze them (first image will be calibration image, should be water blank)
1. Rotate with - + to align spectrum to be horizontal.
2. Move with Arrow Keys & use - + to zoom. Click twice (topleft and bottomright corner) to select the region the spectrum is in.
3. Intensity of Light graph shown. Click at two x-values with "landmarks" to align wavelengths between images.

Written in Python 3.9 with Pygame 2.1.0.
