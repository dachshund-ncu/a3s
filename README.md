# a3s
Python program, that makes FFT from 4096 point autocorrelation function - it is meant to be used with the output data from Niculaus Copernicus University autocorrelator, that is working with 32-meter Radio Telescope, located in Piwnice near Toruń, Poland.

current version: 1.002

Requirements:

- numpy

- mpmath

- astropy

- barycorrpy


Program usage:
```bash
a3s.py list_of_.DAT_files
```

Output:

Script returns file named ```WYNIK.DAT``` with spectra
