README for gradient FLORIS

Installation instructions MAC
-----------------------------
- system requirements
    gfortran
    gcc
    python 2.7.x
    numpy
    openmdao v0.10.3.2
- put all files in desired directory
- run the following commands:
    $ gfortran -c adBuffer.f
    $ gcc -c adStack.c
    $ f2py -c --opt=-O2 -m _floris floris.f90 adBuffer.o adStack.o
    
    
Installation instructions Windows
---------------------------------
- system requirements
    gfortran
    gcc
    mingw
    python 2.7.x
    numpy
    openmdao v0.10.3.2
- put all files in desired directory
- run the following commands:
    $ gfortran -c adBuffer.f
    $ gcc -c adStack.c
    $ python \your\path\to\f2py.py -c --opt=-O2 --compiler=mingw32 --fcompiler=gfortran -m _floris floris.f90 adBuffer.o adStack.o
        (most likely your path is C:\python27\Scripts\f2py.py)
- if you get an error in the line "as=b['args']" try to update numpy 
    ($ pip install numpy --upgrade)
- run the example in an openmdao environment
    $ python example_call.py