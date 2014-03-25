=============
sphere readme
=============
``sphere`` is a 3D discrete element method algorithm utilizing CUDA.

License
-------
``sphere`` is licensed under the GNU General Public License, v.3.
See `LICENSE.txt <LICENSE.txt>`_ for more information.

Important release notes
-----------------------
2013-03-13: Sphere has been updated to work with CUDA 5.0 or newer *only*.

2014-01-20: Version fingerprints have been added to the input/output binary
files, and causes old files to be incompatible with either ``sphere`` or
``sphere.py``.

2014-01-25: The description of the installation procedure is moved to the
general documentation.

2014-03-09: The main sphere class (formerly ``spherebin``) has been renamed to
``sim``.

2014-03-09: The ``writebin`` member function of the ``sim`` class is now
implicitly called when calling the ``run`` member function.

2014-03-25: Fluid phase in ``master`` branch simulated by the full Navier-Stokes
equations.

Documentation
-------------
See the separate documentation for general reference and installation
instructions. The documentation is by default available in
the `html <doc/html/index.html>`_ and `pdf <doc/pdf/sphere.pdf>`_ formats.

Author
------
Anders Damsgaard, `anders.damsgaard@geo.au.dk <mailto:anders.damsgaard@geo.au.dk>`_,
`blog <http://anders-dc.github.io>`_,
`more contact information <https://cs.au.dk/~adc>`_.
