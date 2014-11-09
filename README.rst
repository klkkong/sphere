=============
sphere readme
=============
``sphere`` is a 3D discrete element method algorithm utilizing CUDA. ``sphere``
allows for optional simulation of two-way coupled fluid flow using the
Navier-Stokes or Darcy formulations.

A powerful Nvidia GPU with proper support for double precision floating is
highly recommended. ``sphere`` has been tested with the Nvidia Tesla K20 and
Nvidia Tesla M2050 GPUs.

License
-------
``sphere`` is licensed under the GNU General Public License, v.3.
See `LICENSE.txt <LICENSE.txt>`_ for more information.

Important release notes
-----------------------
2014-11-05: A Darcy solver has been added as an alternative to the Navier-Stokes
solver of the fluid phase. It can be selected with e.g. ``initFluid(cfd_solver =
1)`` in the Python module.

2014-07-28: Fluid flow is no longer simulated in a separate program. Use
``sphere`` with the command line argument ``-f`` or ``--fluid`` instead.

2014-07-05: Fluid phase now discretized on a staggered grid which increases
accuracy and stability.

2014-03-25: Fluid phase in ``master`` branch simulated by the full Navier-Stokes
equations.

2014-03-09: The ``writebin`` member function of the ``sim`` class is now
implicitly called when calling the ``run`` member function.

2014-03-09: The main sphere class (formerly ``spherebin``) has been renamed to
``sim``.

2014-01-20: Version fingerprints have been added to the input/output binary
files, and causes old files to be incompatible with either ``sphere`` or
``sphere.py``.

2014-01-25: The description of the installation procedure is moved to the
general documentation.

2013-03-13: Sphere has been updated to work with CUDA 5.0 or newer *only*.

Documentation
-------------
See the separate documentation for general reference and installation
instructions. The documentation is by default available in
the `html <doc/html/index.html>`_ and `pdf <doc/pdf/sphere.pdf>`_ formats.

Examples
--------
All examples are visualized using `ParaView <http://www.paraview.org>`_.

.. figure:: doc/sphinx/img/stokes.png
   :scale: 100%
   :alt: Particle falling through fluid grid

   A particle falling through a fluid column causing fluid flow.

.. figure:: doc/sphinx/img/diff.png
   :scale: 100%
   :alt: Consolidation test

   Consolidation test of particle/fluid assemblage.

.. figure:: doc/sphinx/img/shear.png
   :scale: 100%
   :alt: Shear test

   Shear of a dense particle assemblage. Top left: particles colored by initial
   positions, top middle: particles colored by horizontal velocity, top right:
   particles colored by pressure. Bottom left: fluid pressures, bottom center:
   porosities, bottom right: porosity changes.

Author
------
Anders Damsgaard, `anders.damsgaard@geo.au.dk <mailto:anders.damsgaard@geo.au.dk>`_,
`blog <http://anders-dc.github.io>`_,
`more contact information <https://cs.au.dk/~adc>`_.
