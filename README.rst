.. image:: https://badges.gitter.im/Join%20Chat.svg
   :alt: Join the chat at https://gitter.im/anders-dc/sphere
   :target: https://gitter.im/anders-dc/sphere?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

=============
sphere readme
=============
``sphere`` is a 3D discrete element method algorithm utilizing CUDA. ``sphere``
allows for optional simulation of two-way coupled fluid flow using the
Navier-Stokes or Darcy formulations.

A powerful Nvidia GPU with proper support for double precision floating is
highly recommended. ``sphere`` has been tested with the Nvidia Tesla K20 and
Nvidia Tesla M2050 GPUs.

    **Note:** CUDA 6.5 is the recommended version. CUDA 7.0 may work but is not
    thoroughly tested yet.

License
-------
``sphere`` is licensed under the GNU General Public License, v.3.
See `LICENSE.txt <LICENSE.txt>`_ for more information.

Important release notes
-----------------------
2015-09-06: A new flux boundary condition has been added to the Darcy fluid
solver.

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
   :scale: 75%
   :alt: Particle falling through fluid grid

   A particle moving downwards through a fluid column causing fluid flow.

.. figure:: doc/sphinx/img/diff.png
   :scale: 100%
   :alt: Consolidation test

   Consolidation test of a particle/fluid assemblage.

.. figure:: doc/sphinx/img/shear.png
   :scale: 100%
   :alt: Shear test

   Shear of a dense particle assemblage. Top left: particles colored by initial
   positions, top center: particles colored by horizontal velocity, top right:
   particles colored by pressure. Bottom left: fluid pressures, bottom center:
   porosities, bottom right: porosity changes.

Publications
------------
``sphere`` has been used to produce results in the following scientific
publications and presentations:

- Damsgaard, A., D.L. Egholm, J.A. Piotrowski, S. Tulaczyk, N.K. Larsen, and
  C.F. Brædstrup (2015), A new methodology to simulate subglacial deformation of
  water saturated granular material, The Cryosphere Discuss., 9(4), 3617-3660,
  `doi:10.5194/tcd-9-3617-2015 <http://dx.doi.org/10.5194/tcd-9-3617-2015>`_.
- Damsgaard, A., D.L. Egholm, J.A. Piotrowski, S. Tulaczyk, N.K. Larsen, and
  C.F. Brædstrup (2014), Numerical modeling of particle-fluid mixtures in a
  subglacial setting. `Poster at Americal Geophysical Union Fall Meeting
  <https://cs.au.dk/~adc/files/AGU2014-Poster.pdf>`_.
- Damsgaard, A., D.L. Egholm, J.A. Piotrowski, S. Tulaczyk, N.K. Larsen, and
  K. Tylmann (2013), Discrete element modeling of subglacial sediment
  deformation, J. Geophys. Res. Earth Surf., 118, 2230–2242,
  `doi:10.1002/2013JF002830 <http://dx.doi.org/10.1002/2013JF002830>`_.
- Damsgaard, A., D.L. Egholm, J.A. Piotrowski, S. Tulaczyk, and N.K. Larsen
  (2013), Discrete element modeling of subglacial sediment deformation.
  Talk at American Geophysical Fall Meeting 2013.
- Damsgaard, A., D.L. Egholm, J.A. Piotrowski, S. Tulaczyk, and N.K. Larsen
  (2013), Numerical modelling of granular subglacial deformation using the
  discrete element method. `Poster at European Geosciences Union General
  Assembly 2013
  <https://cs.au.dk/~adc/files/EGU2013-Poster.pdf>`_.
- Damsgaard, A., D.L. Egholm, J.A. Piotrowski, and S. Tulaczyk
  (2012), Discrete element modelling of subglacial sediment deformation.
  `Poster at European Geosciences Union General Assembly 2012
  <https://cs.au.dk/~adc/files/EGU2012-Poster.pdf>`_.
- Damsgaard, A., D.L. Egholm, and J.A. Piotrowski
  (2011), Numerical modelling of sediment deformation by glacial stress.
  `Poster at International Union for Quaternary Research Congress 2011
  <https://cs.au.dk/~adc/files/INQUA2011-Poster.pdf>`_.
- Damsgaard, A., D.L. Egholm, and J.A. Piotrowski
  (2011), Numerical modelling of subglacial sediment deformation.
  `Poster at European Geosciences Union General Assembly 2011
  <https://cs.au.dk/~adc/files/EGU2011-Poster.pdf>`_.

Author
------
Anders Damsgaard, `anders.damsgaard@geo.au.dk <mailto:anders.damsgaard@geo.au.dk>`_,
`blog <http://anders-dc.github.io>`_,
`more contact information <https://cs.au.dk/~adc>`_.

