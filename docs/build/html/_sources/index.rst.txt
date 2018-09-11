LR Splines
==========

|Build Status| |Coverage Status| |Documentation Status|

An LR-Spline implementation written in Python.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree:: 
  :maxdepth: 2
  
  self 
  basic_usage.rst
  api_reference.rst

This aim of Python library is to provide a lightweight framework for
understanding LR-splines. The library is in no shape or form optimized
for high performance computing, but is rather aimed at being a small
toolkit for gaining some intuition for LR-splines. For more industrial
grade performance and a more complete set of tools, see the `GoTools
library <https://github.com/SINTEF-Geometry/GoTools>`__ written in C++.

Other LR-spline-related projects:

1. `LRSplines <https://github.com/VikingScientist/LRsplines>`__: A C++
   library which some of the code in this repository is based on.
2. `LRSplines: Android
   App <https://github.com/VikingScientist/LR-introduction>`__: An app
   for interactive demonstration of the LR-spline refinement procedure.

Introduction
^^^^^^^^^^^^

The need for adaptive refinement techniques is evident when it comes to
optimizing the tradeoff between computational cost and computational
accuracy. When utilizing spline spaces with an underlying tensor-product
structure, refinement of a mesh induces a global propagation of the
newly introduced meshlines to the whole mesh. This can be very
inefficient. The concept of LR-Splines was introduced in 2013 in the
paper `Polynomial splines over locally refined
box-partitions <https://www.sciencedirect.com/science/article/pii/S0167839613000113>`__,
and can be seen as an attempt to remedy this aforementioned problem.
LR-Splines have several desirable properties:

1. They form a non-negative partition of unity by construction.
2. Linear independence (under some conditions on the refinement).

Construction
^^^^^^^^^^^^

LR-splines are construced by starting with an initial **tensor product**
spline space. Meshlines are then inserted one at the time, making sure
the line **completely traverses** the support of at least one B-spline.

This B-spline is then **split** according to the standard knot insertion
procedure, producing two new B-splines. These new B-splines are
subsequently tested against **all previously existing meshlines**, to
check for further splitting.

Installation
^^^^^^^^^^^^

Download the repository and run:

.. code:: bash

        python setup.py install

Verify the installation by running:

.. code:: bash

        python -m import LRSplines

.. |Build Status| image:: https://travis-ci.org/qTipTip/LRSplines.svg?branch=master
   :target: https://travis-ci.org/qTipTip/LRSplines
.. |Coverage Status| image:: https://coveralls.io/repos/github/qTipTip/LRSplines/badge.svg?branch=master
   :target: https://coveralls.io/github/qTipTip/LRSplines?branch=master
.. |Documentation Status| image:: https://readthedocs.org/projects/lrsplines/badge/?version=latest
   :target: https://lrsplines.readthedocs.io/en/latest/?badge=latest
