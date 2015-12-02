Installation
============

You have multiple options to get up and running, though using ``pip`` is by far the easiest and most reliable.


Downloading Package
-------------------

**Recommended.** To download and setup the last officially released package, you can do so from PYPI directly::

    > pip install scikit-neuralnetwork

This contains its own packaged version of ``pylearn2`` from the date of the release (and tag) but will use any globally installed version if available.

If you want to install the very latest from source, please visit the `Project Page <http://github.com/aigamedev/scikit-neuralnetwork>`_ on GitHub for details.


Pulling Repositories
--------------------

**Optional.** To setup a developer version of the project, you'll need to first install some dependencies manually.  Unfortunately, ``pylearn2`` isn't yet installable via PyPI and recommends an editable (``pip -e``) installation::

    > pip install numpy scipy theano
    > pip install -e git+https://github.com/lisa-lab/pylearn2.git#egg=Package

Once that's done, you can grab this repository and install from ``setup.py`` in the exact same way::

    > git clone https://github.com/aigamedev/scikit-neuralnetwork.git
    > cd scikit-neuralnetwork
    > python setup.py develop

This will make the ``sknn`` package globally available within Python as a reference to the current directory.


Running Tests
-------------

We encourage you to launch the tests to check everything is working using the following commands::

    > pip install nose
    > nosetests -v sknn

Use the additional command-line parameters in the test runner ``--processes=8`` and ``--process-timeout=60`` to speed things up on powerful machines.  The result should look as follows in your terminal.

.. image:: console_tests.png

We strive to maintain 100% test coverage for all code-paths, to ensure that rapid changes in the underlying ``pylearn2`` library are caught automatically.
