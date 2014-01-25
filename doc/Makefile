doc:
	$(MAKE) -C sphinx/

install-debian-pkgs:
	sudo apt-get update && \
	    sudo apt-get install build-essential cmake \
	    python python-numpy ipython python-matplotlib python-vtk \
	    imagemagick libav-tools \
	    python-sphinx python-pip doxygen dvipng \
	    python-sphinxcontrib-programoutput texlive-full && \
	    sudo pip install breathe
