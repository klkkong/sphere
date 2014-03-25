#!/bin/sh
rm ../output/*.vti; rm ../output/*.vtu; \
    cd ~/code/sphere-cfd && cmake .  && make -j2 &&\
    cd ~/code/sphere-cfd/python && ipython -i ns_cons_of_mass.py
