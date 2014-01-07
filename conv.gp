#!/usr/bin/env gnuplot

set terminal dumb

set title "Convergence evolution in CFD solver"
set xlabel "Iteration"
set ylabel "Jacobi iterations"

plot 'conv.dat' with linespoints notitle
