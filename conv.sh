#!/bin/bash
if [[ "$1" == "" ]]; then
    echo "Usage: $0 <simulation id>"
    exit 1
else 
    echo watch -n 10 gnuplot conv.gp $1
fi
