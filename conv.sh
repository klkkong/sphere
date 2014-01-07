#!/bin/bash
if [[ "$1" == "" ]]; then
    echo "Usage: $0 <simulation id>"
    exit 1
else 
    watch -n 5 gnuplot -e \"sid=\'$1\'\" conv.gp
fi
