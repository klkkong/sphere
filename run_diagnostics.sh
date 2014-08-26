#!/bin/bash

OUTFILE=diagnostics.txt

function report {
    echo "\n### $@ ###" >> $OUTFILE
    $@ >> $OUTFILE 2>&1
}

report ls -lhR .
report git show
report git status
report lspci
report uname -a
report cmake --version
report nvcc --version
report cmake .
report make
report make test
report cat Testing/Temporary/LastTestsFailed.log
report cat Testing/Temporary/LastTest.log

echo "### Diagnostics complete ###"
echo "Report bugs and unusual behavior to anders.damsgaard@geo.au.dk"
