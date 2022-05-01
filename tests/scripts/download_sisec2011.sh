#!/bin/bash

root="./tests/.data"
sisec2011_dir="${root}/SiSEC2011"
filename="dev1.zip"

wget -q -P "${root}/" "http://www.irisa.fr/metiss/SiSEC10/underdetermined/${filename}"

if [ ! -d "${sisec2011_dir}/" ]; then
    mkdir -p "${sisec2011_dir}/"
fi

unzip -q -d "${sisec2011_dir}/" "${root}/${filename}"
