#!/bin/bash

root="./tests/.data"
mird_dir="${root}/MIRD"
filename="Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_0.160s__3-3-3-8-3-3-3.zip"

wget -q -P "${root}/" "https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/${filename}"

if [ ! -d "${mird_dir}/" ]; then
    mkdir -p "${mird_dir}/"
fi

unzip -q -d "${mird_dir}/" "${root}/${filename}"