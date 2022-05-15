@echo off

set root=tests\.data
set mird_dir=%root%\MIRD
set filename=Impulse_response_Acoustic_Lab_Bar-Ilan_University__Reverberation_0.160s__3-3-3-8-3-3-3.zip

mkdir %mird_dir%
bitsadmin /transfer "Download" https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/%filename% %CD%\%mird_dir%\%filename%
call powershell -command "Expand-Archive %mird_dir%\%filename% %mird_dir%"
