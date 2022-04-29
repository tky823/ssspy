@echo off

set root=tests\.data
set sisec2011_dir=%root%\SiSEC2011
set filename=dev1.zip

mkdir %sisec2011_dir%
bitsadmin /transfer "Download" http://www.irisa.fr/metiss/SiSEC10/underdetermined/%filename% %CD%\%sisec2011_dir%\%filename%
call powershell -command "Expand-Archive %sisec2011_dir%\%filename% %sisec2011_dir%"
