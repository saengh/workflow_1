@echo off
echo Activating python...
call "C:\Users\SinghDis\3D Objects\my-files\ml-driven-landscapes\venv\Scripts\activate"

cd modules
echo Initializing...
python m1_main.py

echo Parsing XML...
python m2_xml_parser.py

echo Mapping CPC definitions...
python m3_assign_cpc_defs.py

echo Preprocessing text fields...
python m4_preprocessor.py

echo Generating tokens...
python m5_lda_preprocessor.py

echo Performing topic modeling...
python m6_lda_topic_modeler.py

echo Execution finished.
pause