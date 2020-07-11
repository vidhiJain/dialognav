## Setting up the repo, and gym_minigrid submodule
1. git clone https://gitlab.com/cmu_asist/visdial.git
2. git switch dialog_v2n
2. git submodule init
3. git submodule update
4. cd gym_minigrid
5. git switch visdial
6. cd ../

## Installing spacy for dialogs
1.. pip install spacy
2. spacy download en_core_web_sm

## Adding malmoutils.py and MalmoPython.so to Pythonpath
1. Locate the directory malmoutils.py and MalmoPython.so in your malmo installtion. (Usually at, <malmo_directory>/Python_Examples/)
2. Add this line: `export PYTHONPATH="${PYTHONPATH}:<path_to_the_above_two_files>"`



