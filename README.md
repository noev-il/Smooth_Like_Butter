# Smooth_Like_Butter
Spotify Project - Allow Users to 
1. Choose a song
2. Find songs that transition into that one
3. Create a chain of transitions
4. Choose what genre they'd like to focus on
5. Create a playlist with songs that transition into each other

Project Tools = 
Code Set: Python, SQL (Data Storage), JSON (Porting)
Databases: Seaborn (Data Viz), Matplot (Data Viz), SciPy (Algorithm/Comparisons), Librosa (Spectogram), Pandas (Data Analysis), JSON (Manaing Data).


+Journal Entry 08/16/2023
-Downloaded Spyder
-Downloaded Miniconda
-Configured Librosa
-Updated Spyder preferences to use Miniconda PY3.8 with Librosa package installed
-Working with @QED0711 Audio_Analyzer Project on GitHub

+Journal Entry 08/17/2023
-Colorstack Summit
-Created Librosa Spectrogram Graphs with Time as X-Axis
-Loaded files (Bruno Mars Silk Sonic Intro, Kali Uchis After the Storm)
-Installed Seaborn and Pandas through Anaconda Command Prompt
-Created a Copy and Paste Sheet for Code
-Syntax Study of Audio_Analyzer Project on GitHub

+Journal Entry 08/21/2023
-Review Project Roadmap
-Visualize Roadmap (FIGMA)
-Import Roadmap to Github
-Update Calendar with To do, In Progress, Finished Tags on CHecklist

+Joural Entry 11/21/2023
-Imported Anaconda on Home Desktop
-Booted Spyder
-Imported Project Files onto Local Drives
-Cleared Disk Space

+Journal Entry 11/21/2023
-Created Working Python Code to Send Request to Spotify API for Song Analysis
-Used JavaScript in VSCODE and Python in Anaconda/Spyder

+Journal Entry 11/22/203 
-Updated Code to Take User Input of Song URL, Ask for File Name, and Return a JSON File of Song Analysis
-Created a JSON File parser to count the amount of segments in JSON file (in python)

+Journal Entry 12/10/2023
-Project has hit major headway
-Create a individualzed way to grab columbs within the 'section' tabe and create tensors
-PYTORCH, SKLEARN, PANDAS, NUMPY, all were used
Files have been pushed to github.

+Journal Entry 12/13/2023
-Project has moved into the testing phase
-Operating Fuctions - Take two spotify urls, calls for spotify audio analysis, creates tensors of meaningful data points, compares euclidean distance of two songs tensors, returns a 3 dimensional rating of the segment, pitches, and timbres euclidean distance.

-All within a python class called tensor_creator()

The next stage is to move to a linear regression machine learning model using scikit learn. I will need to label good and bad transitions. I have a currently library with 30+ songs that transition well into each other, 15 pairs in total. I need to prepare a test data set to test my models performance. 

+Journal Entry 12/21/2023
-Scikit Learn Model: Random Forest Classifer
-Early tests on the model show progress, accuracy score of 1 on a data set of 6 total songs and 3 pairs. Accuracy score represents ability to decide if it is a "smooth" transition or a "bad" one. The model had 100% accuracy on the small test.
-Created a document with 100 different pairs of songs from the Top 100 Rap Songs. Model will be tested on the desktop PC to take advantage of CUDA cores on my NVIDIA GTX 970 MSI graphics card.
-Next steps are focused on refining my model. 
