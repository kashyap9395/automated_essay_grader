# automated_essay_grader
A project for my class CS 421: Natural Language Processing that grades a bunch of essays using NLP techniques.

Project 1 complete for CS421 – University of Illinois at Chicago  
Name 1: sparim2@uic.edu   Name 2: def34@uic.edu
---------------------------------------------------------Setup------------------------------------------------------------
Alternatively, you can give the following command (in either Windows or Linux):  
Please install required libraries using the following commands:  
Pip install numpy  
Pip install statsmodels  
Pip install Stanfordcorenlp  
Pip install nltk  
Pip install pandas  
Pip install genism
Pip install progressbar
Pip install textblob
After installing nltk please make sure the stopwords and names corpus is accessible
Can be downloaded by nltk.download(‘stopwords’) and nltk.download(‘names’) respectively.  
-----------------------------------------------Running the grader----------------------------------------  
To run the grader:  
• Navigate to the executables folder.
• Type python Main_comp.py “your path to Stanford core nlp folder”.
• If there is an error like xe\02 non ascii character found. It is due to the compiler not recognizing the file as utf-8 encoded. To fix this we included
 -*- coding: utf-8 -*- in the first line of every file.
• The code outputs some warnings on the screen, they do no affect the code run and are printed by pandas.
• Development is done on windows environment and sometimes there might be errors like variable used before assigning. This is not a error in the code, this is due to tab space issue when the files are read by other OS.
• Please run on windows if any error occurs.
• The results are in output folder in the file results_with_grade.txt.
• Uploading a screen cast of the run with the archive at https://drive.google.com/drive/folders/17l6in8gNF8p9MuoYBz_4nt3vVp6FHsX4?usp=sharing showing no errors and a successful run.
• Complete run takes approximately 7 minutes on intel i7 with 16 GB ram.
