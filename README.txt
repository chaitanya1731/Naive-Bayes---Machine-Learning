Name	: Chaitanya Kulkarni
B-Number: B00814455
Email	: ckulkar2@binghamton.edu
------------------------------------------------------------------------------------
Academic Honesty Statement:
I have done this assignment completely on my own. I have not copied it, nor have I given my solution to anyone else.
I understand that if I am involved in plagiarism or cheating I will have to sign an official form that
I have cheated and that this form will be stored in my official university record.
I also understand that I will receive a grade of 0 for the involved assignment for my first offense and that
I will receive a grade of “F” for the course for any additional offense.
-Chaitanya Kulkarni
------------------------------------------------------------------------------------
Q - Report accuracy for Naive Bayes for this filtered set.
Does the accuracy improve? Explain? why the accuracy improves or why it does not?

A - Accuracy of filtered test data = 94.351464
- Accuracy may increase or decrease depending of the quality of stopwords data.
- In our case, accuracy is slightly decreased because the filtered dataset is
almost similar to test data except a few words eliminated.
- If there were more stopwords, the accuracy would have been more different. Hence, accuracy
depends n the quality of the stopwords.
------------------------------------------------------------------------------------
How to Compile and run the program
- program is written in python 3.8
- program is tested on remote.cs.binghamton.edu and is running successfully as expected

Command prompt execution-
    >> python3 naive.py <training-set> <test-set> <stopwords>
    >> python3 naive.py train test stopwords.txt

--------------------------------------------------------------------------------------
Output
- Outputs the accuracy of training data, test data and test data without stopwords
- Accuracy is printed in console as per the required format
- Accuracy is also printed in Accuracy.txt text document in the same format
- Please check the attached ChaitanyaKulkarni_hw2_theory.pdf for theory questions
