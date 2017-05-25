# QABot
Wipro-Stack Route final project 
A question answer bot on Stanford Question Answer Dataset


--Question_Para.py--
Contains the main code for the processing.
Given a query, it points to a paragraph and returns the paragrah as answer.

query_to_paragraph() ---> The main function accepts a question and returns the paragraph as answer.


--Starterbot.py--
Contains the code to integrate the Question_Para.py to run on Slack.

--botid.py--
Returns a bot ID for the bot name.


To run on terminal:
1. Open Question_Para.py
2. Uncomment the last 3 lines.
3. Run on terminal.
Note : 1.The program is written for slack integration. Therefore when run in terminal it gives out the answer and exits the program.
         To give another query, the program has to be run again. There is a delay everytime when you run the program as the models 
         used takes time to get loaded. To avoid this, put the last three commented lines in a recurscive loop before running in 
         terminal.
       2. All the requried/support files for running are not present on GIT.
       
To run on slack:
Requirements: slack bot token - This a unique token that is generated when the bot was created on Slack.
1. Export slack bot token in the terminal.
2. Run botid.py. It returns a bot ID for the bot name.
3. Export the bot ID.
4. Run Starterbot.py. You should get a message on the terminal 'StarterBot connected and running!'
Note: 1. All the requried/support files for running are not present on GIT.
