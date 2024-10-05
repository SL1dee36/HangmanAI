# Hangman with a Neural Network

This project implements a Hangman game with two modes:

1. **AI Guesses:** The neural network attempts to guess a randomly selected word from a provided word list. The AI is trained to predict the probability of each letter in the word. The number of attempts the AI gets is determined by the number of unique letters in the word plus 3 to allow for a few incorrect guesses.  The game logs the AI's guesses and whether they are correct.

2. **User Guesses:** The user attempts to guess a word randomly selected by the AI from the same word list. The game tracks the number of attempts made. The number of attempts allowed is the number of unique letters in the word plus 3.

Both game modes save the results (including all steps and outcomes) to a file named `results.md`.

**Features:**

*   Two game modes: AI guessing and user guessing.
*   A neural network trained to predict the probability of a letter being in the word.
*   Detailed logging of each game to `results.md`.
*   Number of attempts dynamically adjusted based on the number of unique letters in the word.

**Technologies Used:**

*   Python
*   Scikit-learn (for the neural network)
*   NumPy


**How to Use:**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sl1dee36/HangmanAI.git
    ```
2.  **Install dependencies:**  Make sure you have Python 3 installed. Then, install the required libraries:
    ```bash
    pip install numpy scikit-learn
    ```
3.  **Prepare the word list:** Create a file named `words.txt` in the same directory. Each line of the file should contain a single word in uppercase letters.  The code includes a default word list if `words.txt` is not found.
4.  **Run the game:** Execute the Python script:
    ```bash
    python hangman.py
    ```
5.  The game will prompt you to select a game mode.  Follow the on-screen instructions.



**Further Development:**

*   Improved neural network architecture for better accuracy.
*   Implementation of additional game features.


This project demonstrates a fun application of machine learning, combining a classic game with a neural network for a unique twist.
