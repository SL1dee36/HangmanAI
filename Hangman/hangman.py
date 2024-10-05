import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import time
import pickle
import re
import os

# Define the alphabet - crucial for consistency
ALPHABET = "QWERTYUIOPASDFGHJKLZXCVBNM"
game_words = ["WORLD", "AMSTERDAMN", "HAMSTER"] 

def load_words_from_file(filename="words.txt"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            words_str = file.read()
            # Improved regex to handle various whitespace and punctuation
            words_str = re.sub(r"[^a-zA-Z\s]", "", words_str).upper()
            words_str = re.sub(r"\s+", " ", words_str).strip()
            words = words_str.split()
            if not words:
                raise ValueError("File is empty or contains no words.")
            return words
    except FileNotFoundError:
        print(f"File '{filename}' not found. Using default word list.")
        return ["python", "programming", "artificial", "intelligence", "machine", "learning", "algorithm", "dataset", "neural", "network"]
    except Exception as e:  # More general exception handling
        print(f"Error reading file: {e}")
        return []


def generate_data(num_samples, word_list):
    data = []
    for _ in range(num_samples):
        word = random.choice(word_list).upper()
        letters = set(word) # Unique letters in the word
        for letter in letters:
            letter_index = ALPHABET.index(letter)
            guess_data = np.zeros(26, dtype=int)
            guess_data[letter_index] = 1
            data.append((guess_data, 1)) # 1 - means the letter is in the word
    return data


def preprocess_data(data):
    X, y = zip(*data)
    return np.array(X), np.array(y)


def train_model(X_train, y_train, X_test, y_test, hidden_layer_sizes=(50, 25), max_iter=50, learning_rate_init=0.001, early_stopping=True, validation_fraction=0.5, n_iter_no_change=10, verbose=True, random_state=42):
    """Trains the MLPClassifier model with improved parameters and validation."""

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes, 
        max_iter=max_iter, 
        learning_rate_init=learning_rate_init,
        learning_rate='adaptive',  # Adaptive learning rate
        early_stopping=early_stopping,  # Early stopping
        validation_fraction=validation_fraction,  # Validation data fraction
        n_iter_no_change=n_iter_no_change,  # Number of epochs without improvement for early stopping
        verbose=verbose, 
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Evaluation on test data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test data: {accuracy:.2f}")

    return model


def display_word(word, correct_positions):
    display = ""
    for i, (letter, correct) in enumerate(zip(word, correct_positions)):
        display += letter if correct else "_ "
    return display.strip()


def play_game(model, word, game_number=1, results_file="results.md"):
    guessed_letters = set()
    num_unique_letters = len(set(word.upper()))
    num_attempts = num_unique_letters + 3  # Maximum number of attempts
    attempts = num_attempts
    word_length = len(word)
    correct_positions = [False] * word_length
    word_letters = set(word.upper())
    game_log = f"Game number: {game_number} | Word to guess: {word}\n"

    # Clear console
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"Game number: {game_number} | Word to guess: {word}")

    step = 0
    while step < 10 and not all(correct_positions):
        try:
            available_letters = word_letters - guessed_letters
            if not available_letters:
                break  # All letters guessed

            while True:
                predicted_char = random.choice(list(available_letters))
                guess_data = np.zeros(26, dtype=int)
                guess_data[ALPHABET.index(predicted_char)] = 1
                guess_data = guess_data.reshape(1, -1)

                prediction = model.predict(guess_data)[0]
                is_correct = prediction == 1

                game_log += f"Step [{step}]: Neural network response: {predicted_char} | {'Correct' if is_correct else 'Incorrect'}\n"
                
                print(f"\nHidden word: {display_word(word, correct_positions)}")
                print(f"Neural network response: {predicted_char} | {'Correct' if is_correct else 'Incorrect'}")
                print(f"Remaining attempts: {num_attempts-step}")

                if is_correct:
                    break
                else:
                    attempts -= attempts
                    break

            guessed_letters.add(predicted_char)
            if predicted_char in word:
                indices = [i for i, char in enumerate(word) if char == predicted_char]
                for index in indices:
                    correct_positions[index] = True
            step += 1

        except IndexError:
            print("Error. Try again.")
            time.sleep(0.5)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    #Record result
    result = "Win" if all(correct_positions) else "Loss"
    game_log += f"Game result: {result} (Attempts used: {step}/{num_attempts})\n\n"

    with open(results_file, "a", encoding="utf-8") as f:
        f.write(game_log)

    if all(correct_positions):
        print(f"\nWIN! AI guessed the word: {word} in {step}/{num_attempts}.")
    else:
        print(f"\nLOSS! AI did not guess the word: {word}")

    time.sleep(10)
    os.system('cls' if os.name == 'nt' else 'clear')

def play_game_ai_guesses(words, game_number=1, results_file="results.md"):
    word = random.choice(words)
    guessed_letters = set()
    num_unique_letters = len(set(word.upper()))
    num_attempts = num_unique_letters + 3
    attempts = num_attempts
    word_length = len(word)
    correct_positions = [False] * word_length
    word_letters = set(word.upper())
    game_log = f"Game number: {game_number} | Word to guess (AI): {word}\n"

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Game number: {game_number} | Word to guess (AI): {'_ ' * word_length}")

    while attempts > 0 and not all(correct_positions):
        guess = input(f"Guess a letter ({attempts} attempts): ").upper()
        if not guess.isalpha() or len(guess) != 1:
            print("Invalid input. Enter a single letter.")
            continue

        if guess in guessed_letters:
            print("You already guessed that letter. Try another one.")
            continue

        guessed_letters.add(guess)
        game_log += f"Step [{num_attempts - attempts}]: User response: {guess} | {'Correct' if guess in word else 'Incorrect'}\n"
        print(f"User response: {guess} | {'Correct' if guess in word else 'Incorrect'}")

        if guess in word:
            indices = [i for i, char in enumerate(word) if char == guess]
            for index in indices:
                correct_positions[index] = True
        else:
            attempts -= 1

        print(f"Hidden word: {display_word(word, correct_positions)}")

        if all(correct_positions):
            break

    result = "Win" if all(correct_positions) else "Loss"
    game_log += f"Game result: {result} (Attempts used: {num_attempts - attempts}/{num_attempts})\n\n"
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(game_log)
    print(f"Game result: {result}")
    time.sleep(10)
    os.system('cls' if os.name == 'nt' else 'clear')


def save_model(model, filename="hangman_model.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(model, file)

def load_model(filename="hangman_model.pkl"):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading model: {e}. Retraining...")
        return None


def get_word_from_user():
    """Gets a word from the user."""
    while True:
        word = input("Enter a word for the game (only English alphabet letters): ").upper()
        if word.isalpha():
            return word
        else:
            print("Invalid input. Use only English alphabet letters.")


if __name__ == "__main__":
    words = load_words_from_file()
    if not words:
        print("Word list is empty. Exiting.")
        exit()
    num_samples = 10000  

    data = generate_data(num_samples, words)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        model = load_model()
        if model is None:
            print("Training model...")
            model = train_model(X_train, y_train, X_test, y_test)
            save_model(model)
            print("Model trained successfully.")
        else:
            print("Model loaded.")

        game_number = 1
        if not os.path.exists("results.md"):
            with open("results.md", "w", encoding="utf-8") as f:
                f.write("# Hangman Game Results\n")

        while True:
            choice = input("Choose game mode:\n1. Random word\n2. Enter word\n3. AI guesses word\n4. Exit\nYour choice: ")
            if choice == '1':
                word = random.choice(words)
                play_game(model, word, game_number=game_number)
                game_number += 1
            elif choice == '2':
                word = get_word_from_user()
                play_game(model, word, game_number=game_number)
                game_number += 1
            elif choice == '3':
                play_game_ai_guesses(words, game_number=game_number)
                game_number += 1
            elif choice == '4':
                break
            else:
                os.system('cls' if os.name == 'nt' else 'clear')
                print("[!] Invalid choice.")

    except Exception as e:
        print(f"An error occurred: {e}")