"""
1. Formatting.
The Associated Press Stylebook is a style guide that is often used by journalists around the world. 
It recommends the following rules for formatting headlines:

- Capitalize nouns, pronouns, verbs, adjectives, adverbs, and subordinating conjunctions. 
If a word is hyphenated, a capital letter should be added for each part of the word (e.g., "Self-Reflection" is correct, not "Self-reflection"). Capitalize the first and last words of the title, regardless of the part of speech.
Capitalize all other parts of speech: articles/determiners, conjunctions, prepositions, particles, interjections.

Task:
    write a program that formats headings according to the specified rules
    run your program on the corpus of headings from The Examiner
    save the program and the file with the formatted headings in a directory with your name
    how many headers in the corpus were formatted correctly (how many headers remained unchanged?)

Please note that your program must correctly distinguish between prepositions and subordinating conjunctions. 
For example, Do as you want => Do As You Want (because "as" is a conjunction), 
but How to use a Macbook as a table => How to Use a Macbook as a Table (because "as" is a preposition).
"""

import re
import pandas as pd 

def capitalize_word(word: str) -> str:
	# capitalize the word according to the specified rules
	if "-" in word:
		# handle hyphenated words
		parts = word.split("-")
		capitalized_word = [part.capitalize() for part in parts]
		return "-".join(capitalized_word)
	else:
		return word.capitalize()

def format_heading(heading: str) -> str:
	# split the heading into words
	words = heading.split()
	# format each word
	formatted_words = []
	for i, word in enumerate(words, start=0):
		if i == 0 or i == len(words) or word not in ["a", "an", "the", "and", "but", "or",
													"nor", "for", "so", "yet", "as", "at", 
													"by", "in", "of", "on", "to", "with"]:
			formatted_word = capitalize_word(word)
		else:
			formatted_word = word
		formatted_words.append(formatted_word)
	return " ".join(formatted_words)

# read CSV file with headlines
data = pd.read_csv("examiner-date-text.csv")

# Filter out rows with missing values in the 'headline_text' column
data = data[pd.notna(data['headline_text'])]

# extract headlines from CSV file
corpus = data['headline_text'].tolist()

# format the headlines
formatted_corpus = [format_heading(heading) for heading in corpus]

# count how many headers were formatted correctly
correctly_formatted_count = sum(1 for original, formatted in zip(corpus, formatted_corpus)
								if original == formatted)

# save formatted headlines to a file
with open("shara_anatolii_formatted_headlines.txt", "w", encoding="utf-8") as f:
	for formatted in formatted_corpus:
		f.write(f"{formatted}\n")


print("Formatted headlines saved.")
print(f"Number of headers formatted correctly: {correctly_formatted_count}")


















