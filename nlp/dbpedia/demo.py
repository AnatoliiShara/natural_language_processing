from models import Author, Book, FactSource

import re
import spacy
from spacy import displacy

import warnings
warnings.filterwarnings('ignore')

nlp = spacy.load('en_core_web_lg')
def clean_text(text):
    text = re.sub(r'=+.*?=+', '', text)
    text = re.sub(r'[\s\t]+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    return text


def visulize_wiki_ents(author):
    wiki = author.wikipage.get()
    text = clean_text(wiki.text)
    doc = nlp(text)
    displacy.render(doc, style='ent', options={'ents': ['WORK_OF_ART', 'DATE']}, jupyter=True)


def print_books_by_source(author, source):
    print(f'========== {source.upper()} ===============')
    dbpedia = FactSource.get(FactSource.stype == source)
    for b in author.books.select().where(Book.fact_source == dbpedia):
        print(b)
    print('==============================\n')


def print_dbpedia_books(author):
    print_books_by_source(author, 'dbpedia')


def print_ner_books(author):
    print_books_by_source(author, 'spacy_clean_lg')



for author in Author.select()[:10]:
    print(f'-== {author.name.upper()} ==-\n')
    print_dbpedia_books(author)
    print_ner_books(author)
    visulize_wiki_ents(author)
    print('\n\n\n')


from models import Author, Book, FactSource

author_count = Author.select().count()
book_count = Book.select().count()
fact_source_count = FactSource.select().count()

print(f"Author count: {author_count}")
print(f"Book count: {book_count}")
print(f"FactSource count: {fact_source_count}")

from models import Author, Book, FactSource

authors = Author.select().limit(10)  # Retrieve the first 10 authors
books = Book.select().limit(10)      # Retrieve the first 10 books
fact_sources = FactSource.select()    # Retrieve all FactSource records

for author in authors:
    print(f"Author: {author.name}, Birthday: {author.birthday}")

for book in books:
    print(f"Book: {book.title}, Year: {book.year}")

for source in fact_sources:
    print(f"FactSource: {source.stype}")




from models import Author, Book, FactSource

authors = Author.select().limit(10)  # Retrieve the first 10 authors
books = Book.select().limit(10)      # Retrieve the first 10 books
fact_sources = FactSource.select()    # Retrieve all FactSource records

for author in authors:
    print(f"Author: {author.name}, Birthday: {author.birthday}")

for book in books:
    print(f"Book: {book.title}, Year: {book.year}")

for source in fact_sources:
    print(f"FactSource: {source.stype}")










