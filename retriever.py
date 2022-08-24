import re
import numpy as np
import pandas

import tabula
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader

import streamlit as st

class TableReader:

  def __init__(self, pdf):
    self.pdf = pdf
    self.dfs = None

  def read_pages(self, pages="all", multiple_tables=True, stream=True):
    '''
    Return tables discovered within pdf.
    '''
    self.dfs = tabula.read_pdf(self.pdf, pages=pages, multiple_tables=multiple_tables, stream=stream)
    self.__clean_dfs()
    return self.dfs

  def __clean_dfs(self, thresh=2):
    self.dfs = [df.dropna(thresh=thresh) for df in self.dfs]


class Reader:

  def __init__(self, filename):
    self.reader = PdfReader(filename)
    self.tb = TableReader(filename)
    self.page_viewer = {page_num : {} for page_num in range(self.reader.numPages)}
    self.idx2page_item = []
  
  def __extract_text(self,):
    '''
    Page-wise text extraction and tokenize for BM25.
    '''
    # List to store each tokenized corpus
    tokenized_corpus_list = []
    for i in range(self.reader.numPages):
      raw_text = self.reader.getPage(i).extractText()
      self.page_viewer[i]["raw_text"] = raw_text
      # Split text
      corpus = raw_text.split("\n \n")
      # Store results.
      self.page_viewer[i]["corpus"] = corpus
      for item in corpus:
        self.idx2page_item.append((i, item)) # page,textItem
      # Tokenize
      tokenized_corpus = [doc.split(" ") for doc in corpus]
      tokenized_corpus_list.append(tokenized_corpus)
    # BM25 computations only after the complete tokenized corpus is collated. 
    # Merge tokenized corpus'.
    tokenized_corpus_complete = [item for sublist in tokenized_corpus_list for item in sublist]
    # BM25
    self.bm25 = BM25Okapi(tokenized_corpus_complete)

  def __extract_tables(self):
    '''
    Page-wise table extractor.
    '''
    for i in range(self.reader.numPages):
      # page=0 will throw error using tabula.
      page = str(i+1)
      self.page_viewer[i]["tables"] = self.tb.read_pages(pages=page)

  def extract_pdf(self):
    # Extract data
    self.__extract_text()
    self.__extract_tables()

  def print_page(self, page_num):
    '''
    Print separated sections of text given a page.
    '''
    corpus = self.page_viewer[page_num]["corpus"]
    for item in (corpus):
      st.write("\n{}\n".format("-"*60))
      st.write(item)
    st.write("\n{}\n".format("-"*60))
    for df in self.page_viewer[page_num]["tables"]:
      st.dataframe(df)

def score(reader, queries, weights):
  '''
  Compute the average BM25 score of each given query on each page of text.
  '''
  ranked_scores = []
  average_scores = []
  for query in queries:
    # tokenize query by whitespace.
    tokenized_query = query.split()
    # Compute score.
    doc_scores = reader.bm25.get_scores(tokenized_query)
    ranked_scores.append(doc_scores)
  # Compute average (weighted) score against all queries.
  if not len(weights):
    # Equal weighting.
    average_scores = np.average(ranked_scores, axis=0)
  elif len(queries) != len(weights):
      # Unequal number of elements.
      raise ValueError("Number of query and weight elements passed must be equal.")
  else:
    # Weighted average.
    average_scores = np.average(ranked_scores, weights=weights, axis=0)
  
  return average_scores

def get_ranked_texts(reader, queries, weights=[], n=5):
  '''
  Return n pages which scored highest using BM25.
  '''
  # Run score method to calculate BM25.
  average_scores = score(reader, queries, weights)
  idx = sorted(range(len(average_scores)), key=lambda i: average_scores[i], reverse=True)[:n]

  final_results = []
  for i in range(n):
    page_num, text = reader.idx2page_item[idx[i]]
    tables = reader.page_viewer[page_num]["tables"]
    final_results.append({"page_num":page_num, "text":text, "tables":tables})

  return final_results
    
# filename = "Globalworth-Green-Bond-Report-2020-20-July-2021.pdf"
# reader = Reader(filename)

# reader.extract_pdf()

# queries = [
#     "use of proceeds",
#     "allocation of proceeds",
#     # "projects financed"
#     ]

# top_n = 5
# top_items = reader.get_ranked_texts(queries, n=top_n)

# for item in top_items:
#   page_num = item["page_num"]
#   text = item["text"]
#   tables = item["tables"]
#   print("Page: {}\n\n{}".format(page_num,text))
#   for table in tables:
#     display(table.style)
#   print("-"*60)
#   print("\n\n")

# texts = [Text(item["text"], {"docid" : i}, 0) for i, item in enumerate(top_items)]
