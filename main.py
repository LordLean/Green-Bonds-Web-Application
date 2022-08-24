import streamlit as st
import os

from transformers import pipeline

from annotated_text import annotated_text

from retriever import Reader, get_ranked_texts


st.set_page_config(
    layout="wide",
    page_title="Green Bond Analyzer",
    page_icon="🌎",
    initial_sidebar_state="expanded",
    )

###########################################
# Load QA model.
###########################################
cwd = os.getcwd()
model_dir = cwd + "\/finbert-pretrain-finetuned-squad"
question_answering = pipeline("question-answering", model=model_dir, tokenizer=model_dir)
#############################################################
# question = "how are proceeds allocated?"

###########################################
# Sidebar
###########################################
st.sidebar.title("Application Settings Menu")
uploaded_file = st.sidebar.file_uploader("Upload your PDF file", type=["pdf"])

###########################################
# Title
###########################################
title = "Green Bond Analyzer"
st.markdown("<h1 style='text-align: center; color: white;'>{}</h1>".format(title), unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: center; color: #0E1117;'>{}</h3>".format(title), unsafe_allow_html=True)
st.text("")
st.text("")
###########################################
# Title - end
###########################################

###########################################
# Rerankers
###########################################
@st.cache
def get_rerankers():
    return MonoT5(), MonoBERT()

###########################################
# Upload widget
###########################################
@st.cache
def load_pdf(file):
    reader = Reader(file)
    with st.spinner("Extracting textual and tabular data..."):
        reader.extract_pdf()
    return reader

if uploaded_file is not None:
    reader = load_pdf(uploaded_file)
    st.success('PDF data extracted')
else:
    st.info("Please upload a PDF file to get started.")



###########################################
# Answer Retrieval/Reranking
###########################################
with st.form(key="Answer Retrieval Settings"):
    st.markdown("### Answer Retrieval and Reranking")
    col1, col2 = st.columns((8,2))
    queries = col1.text_area("Enter multiline answer retrieval queries")
    weights = col2.text_area("Enter query weights")

    col3, col4 = st.columns((8,2))
    n_items = col3.number_input(label="Return top N items", min_value=1)
    rerank = col4.radio("Answer Reranker", ["None", "T5", "BERT",])

    st.markdown("### Question Answering")
    question = st.text_input("Enter query for question-answer system")

    submitted = st.form_submit_button("Submit")

# If form submit button
if submitted and uploaded_file and queries:
    # Split multiline inputs by line.
    split_queries = queries.splitlines()
    split_weights = weights.splitlines()

    # Create list of all query keywords.
    keyword_list = [query.split() for query in split_queries]
    # Remove duplicate query keywords.
    keyword_set = set([item.lower() for sublist in keyword_list for item in sublist])
    # annotated_text(*[(item+"",) for item in keyword_set])

    # Exception handing.
    if split_weights:
        try:
            split_weights = [float(weight) for weight in split_weights]
        except ValueError:
            st.error("Please all ensure weight values passed are floats.")
        if sum(split_weights) > 1.0 or sum(split_weights) < 1.0:
            st.error("Please ensure weights sum to 1.0")
        if len(split_queries) != len(split_weights):
            st.error("Number of query and weight elements passed must be equal.\nQueries: {}\nWeights: {}".format(len(split_queries),len(split_weights)))
    else:
        split_weights = []
        
    # Display keywords
    st.markdown("### Retrieval Query Keywords:")
    for item in keyword_set:
        annotated_text(
            (item,)
        )

    # Get bm25 top results.
    top_items = get_ranked_texts(reader, queries=split_queries, weights=split_weights, n=n_items)
    
    # Display top items.
    for i, item in enumerate(top_items):
        # Title
        st.markdown("## Retrieved Text {}".format(i+1))
        # Create display columns
        text_col, result_col = st.columns((6,2))

        # Results column containing query answer/result + QA model score.
        with result_col:
            # If question has been asked.
            if question:
                try:
                    # QA on text:
                    results = question_answering(question=question, context=item["text"], device=0, top_k=3)
                    result = results[0] # Get highest scoring result.
                    # Answer
                    st.markdown("### Query Answer")
                    annotated_text((result['answer'], "", "#008000"))
                    # st.write(result['answer'])
                    # Score
                    st.markdown("### Score:")
                    st.markdown("#### **{}**".format(round(result['score'],5)))
                    # st.write(results)
                except IndexError:
                    st.error("No Result")
            else:
                pass
        # Text column containing Page no. and text.
        with text_col:
            # Get answer text:
            if question:
                # For annotations sake.
                # Get ans string.
                ans = item["text"][result["start"]:result["end"]]
                # Split text before and after answer.
                before_ans = item["text"][:result["start"]].split()
                after_ans = item["text"][result["end"]:].split()
                st.markdown("### Page: **{}**".format(item["page_num"]))
                # Create list of str and tuples for input to text_annotation
                ans_annotated = [(ans+ " ", "", "#008000")]
                before_ans_annotated = [token+ " " if token.lower() not in keyword_set else (token+" ",) for token in before_ans]
                after_ans_annotated = [token+ " " if token.lower() not in keyword_set else (token+" ",) for token in after_ans]
                full_text_annotated = before_ans_annotated + ans_annotated + after_ans_annotated
                # Unpack list.
                annotated_text(*full_text_annotated)
            else:
                st.markdown("### Page: **{}**".format(item["page_num"]))
                # Split passage by whitespace
                full_text_split = item["text"].split()
                # Create list of str and tuples for input to text_annotation
                full_text_annotated = [token+ " " if token.lower() not in keyword_set else (token+" ",) for token in full_text_split]
                # Unpack list.
                annotated_text(*full_text_annotated)

        for table in item["tables"]:
            st.dataframe(table)
        st.write("\n\n")

# Exception handing.
elif submitted and uploaded_file is None:
    st.error("Please upload a pdf.")
elif submitted and uploaded_file and not queries:
    st.error("Please enter appropriate inputs to the fields above.")

