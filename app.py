import streamlit as st
import spacy
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def sumy_summarizer(docx):
    parser = PlaintextParser.from_string(docx, Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document, 3)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

def text_analyzer(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    allData = [('"Tokens":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in docx]
    return allData

def entity(my_text):
    nlp = spacy.load('en_core_web_sm')
    docx = nlp(my_text)
    tokens = [token.text for token in docx]
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    allData = ['"Tokens:":{}\n"Entities":{}'.format(tokens, entities)]
    return allData

def main():
    st.title("Named Entity Recognition App")
    st.subheader("Try out several natural language processing features!")


    st.write("First enter your text, then click the button of whichever feature you want to try out!")
    message = st.text_area("Enter your text")
    if st.button("Tokenize your text"):
        nlp_result = text_analyzer(message)
        st.json(nlp_result)
    
    if st.button("Extract Entities from your text."):
        nlp_result = entity(message)
        st.json(nlp_result)
    
    #Sentiment score feautre
    if st.button("Sentiment score of your text"):
        #Using TextBlob to evaluate sentiment score of user input
        blob = TextBlob(message)
        result_sentiment = blob.sentiment
        st.success(result_sentiment)

    #Summarization feature
    if st.button("Summarize your text"):
        #Using sumy library
        summary_result = sumy_summarizer(message)
        st.success(summary_result)

if __name__ == "__main__":
    main()