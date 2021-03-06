import streamlit as st
import plotly.express as px
from data import get_standup_responses
import model


bert = model.BertModel()


if __name__ == '__main__':
    st.header('Visualize Standupman Responses using BentoML')
    st.write(
        'The chart below contains 3-dimensional representation of responses from Standupman.'
    )

    # get the responses from Standupman
    responses = get_standup_responses()
    # load bert from BentoML and plot the embeddings
    st.plotly_chart(
        bert.visualize(responses)
    )
