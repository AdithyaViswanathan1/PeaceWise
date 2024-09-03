# PeaceWise
An ML chatbot on mental health information.

## Datasets

My [primary dataset](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data) was a mental health conversational Q/A dataset from Kaggle. This dataset contains 80 sets of Q/A, although its actual size was larger because each row actually contained multiple questions and answers. The “Reformatting Data” section talks about how I extended the effective size of this dataset. The features are “patterns” (or questions) and “responses”. The label is “tag”, also known as the type of question (ex. greeting, help, stress, etc.). 

My [secondary dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset) was another mental health Q/A dataset, which I got from HuggingFace. It contained 172 sets of Q/A. My processing in order to bring this to the same structure as primary dataset is described in the “Reformatting Data” sub-section below. Even after adding the second dataset, which brought my total Q/A rows to 250, I felt my data was not big enough for a machine learning model to handle a conversation. Therefore, I performed the following.
