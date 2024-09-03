# PeaceWise
An ML chatbot on mental health information.

## Datasets

My [primary dataset](https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data) was a mental health conversational Q/A dataset from Kaggle. This dataset contains 80 sets of Q/A, although its actual size was larger because each row actually contained multiple questions and answers. The “Reformatting Data” section talks about how I extended the effective size of this dataset. The features are “patterns” (or questions) and “responses”. The label is “tag”, also known as the type of question (ex. greeting, help, stress, etc.). 

My [secondary dataset](https://huggingface.co/datasets/heliosbrahma/mental_health_chatbot_dataset) was another mental health Q/A dataset, which I got from HuggingFace. It contained 172 sets of Q/A. My processing in order to bring this to the same structure as primary dataset is described in the “Reformatting Data” sub-section below. Even after adding the second dataset, which brought my total Q/A rows to 250, I felt my data was not big enough for a machine learning model to handle a conversation. Therefore, I performed the following.

## Objectives

The main objective of this project is:
> Develop a chatbot that has coherent conversations on mental health information.

To achieve the main objective, here are sub-objectives:
- Pre-processing data to fit prediction structure
- Data analysis to visualize underlying patterns
- Applying NLP and ML techniques to predict correct answer to question
- Pipeline for user to communicate to chatbot
- Evaluation of chatbot's strengths and weaknesses

## Pre-processing

### Data Augmentation

Even after adding my secondary dataset, I felt the size of my dataset to be inadequate. Furthermore, I noticed the questions and responses in my secondary dataset contained only one sentence. This would make generalization tough for my model, as it would only accept the question valid if it appears exactly as it does in my dataset during training. Therefore, I decided to paraphrase all the questions in my secondary dataset and some in my primary dataset. This allowed the model to account for multiple variations in the user’s query for the same question. `For example, if the training data had the question “What is mental health?”, I wanted it to also train on the questions “What is the definition of mental health?”, “Define mental health”, and “What constitutes mental health”.` This is representative of user queries since each user might word their query differently, and it is my job as a model builder to account for variation in their language. I achieved this augmentation by utilizing a pretrained t5 model on HuggingFace, which takes a question as input, and outputs N paraphrased questions. Here is the link. One thing which I would try to improve in future work is the running time of this paraphrasing process. It scales linearly with the number of questions in the dataset, so running time will linearly grow with size of dataset.

### Reformatting data
Now it is time to tackle my primary dataset. I noticed that each row contained multiple questions and answers, so it would be necessary to extract questions into their own rows. This is done so the model only trains on one question and label at a time. The number of responses can still be numerous. I did this by creating a new row for each additional question that appears in a given row in my primary dataset. This resulted in approximately double the rows from the original primary dataset. For my secondary dataset, I had to extract the questions and answers, which were stored in the same column. I used regular expressions to separate the sentences into question and answer. Then, I applied the same transformation I made on my primary dataset; extract questions into their own rows so that each row contained only one question. Doing this to the primary and secondary dataset resulted in massive growth of the dataset size; I increased my dataset from 250 instances to nearly 1200.

### Data Analysis
The purpose of data analysis was to learn more about my data before diving into model building. First, I plotted the distribution of my tag column (class label). Specifically, I plotted the top 15 tags. My highest-occurring label was “greeting”. Due to my augmentation, certain Q/A about mental health made it into my distribution as well. Next, I plotted the distribution of the responses column. This was done to see which type of question had the most answers. It ended up being tags like “about”, “casual”, “stressed”, and “anxious”. The fact that mental health tags had a lot of responses was a sign of good training, as it would lead to varied answers to these common mental health questions from the user. Finally, I created a word cloud of the training data (questions) as it visually displays the most common words with relative size.

### Final Pre-processing
Finally, in order to prepare my data for inputting into a model, split the data into X (training) and Y (label). X is the “pattern” column and Y is the “tag” column. Then, for each row, I tokenized, lowercased, and lemmatized each word, and removed stopwords. This lead my total training instances to be round 1200. As previously mentioned, I kept training and testing data constant so that future iterations and models would receive the same data for the purpose of equal comparison and analysis afterward.




