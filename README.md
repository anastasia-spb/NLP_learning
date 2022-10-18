## Text Classification

https://www.kaggle.com/competitions/nlp-txt-classification

The goal is to classify tweets. There are 5 categories: Extremely Negative, Negative, Neutral, Positive, Extremely Positive.
This falls into the "Classifying whole sentences" category of common NLP tasks.

#### Examples:

* Extremely Positive:

"Due to the Covid-19 situation, we have increased demand for all food products. 
The wait time may be longer for all online orders, particularly beef share and freezer packs. 
We thank you for your patience during this time."

* Extremely Negative:

with 100 nations inficted with covid 19 the world must not play fair with china 100 goverments must demand china adopts new guilde lines on food safty the chinese goverment is guilty of being irosponcible with life  on a global scale

* Neutral:

The COVID-19 coronavirus pandemic is impacting consumer shopping behavior, purchase decisions and retail sales, according to a First Insight study

There are two approach:

1. Use RNN model, such as LSTM - LSTM_from_glove.ipynb
2. Use transformers - Transformers.ipynb

With LSTM max accuracy score is 0.487.
Transformers gave much better result - 0.861.

