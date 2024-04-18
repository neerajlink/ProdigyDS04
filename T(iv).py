import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Sample data (you can replace this with your actual dataset)
data = {
    'ID': [2401, 2401, 2401, 2402, 2402, 2403, 2403, 2404, 2404],
    'Topic': ['Borderlands', 'Borderlands', 'Borderlands', 'Borderlands', 'Borderlands', 'Borderlands', 'Borderlands', 'Borderlands', 'Borderlands'],
    'Sentiment': ['Positive', 'Positive', 'Neutral', 'Positive', 'Neutral', 'Neutral', 'Neutral', 'Positive', 'Positive'],
    'Text': ['im getting on borderlands and i will murder you all',
             'I am coming to the borders and I will kill you all',
             'Rock-Hard La Varlope, RARE & POWERFUL...',
             'So I spent a few hours making something for fun...',
             '2010 So I spent a few hours making something for fun...',
             'that was the first borderlands session in a long time...',
             'this was the first Borderlands session in a long time...',
             'im getting on borderlands 2 and i will murder you me all',
             'im getting into borderlands and i can murder you all']
}

# Create DataFrame from sample data
df = pd.DataFrame(data)

# Initialize SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Perform sentiment analysis and add scores to the DataFrame
df['Sentiment_Score'] = df['Text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Plot sentiment score against ID
plt.figure(figsize=(10, 6))
plt.scatter(df['ID'], df['Sentiment_Score'], color='blue')
plt.xlabel('ID')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis by ID')
plt.grid(True)
plt.show()

# Plot sentiment score against Topic
plt.figure(figsize=(10, 6))
plt.scatter(df['Topic'], df['Sentiment_Score'], color='green')
plt.xlabel('Topic')
plt.ylabel('Sentiment Score')
plt.title('Sentiment Analysis by Topic')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

