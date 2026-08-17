import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot
import traceback
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import wordcloud

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment



# Load the model and vectorizer from the model registry and local storage
# Inside your load function
def load_model_and_vectorizer(model_name, stage, vectorizer_path):
    mlflow.set_tracking_uri("http://ec2-3-109-47-234.ap-south-1.compute.amazonaws.com:5000/") 
    
    # Use the alias/stage instead of a hardcoded number
    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.sklearn.load_model(model_uri)
    
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

# Initialize the model by calling the stage
model, vectorizer = load_model_and_vectorizer("my_model", "Staging", "./tfidf_vectorizer.pkl")



#def load_model(model_path, vectorizer_path):
 #   """Load the trained model."""
  #  try:
   #     with open(model_path, 'rb') as file:
     #       model = pickle.load(file)
    #    
      #  with open(vectorizer_path, 'rb') as file:
       #     vectorizer = pickle.load(file)
      #
       # return model, vectorizer
    #except Exception as e:
     #   raise


# Initialize the model and vectorizer
#model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")  

# Initialize the model and vectorizer

@app.route('/')
def home():
    return "Welcome to our flask api"



@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    try:
        # 1. Force JSON parsing to prevent NoneType crashes
        data = request.get_json(force=True)
        comments_data = data.get('comments')
        
        if not comments_data:
            return jsonify({"error": "No comments provided"}), 400

        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # 2. Preprocess
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # 3. Transform to SPARSE matrix (Do NOT use .toarray() here!)
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # 4. Predict directly on the sparse matrix (Zero memory spikes)
        predictions = model.predict(transformed_comments).tolist()
        
        # 5. Map LightGBM's '2' back to '-1' for your charts to work
        # Original map during training: {-1: 2, 0: 0, 1: 1}
        label_mapping = {0: "0", 1: "1", 2: "-1"}
        mapped_predictions = [label_mapping.get(pred, str(pred)) for pred in predictions]
        
        # 6. Format Response
        response = [
            {"comment": c, "sentiment": s, "timestamp": t} 
            for c, s, t in zip(comments, mapped_predictions, timestamps)
        ]
        return jsonify(response)
        
    except Exception as e:
        # 7. Actually trigger the traceback to print the crash report in the terminal
        print("\n=== CRASH REPORT ===")
        traceback.print_exc()
        print("====================\n")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True) 
        
        if not data or 'comments' not in data:
            return jsonify({"error": "No comments provided in the JSON body"}), 400
            
        comments = data.get('comments')
        
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # USE SPARSE MATRIX directly. No .toarray(), no pd.DataFrame()
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        predictions = model.predict(transformed_comments).tolist() 
        
        # Map labels for frontend compatibility
        label_mapping = {0: "0", 1: "1", 2: "-1"}
        mapped_predictions = [label_mapping.get(pred, str(pred)) for pred in predictions]
        
        response = [{"comment": c, "sentiment": s} for c, s in zip(comments, mapped_predictions)]
        return jsonify(response)
        
    except Exception as e:
        print("\n=== CRASH REPORT ===")
        traceback.print_exc()
        print("====================\n")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # 1. Convert to DataFrame and set timestamps
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        # Catch any lingering '2's from the old frontend state and force them to '-1'
        df['sentiment'] = df['sentiment'].replace({2: -1})
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # 🚨 THE FIX: Dynamic Time Binning based on the data's actual timespan
        time_span = df.index.max() - df.index.min()
        
        if time_span.days >= 60:
            freq = 'ME'         # Month End
            date_fmt = '%Y-%m'
            title_prefix = 'Monthly'
        elif time_span.days >= 7:
            freq = 'D'          # Daily
            date_fmt = '%Y-%m-%d'
            title_prefix = 'Daily'
        elif time_span.total_seconds() >= 7200: # Greater than 2 hours
            freq = 'h'          # Hourly
            date_fmt = '%m-%d %H:%M'
            title_prefix = 'Hourly'
        else:
            freq = '5min'       # 5-minute intervals for brand new videos
            date_fmt = '%H:%M'
            title_prefix = 'Minute-by-Minute'

        # 2. Resample dynamically
        trend_counts = df.resample(freq)['sentiment'].value_counts().unstack(fill_value=0)

        # Hack to fix the floating dot bug if the API somehow returns exact same-second timestamps
        if len(trend_counts) == 1:
            # Duplicate the row slightly offset so Matplotlib is forced to draw a horizontal line
            duplicate_row = trend_counts.copy()
            duplicate_row.index = duplicate_row.index + pd.Timedelta(minutes=1)
            trend_counts = pd.concat([trend_counts, duplicate_row])

        # 3. Calculate percentages
        trend_totals = trend_counts.sum(axis=1)
        trend_percentages = trend_counts.div(trend_totals, axis=0).fillna(0) * 100

        # Ensure all columns exist
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in trend_percentages.columns:
                trend_percentages[sentiment_value] = 0

        trend_percentages = trend_percentages[[-1, 0, 1]]

        # 4. Plotting
        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                trend_percentages.index,
                trend_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                linewidth=2,
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title(f'{title_prefix} Sentiment Percentage Over Time')
        plt.xlabel('Time')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Apply the dynamic date formatter
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

        plt.legend()
        plt.tight_layout()

        # Save to BytesIO and return
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
        
    except Exception as e:
        print("\n=== GRAPH CRASH REPORT ===")
        traceback.print_exc()
        print("==========================\n")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)