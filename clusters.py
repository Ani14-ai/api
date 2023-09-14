import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify,send_file
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from io import BytesIO
from sklearn.pipeline import Pipeline
import numpy as np

app = Flask(__name__)

@app.route('/cluster', methods=['POST'])
def cluster_data():
    try:
        # Read the uploaded CSV file
        file = request.files['file']
        if file:
            csv_data = file.read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_data))
            print(df.head())
            # Data preprocessing (cleaning and scaling)
            #cleaned_data = preprocess_data(df)

            # Perform clustering on selected columns
            columns = request.form['columns'].split(',')
            selected_columns = df[columns]
            print(selected_columns)
            x = selected_columns.iloc[:, 0].values
            y = selected_columns.iloc[:, 1].values
            data = list(zip(x, y))

            # Perform K-Means clustering with the user-selected number of clusters
            num_clusters = int(request.form['num_clusters'])
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(data)

            # Get cluster labels
            cluster_labels = kmeans.labels_.tolist()

            return jsonify({'cluster_labels': cluster_labels})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/plot', methods=['POST'])
def plot_clusters():
    try:
        # Read the uploaded CSV file
        file = request.files['file']
        if file:
            csv_data = file.read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_data))
            columns = request.form['columns'].split(',')
            selected_columns = df[columns]
            x = selected_columns.iloc[:, 0].values
            y = selected_columns.iloc[:, 1].values
            data = list(zip(x, y))

            # Create a single figure with multiple subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

            # Elbow method plot
            inertias = []
            for i in range(1, 11):
                kmeans = KMeans(n_clusters=i)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            ax1.plot(range(1, 11), inertias, marker='o')
            ax1.set_title('Elbow method')
            ax1.set_xlabel('Number of clusters')
            ax1.set_ylabel('Inertia')
            silhouette_scores = []
            for i in range(2, 11):  # Silhouette Score requires at least 2 clusters
                kmeans = KMeans(n_clusters=i)
                kmeans.fit(data)
                labels = kmeans.labels_
                silhouette_avg = silhouette_score(data, labels)
                silhouette_scores.append(silhouette_avg)
            ax3.plot(range(2, 11), silhouette_scores, marker='o')
            ax3.set_title('Silhouette Score')
            ax3.set_xlabel('Number of clusters')
            ax3.set_ylabel('Silhouette Score')

            # K-Means Clustering plot
            num_clusters = int(request.form['num_clusters'])
            kmeans = KMeans(n_clusters=num_clusters)
            kmeans.fit(list(zip(x, y)))
            ax2.scatter(x, y, c=kmeans.labels_)
            ax2.set_title('K-Means Clustering')
            ax2.set_xlabel('Column 1')
            ax2.set_ylabel('Column 2')

            # Save the figure in a single buffer
            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)

            # Return the image as a response
            return send_file(img_buffer, mimetype='image/png')
    except Exception as e:
        # Handle exceptions here
        return str(e)

    except Exception as e:
        return jsonify({'error': str(e)})

def preprocess_data(df):
    # Separate numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Define data transformations for numeric and categorical columns
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])

    # Apply transformations to columns based on data type
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Fit and transform the data
    preprocessed_data = preprocessor.fit_transform(df)

    # Convert the preprocessed data back to a DataFrame
    df_preprocessed = pd.DataFrame(preprocessed_data)

    return df_preprocessed

if __name__== '__main__':
    app.run(debug=False)