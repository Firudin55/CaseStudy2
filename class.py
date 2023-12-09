# THREE PARTS

# first part
import streamlit as st
import pandas as pd
from PIL import Image

# User input via Streamlit sidebar
st.sidebar.write("# User input parameters")

sepal_length = st.sidebar.slider('Sepal length', 4.0, 10.0, 5.4)
sepal_width = st.sidebar.slider('Sepal width', 2.0, 6.0, 3.4)
petal_length = st.sidebar.slider('Petal length', 1.0, 8.0, 1.3)
petal_width = st.sidebar.number_input('Petal width', 0.1, 3.0, 0.2)


# Prepare data for prediction
data = {'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width}

df = pd.DataFrame(data, index = [0])

# Display user input parameters
st.write("## User Input parameters")
st.write("These are the parameters you have chosen. Make sure there are no errors")
st.write(df.T.rename(columns={0: "Length in cm"}))


### second part

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load and prepare the Iris dataset

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
#
# st.write(target_names)

# Model creation and training
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Make predictions based on user input
prediction = clf.predict(df)
probability = np.max(clf.predict_proba(df)) * 100

#st.write(clf.predict_proba(df))

# Display prediction results
st.write("## Prediction part")
st.write("### Here is your prediction based on your inputs")
st.write(f"""The flower you are trying 
             to predict appears to be 
             **{target_names[prediction][0]}**
             with a probability of {probability}. %
          """)



# Map the predicted class to the corresponding image file
image_files = {
    'setosa': 'irisSetosa.jfif',
    'versicolor': 'irisVersicolor.jfif',
    'virginica': 'irisVirginica.jfif'
}

# Display the corresponding image
image_file = image_files.get(target_names[prediction][0].lower())

if image_file:
    image = Image.open(image_file)
    st.image(image, f"The sample image of Iris {target_names[prediction][0]} type", width= 200)
else:
    st.write("No image available for this class.")

