from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

import glob
import os
import urllib.request

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


def update_params():
    st.experimental_set_query_params(challenge=st.session_state.day)


md_files = sorted(
    [int(x.strip("Day").strip(".md")) for x in glob.glob1("content", "*.md")]
)


st.set_option('deprecation.showPyplotGlobalUse', False)



def main():
    st.title("Cogxta - Insight Driven Messaging")
    
    
# Logo and Navigation
col1, col2, col3 = st.columns((1, 4, 1))
#with col2:
#st.sidebar.image(Image.open("./Capture.png"))
#img_path='Capture.png'
#st.image(str(img_path))

avatar_path = "https://icon-library.com/images/default-profile-icon/default-profile-icon-16.jpg"
st.caption(
                        f'<a href="https://github.com/{c.github_author}"><img src="{avatar_path}" style="border: 1px solid #D6D6D9; width: 20px; height: 20px; border-radius: 50%"></a> &nbsp; <a href="https://github.com/{c.github_author}" style="color: inherit; text-decoration: inherit">{c.github_author}</a>',
                        unsafe_allow_html=True,
                    )

days_list = [f"Day {x}" for x in md_files]

query_params = st.experimental_get_query_params()

try:
    if query_params and query_params["challenge"][0] in days_list:
        st.session_state.day = query_params["challenge"][0]
except KeyError:
    st.session_state.day = days_list[0]

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown(
    """
- [Book](https://cogxta.com/) 
- [Blog](https://blog.cogxta.com/)
"""
)

# Display content
for i in days_list:
    if selected_day == i:
        st.markdown(f"# 🗓️ {i}")
        j = i.replace(" ", "")
        with open(f"content/{j}.md", "r") as f:
            st.markdown(f.read())
        if os.path.isfile(f"content/figures/{j}.csv") == True:
            st.markdown("---")
            st.markdown("### Figures")
            df = pd.read_csv(f"content/figures/{j}.csv", engine="python")
            for i in range(len(df)):
                st.image(f"content/images/{df.img[i]}")
                st.info(f"{df.figure[i]}: {df.caption[i]}") 
    
@st.cache(persist= True)
def load():
    data= pd.read_csv(r"./mushrooms.csv")
    label= LabelEncoder()
    for i in data.columns:
        data[i] = label.fit_transform(data[i])
    return data
df = load()

if st.sidebar.checkbox("Display data", False):
    st.subheader("Show Email dataset")
    data_1= pd.read_csv(r"./spam.csv",encoding='latin-1')
    st.dataframe(data_1)
    
    
@st.cache(persist=True)
def split(df):
    y = df.type
    x = df.drop(columns=["type"])
    x_train, x_test, y_train, y_test =     train_test_split(x,y,test_size=0.3, random_state=0)
    
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split(df)


def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=   class_names)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
class_names = ["edible", "poisnous"]

st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
    gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")
metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"),key='one')

if st.sidebar.button("Classify", key="classify_svm"):
    st.subheader("Support Vector Machine (SVM) results")
    model = SVC(C=0.1, kernel='rbf', gamma='scale')
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    st.write("Accuracy: ", accuracy.round(2))
    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
    plot_metrics(metrics)
    
if classifier == "Logistic Regression":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify_log"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)
        
if classifier == "Random Forest":
    st.sidebar.subheader("Hyperparameters")
    n_estimators= st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step =1, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key="bootstrap")
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify_rf"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap= bootstrap, n_jobs=-1 )
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)
        

if __name__ == '__main__':
    main()