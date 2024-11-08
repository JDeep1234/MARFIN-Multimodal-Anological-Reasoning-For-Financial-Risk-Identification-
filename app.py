import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sentence_transformers import SentenceTransformer

# Load the model
model = pickle.load(open("C:\\Users\\jnyanadeep\\Downloads\\marfin_model (1).pkl", "rb"))

# Title and description
st.title("Financial Risk Assessment System")
st.write("""
This app uses a multimodal knowledge graph and retrieval-augmented generation (RAG) 
to assess financial risk and provide insights on bankruptcy risk.
""")

# Upload data
st.sidebar.header("Upload Your CSV Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded, display it
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())

    # Data Preprocessing
    df.columns = [c.replace(' ', '_') for c in df.columns]
    X = df.drop(columns=['Bankrupt?'])
    y = df['Bankrupt?']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Oversample for class balance
    over_sampler = RandomOverSampler(random_state=42)
    X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
    
    # Display Class Balance
    st.subheader("Class Balance")
    fig, ax = plt.subplots()
    df['Bankrupt?'].value_counts(normalize=True).plot(kind='bar', ax=ax)
    plt.xlabel("Bankrupt classes")
    plt.ylabel("Frequency")
    st.pyplot(fig)
    
    # Display feature distribution for selected column
    st.subheader("Feature Distribution by Class")
    fig, ax = plt.subplots()
    sns.boxenplot(x="Bankrupt?", y="_Net_Income_to_Total_Assets", data=df, ax=ax)
    plt.title("Net Income to Total Assets Ratio by Class")
    st.pyplot(fig)

    # Train and Evaluate Model
    if st.sidebar.button("Train Model"):
        # Training the RandomForest with GridSearchCV
        params = {
            "n_estimators": range(25, 100, 25),
            "max_depth": range(10, 70, 10)
        }
        clf = RandomForestClassifier(random_state=42)
        model = GridSearchCV(clf, param_grid=params, cv=5, n_jobs=-1)
        model.fit(X_train_over, y_train_over)
        
        # Save model
        with open("marfin_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Evaluation
        st.subheader("Model Evaluation")
        train_acc = model.score(X_train_over, y_train_over)
        test_acc = model.score(X_test, y_test)
        st.write(f"Training Accuracy: {train_acc:.4f}")
        st.write(f"Test Accuracy: {test_acc:.4f}")
        st.text("Classification Report")
        st.text(classification_report(y_test, model.predict(X_test)))

    # Knowledge Graph
    st.subheader("Knowledge Graph")
    def build_knowledge_graph(data):
        G = nx.Graph()
        for column in data.columns:
            G.add_node(column, modality='numerical')
        for i in range(len(data.columns)):
            for j in range(i + 1, len(data.columns)):
                G.add_edge(data.columns[i], data.columns[j])
        return G

    knowledge_graph = build_knowledge_graph(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(knowledge_graph)
    nx.draw(knowledge_graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    st.pyplot(fig)

    # RAG - Retrieval-Augmented Generation
    st.subheader("Advanced Insights with RAG")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Or use another pre-trained model
    high_risk_descriptions = [
        "This company has a high debt to asset ratio, indicating potential financial instability.",
        "The net income to total assets ratio is low, raising concerns about profitability.",
        "Company shows increasing liabilities with declining assets."
    ]
    
    if st.button("Generate Insights"):
        embeddings = model.encode(high_risk_descriptions)
        st.write("RAG-based insights generated for high-risk predictions:")
        for i, desc in enumerate(high_risk_descriptions):
            st.write(f"Insight {i+1}: {desc}")
