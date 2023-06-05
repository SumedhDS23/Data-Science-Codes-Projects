#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the Necessary Dependencies
import pandas as pd
import seaborn as sns
import string
import nltk
from nltk.tokenize import TweetTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
import itertools
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import classification_report


# In[2]:


#Read and Load the Dataset
df = pd.read_csv("D:/EXCELR/Project/tweet.csv")


# In[3]:


#Exploratory Data Analysis
df.head()


# In[4]:


df.columns


# In[5]:


print('length of data is', len(df))


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.dtypes


# In[9]:


#Checking null spaces
df.isna().sum()


# In[10]:


# Rows and columns in the dataset
print('Count of columns in the data is:  ', len(df.columns))
print('Count of rows in the data is:  ', len(df))


# In[11]:


#Check unique target values
df['class'].unique()


# In[15]:


# Count the number of tweets in each class
class_counts = df['class'].value_counts()
print(class_counts)
#Plotting a bar chart
class_counts.plot(kind='bar', rot=0)
plt.xlabel('Class')
plt.ylabel('Number of Tweets')
plt.title('Class Distribution')
plt.show()


# In[16]:


#Plotting a pie chart
class_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()


# In[13]:


sns.countplot(x='class', data=df)


# In[13]:


#Data Preprocessing
data=df[['tweets','class']]


# In[14]:


# Separating figurative,irony, regular and sarcasm classified data
df_fig=data[data['class']=='figurative']
df_irony=data[data['class']=='irony']
df_regular=data[data['class']=='regular']
df_sarcasm=data[data['class']=='sarcasm']

# Create a list of labels corresponding to the combined_list
Y = ['figurative'] * len(df_fig) + ['irony'] * len(df_irony) + ['Regular_list'] * len(df_regular) + ['sarcasm'] * len(df_sarcasm) 


# In[15]:


#Combining tweets
dataset = pd.concat([df_fig, df_irony, df_regular, df_sarcasm,])


# In[16]:


#Making statement text in lowercase
dataset['tweets']=dataset['tweets'].str.lower()
dataset['tweets'].tail()


# In[17]:



# Define the stopwords
stopwordlist = stopwords.words('english')
STOPWORDS = set(stopwordlist)

#Cleaning and removing the above stop words list from the tweet text
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
dataset['tweets'] = dataset['tweets'].apply(lambda text: cleaning_stopwords(text))
dataset['tweets'].head()


# In[18]:


#Cleaning and removing punctuations
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
dataset['tweets']= dataset['tweets'].apply(lambda x: cleaning_punctuations(x))
dataset['tweets'].tail()


# In[19]:


#remove special characters and punctuation from the 'tweets' column
import re

# Define a function to remove special characters and punctuation
def remove_special_chars(text):
    # Remove any URLs
    text = re.sub(r'http\S+', '', text)
    # Remove any non-alphanumeric characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove any digits
    text = re.sub(r'\d+', '', text)
     # Remove any mentions
    text = re.sub(r'@\w+', '', text)
    return text
# Apply the function to the 'tweets' column
dataset['tweets'] = dataset['tweets'].apply(remove_special_chars)
dataset


# In[20]:


#Getting tokenization of tweet text

tokenizer = TweetTokenizer()
dataset['tweets'] = dataset['tweets'].apply(tokenizer.tokenize)
dataset['tweets'].head()


# In[21]:


#Applying stemming
st = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [st.stem(word) for word in data]
    return data
dataset['tweets']= dataset['tweets'].apply(lambda x: stemming_on_text(x))
dataset['tweets'].head()


# In[22]:


#Applying lemmatizer
lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
dataset['tweets'] = dataset['tweets'].apply(lambda x: lemmatizer_on_text(x))
dataset['tweets'].head()


# In[23]:


# Separating figurative,irony, regular and sarcasm classified data
df_fig1=dataset[dataset['class']=='figurative']
df_fig1 = df_fig1.astype(str)
# Combine all the tweets into a single string
all_tweets = ' '.join(df_fig1['tweets'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_tweets)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[24]:


df_irony1=dataset[dataset['class']=='irony']
df_irony1 = df_irony1.astype(str)
# Combine all the tweets into a single string
all_tweets = ' '.join(df_irony1['tweets'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_tweets)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[25]:


df_regular1=dataset[dataset['class']=='regular']
df_regular1 = df_regular1.astype(str)
# Combine all the tweets into a single string
all_tweets = ' '.join(df_regular1['tweets'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_tweets)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[26]:


df_sarcasm1=dataset[dataset['class']=='sarcasm']
df_sarcasm1 = df_sarcasm1.astype(str)
# Combine all the tweets into a single string
all_tweets = ' '.join(df_sarcasm1['tweets'])

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_tweets)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[27]:


#Separating input feature and label
X = dataset["tweets"]
X = X.astype(str)
Y 


# In[28]:


#Splitting Our Data Into Train and Test Subsets
# Separating the 80% data for training data and 20% for testing data
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.20, random_state =26)


# In[29]:


# Create an instance of the TF-IDF vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=300)

# Fit the vectorizer on the training data
vectorizer.fit(X_train)


# In[30]:


# Get the feature names
feature_names = vectorizer.get_feature_names_out()

# Print the feature names
for feature_name in feature_names:
    print(feature_name)


# In[31]:


#Transform the data using TF-IDF Vectorizer
X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)


# In[32]:



# Initialize and train the MNB model
mnb_model = MultinomialNB(alpha=1.0)
mnb_model.fit(X_train, y_train)

# Make predictions on the test set
mnb_pred = mnb_model.predict(X_test)

# Compute accuracy score
accuracy = accuracy_score(y_test, mnb_pred)

# Compute precision score
precision = precision_score(y_test, mnb_pred, average='weighted')

# Compute recall score
recall = recall_score(y_test, mnb_pred, average='weighted')

# Compute F1 score
f1 = f1_score(y_test, mnb_pred, average='weighted')


# Evaluate the performance of the classifier
print(classification_report(y_test, mnb_pred))

# Print the scores
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# In[33]:


# Calculate the confusion matrix
cm = confusion_matrix(y_test, mnb_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
classes = np.unique(y_test)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Add labels to each cell in the confusion matrix
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.show()


# In[34]:


# Calculate the probabilities for each class
probs = mnb_model.predict_proba(X_test)
probs = probs[:, 1]  # Keep probabilities of positive class only

# Binarize the labels for ROC curve plotting
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# Calculate the probabilities for each class
probs = mnb_model.predict_proba(X_test)

# Plot the ROC curve for each class
plt.figure(figsize=(8, 6))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[35]:


# Binarize the labels for ROC curve plotting
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

# Calculate the probabilities for each class
probs = mnb_model.predict_proba(X_test)

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and AUC
fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i in range(len(classes)):
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])

# Plot micro-average ROC curve
plt.plot(fpr_micro, tpr_micro, label='Micro-average ROC curve (area = %0.2f)' % roc_auc_micro, linestyle=':', linewidth=4)

# Plot random guessing line
plt.plot([0, 1], [0, 1], 'k--')

# Set plot properties
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[88]:


"""LRmodel = LogisticRegression(random_state=42, max_iter=1000, penalty='l2', C=1.0, solver='saga')
LRmodel.fit(X_train, y_train)

# Predict the class labels for the testing tweets
y_pred = LRmodel.predict(X_test)

# Compute accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Compute precision score
precision = precision_score(y_test, y_pred, average='weighted')

# Compute recall score
recall = recall_score(y_test, y_pred, average='weighted')

# Compute F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
from sklearn.metrics import classification_report
# Evaluate the performance of the classifier
print(classification_report(y_test, y_pred))

# Print the scores
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1) """


# In[74]:


"""# Train an SVM classifier
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train, y_train)

# Predict the class labels for the testing tweets
y_pred = classifier.predict(X_test)

# Compute accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Compute precision score
precision = precision_score(y_test, y_pred, average='weighted')

# Compute recall score
recall = recall_score(y_test, y_pred, average='weighted')

# Compute F1 score
f1 = f1_score(y_test, y_pred, average='weighted')
from sklearn.metrics import classification_report
# Evaluate the performance of the classifier
print(classification_report(y_test, y_pred))

# Print the scores
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)"""


# In[75]:


"""from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Assuming you have X_train, X_test, y_train, y_test as your training and testing data

# Initialize models
nb_model = MultinomialNB()
rf_model = RandomForestClassifier()
mlp_model = MLPClassifier()

# Fit models
nb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
mlp_model.fit(X_train, y_train)

# Predict using trained models
nb_pred = nb_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
mlp_pred = mlp_model.predict(X_test)

# Generate evaluation metrics for each model
nb_report = classification_report(y_test, nb_pred)
rf_report = classification_report(y_test, rf_pred)
mlp_report = classification_report(y_test, mlp_pred)

nb_accuracy = accuracy_score(y_test, nb_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
mlp_accuracy = accuracy_score(y_test, mlp_pred)

nb_f1 = f1_score(y_test, nb_pred, average='weighted')
rf_f1 = f1_score(y_test, rf_pred, average='weighted')
mlp_f1 = f1_score(y_test, mlp_pred, average='weighted')

nb_precision = precision_score(y_test, nb_pred, average='weighted')
rf_precision = precision_score(y_test, rf_pred, average='weighted')
mlp_precision = precision_score(y_test, mlp_pred, average='weighted')

nb_recall = recall_score(y_test, nb_pred, average='weighted')
rf_recall = recall_score(y_test, rf_pred, average='weighted')
mlp_recall = recall_score(y_test, mlp_pred, average='weighted')

# Print the evaluation metrics for each model
print("Naive Bayes Report:")
print(nb_report)
print("Naive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes F1 Score:", nb_f1)
print("Naive Bayes Precision Score:", nb_precision)
print("Naive Bayes Recall Score:", nb_recall)

print("\nRandom Forest Report:")
print(rf_report)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest F1 Score:", rf_f1)
print("Random Forest Precision Score:", rf_precision)
print("Random Forest Recall Score:", rf_recall)

print("\nMLP Classifier Report:")
print(mlp_report)
print("MLP Classifier Accuracy:", mlp_accuracy)
print("MLP Classifier F1 Score:", mlp_f1)
print("MLP Classifier Precision Score:", mlp_precision)
print("MLP Classifier Recall Score:", mlp_recall)
"""


# In[81]:


"""from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Initialize additional models
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
ada_model = AdaBoostClassifier()
gb_model = GradientBoostingClassifier()


# Fit additional models
dt_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)
ada_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)

# Predict using additional models
dt_pred = dt_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
ada_pred = ada_model.predict(X_test)
gb_pred = gb_model.predict(X_test)

# Generate evaluation metrics for additional models
dt_report = classification_report(y_test, dt_pred)
knn_report = classification_report(y_test, knn_pred)
ada_report = classification_report(y_test, ada_pred)
gb_report = classification_report(y_test, gb_pred)

dt_accuracy = accuracy_score(y_test, dt_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)
ada_accuracy = accuracy_score(y_test, ada_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)

dt_f1 = f1_score(y_test, dt_pred, average='weighted')
knn_f1 = f1_score(y_test, knn_pred, average='weighted')
ada_f1 = f1_score(y_test, ada_pred, average='weighted')
gb_f1 = f1_score(y_test, gb_pred, average='weighted')

dt_precision = precision_score(y_test, dt_pred, average='weighted')
knn_precision = precision_score(y_test, knn_pred, average='weighted')
ada_precision = precision_score(y_test, ada_pred, average='weighted')
gb_precision = precision_score(y_test, gb_pred, average='weighted')

dt_recall = recall_score(y_test, dt_pred, average='weighted')
knn_recall = recall_score(y_test, knn_pred, average='weighted')
ada_recall = recall_score(y_test, ada_pred, average='weighted')
gb_recall = recall_score(y_test, gb_pred, average='weighted')

# Print the evaluation metrics for additional models
print("Decision Tree Report:")
print(dt_report)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree F1 Score:", dt_f1)
print("Decision Tree Precision Score:", dt_precision)
print("Decision Tree Recall Score:", dt_recall)

print("\nK-Nearest Neighbors Report:")
print(knn_report)
print("K-Nearest Neighbors Accuracy:", knn_accuracy)
print("K-Nearest Neighbors F1 Score:", knn_f1)
print("K-Nearest Neighbors Precision Score:", knn_precision)
print("K-Nearest Neighbors Recall Score:", knn_recall)

print("\nAdaBoost Report:")
print(ada_report)
print("AdaBoost Accuracy:", ada_accuracy)
print("AdaBoost F1 Score:", ada_f1)
print("AdaBoost Precision Score:", ada_precision)
print("AdaBoost Recall Score:", ada_recall)

print("\nGradient Boosting Report:")
print(gb_report)
print("Gradient Boosting Accuracy:", gb_accuracy)
print("Gradient Boosting F1 Score:", gb_f1)
print("Gradient Boosting Precision Score:", gb_precision)
print("Gradient Boosting Recall Score:", gb_recall)
"""


# In[104]:


"""from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier()
mlp_model.fit(X_train, y_train)
mlp_pred = mlp_model.predict(X_test)

mlp_report = classification_report(y_test, mlp_pred)
mlp_accuracy = accuracy_score(y_test, mlp_pred)
mlp_f1 = f1_score(y_test, mlp_pred, average='weighted')
mlp_precision = precision_score(y_test, mlp_pred, average='weighted')
mlp_recall = recall_score(y_test, mlp_pred, average='weighted')

print("Multi-Layer Perceptron Report:")
print(mlp_report)
print("Multi-Layer Perceptron Accuracy:", mlp_accuracy)
print("Multi-Layer Perceptron F1 Score:", mlp_f1)
print("Multi-Layer Perceptron Precision Score:", mlp_precision)
print("Multi-Layer Perceptron Recall Score:", mlp_recall)
"""


# In[ ]:




