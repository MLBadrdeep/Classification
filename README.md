# Getting started with Classification
  As the name suggests, Classification is the task of “classifying things” into sub-categories. Classification is part of supervised machine learning in which we put labeled data for training.

The article serves as a comprehensive guide to understanding and applying classification techniques, highlighting their significance and practical implications.

## What is Supervised Machine Learning?
Supervised Machine Learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output Y = f(X). The goal is to approximate the mapping function so well that when you have new input data (x) you can predict the output variables (Y) for that data.
Supervised learning problems can be further grouped into Regression and Classification problems.

#### Regression: 
Regression algorithms are used to predict a continuous numerical output. For example, a regression algorithm could be used to predict the price of a house based on its size, location, and other features.
#### Classification: 
Classification algorithms are used to predict a categorical output. For example, a classification algorithm could be used to predict whether an email is spam or not.

## Machine Learning for classification
Classification is a process of categorizing data or objects into predefined classes or categories based on their features or attributes.

Machine Learning classification is a type of supervised learning technique where an algorithm is trained on a labeled dataset to predict the class or category of new, unseen data.

The main objective of classification machine learning is to build a model that can accurately assign a label or category to a new observation based on its features.

For example, a classification model might be trained on a dataset of images labeled as either dogs or cats and then used to predict the class of new, unseen images of dogs or cats based on their features such as color, texture, and shape.

### Classification Types
There are two main classification types in machine learning:

#### Binary Classification
In binary classification, the goal is to classify the input into one of two classes or categories. Example – On the basis of the given health conditions of a person, we have to determine whether the person has a certain disease or not.

#### Multiclass Classification
In multi-class classification, the goal is to classify the input into one of several classes or categories. For Example – On the basis of data about different species of flowers, we have to determine which specie our observation belongs to.

### Classification Algorithms
There are various types of classifiers algorithms. Some of them are : 

##### Linear Classifiers
Linear models create a linear decision boundary between classes. They are simple and computationally efficient. Some of the linear classification models are as follows: 

- Logistic Regression
- Support Vector Machines having kernel = ‘linear’
- Single-layer Perceptron
- Stochastic Gradient Descent (SGD) Classifier
  
##### Non-linear Classifiers
- Non-linear models create a non-linear decision boundary between classes. They can capture more complex relationships between the input features and the target variable. Some of the non-linear - -
 
#### classification models are as follows: 

- K-Nearest Neighbours
- Kernel SVM
- Naive Bayes
- Decision Tree Classification
- Ensemble learning classifiers: 
- Random Forests, 
- AdaBoost, 
- Bagging Classifier, 
- Voting Classifier, 
- ExtraTrees Classifier
- Multi-layer Artificial Neural Networks


---------------------------------

#### Examples of Machine Learning Classification in Real Life
Classification algorithms are widely used in many real-world applications across various domains, including:

- Email spam filtering
- Credit risk assessment
- Medical diagnosis
- Image classification
- Sentiment analysis.
- Fraud detection
- Quality control
- Recommendation systems


#### Implementation of Classification Model in Machine Learning
Let’s get a hands-on experience with how Classification works. We are going to study various Classifiers and see a rather simple analytical comparison of their performance on a well-known, standard data set, the Iris data set.  

##### Requirements for running the given script:

Python
Scipy
Numpy
Pandas
matplotlib 

#### Conclusion
In conclusion, classification is a fundamental task in machine learning, involving the categorization of data into predefined classes or categories based on their features.


Results and Comparisons with model performance
