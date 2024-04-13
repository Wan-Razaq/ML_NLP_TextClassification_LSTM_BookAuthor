#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import csv
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


# In[2]:


# Step 1: Read the CSV file as a single column of text.
df = pd.read_csv('train_spooky_author.csv', usecols=[0], header=None, names=['combined'])


# In[3]:


# Define a function to extract id, text, and author using regular expression.
def extract_data(row):
    # This regular expression looks for a pattern with id followed by numbers,
    # then any text within quotes, and ends with an acronym within quotes
    match = re.match(r'(id\d+),"(.*?)","(\w+)"', row)
    if match:
        return match.groups()
    else:
        # Return None for rows that do not match the pattern
        return [None, None, None]


# In[4]:


# Apply the function to each row
extracted_data = df['combined'].apply(extract_data)


# In[5]:


# Create a new DataFrame from the extracted data
df_extracted = pd.DataFrame(extracted_data.tolist(), columns=['id', 'text', 'author'])


# In[6]:


df_extracted.head()


# In[7]:


# Drop rows with None values which did not match the pattern
df_extracted.dropna(inplace=True)


# In[8]:


df_extracted.head()


# In[9]:


# Reset the index of the DataFrame
df_extracted.reset_index(drop=True, inplace=True)


# In[10]:


# Display the DataFrame to confirm the format
df_extracted.head()


# In[11]:


print(df_extracted.shape)


# In[12]:


# One hot-encoding
category = pd.get_dummies(df_extracted.author)
df_baru = pd.concat([df_extracted, category], axis=1)
df_baru = df_baru.drop(columns=['id','author'])

# Convert bolean values to integers 0 or 1
df_baru[['EAP', 'HPL', 'MWS']] = df_baru[['EAP', 'HPL', 'MWS']].astype(int)


# In[13]:


df_baru.head()


# In[14]:


# convert to Numpy Array
text = df_baru['text'].values
label = df_baru[['EAP', 'HPL', 'MWS']].values


# In[15]:


text


# In[16]:


label


# In[17]:


# Divide training data set
text_train , text_test, label_train, label_test = train_test_split(text, label, test_size=0.2)


# In[18]:


# Changing every words into number using Tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token='x')
tokenizer.fit_on_texts(text_train)
tokenizer.fit_on_texts(text_test)

Sequence_train = tokenizer.texts_to_sequences(text_train)
Sequence_test = tokenizer.texts_to_sequences(text_test)

padded_train = pad_sequences(Sequence_train)
padded_test = pad_sequences(Sequence_test)



# In[19]:


print("padded_train", padded_train)


# In[36]:


# Embedding and LSTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.Dropout(0.5), # Added dropout
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5), # Added dropout
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5), # Added dropout
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) # Early stopping



# In[37]:


# Training dataset
num_epochs = 30
history = model.fit(padded_train, label_train, epochs=num_epochs,
                    validation_data=(padded_test, label_test), 
                    callbacks=[callback], # Early stopping callback
                    verbose=2)


# In[ ]:




