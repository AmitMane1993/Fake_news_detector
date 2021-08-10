import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class Dataprep:
    
    def __init__(self,dataframe):
        self.df = dataframe
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def train_and_test_split(self):
        # Pre processing - reshape the data and preparing to train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.cleanedText, 
                                                                                self.df.label, 
                                                                                test_size=0.25,
                                                                                random_state=42)
    
    def tokenizing(self):
        t = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
        
        # fit the tokenizer on the documents
        t.fit_on_texts(self.df.cleanedText)
        t.word_index['<PAD>'] = 0
        
        self.train_and_test_split()
        
        train_sequences = t.texts_to_sequences(self.X_train)
        test_sequences = t.texts_to_sequences(self.X_test)
        
        MAX_SEQUENCE_LENGTH = 120
        
        # pad dataset to a maximum review length in words
        self.X_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
        self.X_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH,padding='post')
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        
        return self.X_train, self.X_test, self.y_train, self.y_test, len(t.word_index)