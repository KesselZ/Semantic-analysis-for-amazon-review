import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re
import sklearn.model_selection as ms
from keras import models
import tokenization
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay

#1.Read the dataset
data = pd.read_csv('Reviews.csv')
print("The shape of dataset is:",data.shape)
data_number=500000

#2.Keep only 500000 data, and make some balance on the dataset
data=data.loc[1:data_number,:]
data=data.drop(data[(data['Score']==5)&(data['Id']).isin(range(int(data_number*0.8)))].index)
data=data.drop(data[(data['Score']==4)&(data['Id']).isin(range(int(data_number*0.8)))].index)

#3.Analysis the number of scores
fig= plt.subplots(1, 1, figsize=(8, 4))
sns.countplot(data=data, x='Score')
plt.show()

#4.Do more analysis
print(data.describe())
print(data.dtypes)

#5.Clean the data, drop the NULL data
print(data.isnull().sum())
data=data.dropna()


#6.If score <=2, it is a negative comment(0), otherwise it is a positive comment(1)
data.loc[data['Score'] <=2, 'Score'] = 0
data.loc[data['Score'] > 3, 'Score'] = 1


#7.The score=3 is considered a neutral comment, drop that.
data.drop(data[data['Score']==3].index,inplace=True)

#8.Calculate and record the length of text for each row
data['len'] = data.Text.apply(lambda x: len(x.split()))

#9.Drop rows that has too words in "text"
data = data[data.len<50]

#10.Remove html tags
data['Text']=data['Text'].apply(lambda row : re.sub('<.*?>','',row))

#11.Split the data
X_train, X_test, y_train, y_test = ms.train_test_split(data[['Text','len']],data.Score, test_size=0.2, stratify=data.Score)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

#12. Check the shape
print("Shape of X_train and X_test:")
print(X_train.shape,X_test.shape)
print("Shape of y_train and y_test:")
print(y_train.shape,y_test.shape)

#13. Build the bert input
max_seq_length = 55
# tf.keras.backend.clear_session()

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

#14. Import the bert
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=False)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output)
bert_model.summary()

#15. Load the tokenizer
print("Loading tokenizer...")
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file,do_lower_case)

#16. Process the data into the bert input
def bert_preprocess(sentence, tokenizer, max_seq_len=512):
    tokens = []
    masks = []
    segments = []
    for word in sentence:
        word = tokenizer.tokenize(word)
        word = word[:max_seq_len - 2]
        seq = ["[CLS]"] + word + ["[SEP]"]
        # print(seq)
        token = tokenizer.convert_tokens_to_ids(seq)
        # print(token)
        padding_tokens = token + [0] * (max_seq_len - len(token))
        mask = [1] * len(seq)
        masking = mask + [0] * (max_seq_len - len(token))
        segment = np.zeros(max_seq_length)

        tokens.append(padding_tokens)
        masks.append(masking)
        segments.append(segment)
    return np.array(tokens), np.array(masks), np.array(segments)

print("Preprocessing bert input...")
X_train_tokens, X_train_mask, X_train_segment=bert_preprocess(X_train.Text.values,tokenizer,55)
X_test_tokens, X_test_mask, X_test_segment=bert_preprocess(X_test.Text.values,tokenizer,55)


#17. Get the bert output
print("Calculating BERT output...")
X_train_pooled_output=bert_model.predict([X_train_tokens,X_train_mask,X_train_segment])
X_test_pooled_output=bert_model.predict([X_test_tokens,X_test_mask,X_test_segment])


#18. Set the callback for the AUC calculating
auc = []
val_auc = []
class AUC_CallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        auc_train = (roc_auc_score(y_train, (self.model.predict(X_train_pooled_output))))
        auc_test = (roc_auc_score(y_test, (self.model.predict(X_test_pooled_output))))
        print(' - train auc: {0:.4f} - test auc: {1:.4f}'.format(auc_train, auc_test))
        auc.append(auc_train)
        val_auc.append(auc_test)

myCallback = AUC_CallBack()


#19. Build the supervised learning model by Keras
model = models.Sequential()
model.add(Dense(400, activation='relu',input_shape=(768,)))
model.add(Dense(220, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

opt= tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss=tf.keras.losses.BinaryCrossentropy(),metrics='accuracy')

tf.keras.backend.clear_session()


#20. Train the model
his=model.fit(X_train_pooled_output,y_train,validation_data=(X_test_pooled_output,y_test),epochs=50,callbacks=[myCallback])


#21. Plot the picture
plt.figure(figsize=(12,10))
plt.plot(np.arange(len(auc)),auc)
plt.plot(np.arange(len(auc)),val_auc)
plt.scatter(np.arange(len(auc)),auc)
plt.scatter(np.arange(len(auc)),val_auc)
plt.xlabel('Epochs',fontsize=18)
plt.ylabel('AUC Score',fontsize=18)
plt.legend(['Train AUC','Test AUC'],fontsize=18)
plt.title("The performance of BERT model on Amazon dataset",fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


#22. Make the classification report, and confusion matrix
y_pred = model.predict(X_test_pooled_output)
y_p=np.argmax(y_pred,axis=1)
y_t=np.argmax(y_test,axis=1)
print('Classification report:')
print(classification_report(y_t,y_p))

cm = confusion_matrix(y_t,y_p)
labels = ['Negative','Positive']
disp = ConfusionMatrixDisplay(cm,display_labels = labels)
disp. plot()
plt. title('Confusion matrix')
plt. show()


#23. Apply some experiment on this model
example=["This is great stuff. Made some really tasty banana bread. Good quality and lowest price in town.",
"This oatmeal is good. Its mushy, soft, I like it. Quaker Oats is the way to go.",
"Terrible! Artificial lemon taste, like Pledge Lemon Furniture Polish. Don't do this to yourself. ",
"Just awful! I thought food was supposed to taste good! I had to eat ice cream afterwards to get the taste out of my mind"
]

print("Calculating example sentence...")
E_test_tokens, E_test_mask, E_test_segment=bert_preprocess(example,tokenizer,55)
E_test_pooled_output=bert_model.predict([E_test_tokens,E_test_mask,E_test_segment])
example_pred=model.predict(E_test_pooled_output)
print("Their possibility distribution is:",example_pred)
example_pred=np.argmax(example_pred,axis=1)
print("Their predicted classification is:",example_pred)
