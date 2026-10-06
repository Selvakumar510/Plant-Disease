Project structure and dataset assumption
From the repo, the app is split into a Python AI service and a Node/React frontend shell:

app.py — Flask API for prediction
model_utils.py — loads the trained model and runs prediction
train.py — trains the CNN on the split dataset
split.py — splits the original raw dataset into train/val/test folders
server.js — Node backend entry
package.json — frontend app config
requirements.txt — Python dependencies
README.md — currently minimal
.env and .env.example — runtime environment values
What the code assumes about the dataset
The training code in train.py explicitly expects the dataset to be in a sibling folder like this:

../backend/data_split/train
../backend/data_split/val
../backend/data_split/test
That means the project assumes:

the raw dataset is first split into classes

each class sits in its own directory

the final folder structure is similar to:

data_split/
train/
Healthy_Leaf_Rose/
Mango-Healthy/
Mango-Powdery Mildew/
...
val/
Healthy_Leaf_Rose/
...
test/
Healthy_Leaf_Rose/
...

This matches the TensorFlow image loading call in train.py, where it uses:

tf.keras.preprocessing.image_dataset_from_directory(...)
class_names = train_ds.class_names
So the model is trained by folder names, not by filenames.

 
