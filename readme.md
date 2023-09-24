# Garbage Man

### 1. Add dataset
To run this code, first add dataset under the folder
garbage_classification, for example, after adding dataset, the sub folder and images
should be stored like "garbage_classification/battery/battery1.jpg..."

### 2. Process dataset
After adding dataset, create another folder 'processed_data' under the project root path,
then call the 'build_dataset' function in load_data.npy with specific dataset ("train", "test"
, or "val"). You need to call the function for these three sub-dataset separately, and the function
will normalize data, adjust image size, collect labels and save them as numpy array.

### 3. Train and test the model
Run train.py, the train_and_val function will train the model and validate the performance on the 
validation set. The model parameter is saved under the saved_model folder.
After finishing training, run test.py to test the model on test set, you will see the accuracy and 
confusion matrix.

## Classification result
![image](https://github.com/hangqkth/Garbage_man/assets/112768477/a49fcc49-90b4-4013-a00a-7493e5f33d15)
