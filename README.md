# CIADEX

## Running Instructions
You can run the project by executing the following command:
'''
python main.py
'''

After running the above command, the program will generate three Excel files, which respectively display the processing results of the test dataset, the training dataset, and the validation dataset. The specific descriptions are as follows:  

output.xlsx: This file presents the processing results of the test dataset. The test dataset is used to evaluate the performance of the model on unseen data. The content in this file can help you understand the generalization ability of the model and its practical application effects.
train_data.xlsx: This file shows the processing results of the training dataset. The training dataset is the basic data used for training the model. By checking this file, you can view how the model learns from the data during the training process, which is helpful for analyzing the training effect and convergence of the model.  

valid_data.xlsx: This file displays the processing results of the validation dataset. The validation dataset is mainly used to adjust the hyperparameters of the model during the training process and evaluate the performance of the model under different parameter settings, so as to select the optimal model configuration. This file can assist you in judging the performance of the model on the validation set, ensuring the stability and reliability of the model.  

By reviewing these three files, you can comprehensively understand the performance of the model on different datasets, and then evaluate and optimize the model.
