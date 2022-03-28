from torch import true_divide
from train_test import train

if __name__ == "__main__":    
    data_folder = 'BRCA'
    testonly = True
    modelpath = './model/'
    train(data_folder, modelpath, testonly)