import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_train_split(data,train_ratio):
    
    ''' 
    Splits the dataset with respect to the test train ratio
    data: dataset to split
    train_ratio: the ratio of train part to the whole dataset 
    returns test set, train set
    '''

    lenght_of_dataset = len(data)
    amount_of_data = int((train_ratio/100)*lenght_of_dataset)

    train_set = []
    test_set = []

    random_index_list = []

    for index in range(amount_of_data):
        while len(random_index_list) <= amount_of_data:
            random_index = random.randrange(lenght_of_dataset)
            if random_index not in random_index_list:
                random_index_list.append(random_index)
            else:
                pass

    for index in range(lenght_of_dataset):
        if index not in random_index_list:
            test_set.append(data.iloc[index])
        else:
            train_set.append(data.iloc[index])

    return pd.DataFrame(test_set), pd.DataFrame(train_set)

def select_future(data = pd.DataFrame): 
    
    ''' 
    This function takes the best feature to choose for optimal gain
    returns the classes and their relevence, the higher is better
    '''

    # its assumed that one column is for results and its the last column
    number_of_features = len(data.columns) - 1

    purity = {}
    

    for index in range(number_of_features):

        
        Kirmizi_Pistachio = 0
        Siit_Pistachio = 0
        
        Kirmizi_Pistachio_values = 0
        Siit_Pistachio_values = 0

        std_dev = np.std(data[data.columns[index]])
        mean = np.mean(data[data.columns[index]])

        for inner_index in range(len(data[data.columns[index]])):
            
            # i know that there is 2 possible pistachios in the dataset, so i hard-coded them.
            if data.iloc[inner_index][-1] == 'Kirmizi_Pistachio':
                Kirmizi_Pistachio_values += data.iloc[inner_index][0]
                Kirmizi_Pistachio += 1
            elif data.iloc[inner_index][-1] == 'Siit_Pistachio':
                Siit_Pistachio_values += data.iloc[inner_index][0]
                Siit_Pistachio += 1

        mean_siit = Siit_Pistachio_values/len(data[data.columns[index]])
        mean_kirmizi = Kirmizi_Pistachio_values/len(data[data.columns[index]])

        diff_ratio = str(abs(mean_kirmizi/mean_siit))

        if std_dev > float(diff_ratio):
            print(f'First breakpoint {mean - float(std_dev)*float(diff_ratio)}')
            print(f'Second breakpoint {mean + float(std_dev)*float(diff_ratio)}')
            purity.update({data.columns[index] : (abs(Kirmizi_Pistachio/Siit_Pistachio),'HP',((mean - float(std_dev)*float(diff_ratio)),(mean + float(std_dev)*float(diff_ratio))))})
            # HP -> high priority
            

        else:
            print(f'breakpoint {mean}')
            purity.update({data.columns[index] : (abs(Kirmizi_Pistachio/Siit_Pistachio),'LP',(mean))})
            #LP -> low priority

    return purity


def main():

    data = pd.read_csv('pistachio.csv')
    test,train = test_train_split(data,80)
    relevence = select_future(train)
    print(relevence)

if __name__=="__main__":
    main()