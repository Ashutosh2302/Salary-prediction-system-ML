import os,time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
def welcome():
    print("Wlecome to Salary prediction Application")
    input("Press enter to continue...")
def graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred):
    plt.scatter(X_train,Y_train,color='red',label='training data')
    plt.plot(X_train,regressionObject.predict(X_train),color='blue',label='Best Fit')
    plt.scatter(X_test,Y_test,color='green',label='Test Data')
    plt.scatter(X_test,Y_pred,color='black',label='Predicted Test Data')
    plt.title("Salary vs Grades")
    plt.xlabel("Grades")
    plt.ylabel("Salary")
    plt.legend()
    plt.show()

def main():
    welcome()
    try:
        csv_files=checkcsv()
        if csv_files==0:
            raise FileNotFoundError('No csv file in directory')
        print(csv_files)
        csv_file=get_user_choice_csv(csv_files)
        print(csv_file," is selected")
        print('Reading csv file')
    
        print('Creating Dataset')
       
        dataset=pd.read_csv(csv_file)
     
        print(dataset)
        print('Dataset created')
        X=dataset.iloc[:,:-1].values
        Y=dataset.iloc[:,-1].values
        test_data_size=float(input("Enter test data size (between 0 and 1)"))
        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_data_size)
       
        
        print("Model creation in progress..")
        regressionObject=LinearRegression()
        regressionObject.fit(X_train,Y_train)
        print("Model is created")
        print("Press ENTER key to predict test data in trained model")
        input()
        Y_pred=regressionObject.predict(X_test)
        index=0
        print('X test','...','Y test','...','Y predicted')
        while index<len(X_test):
            print(X_test[index],'...',Y_test[index],'...',Y_pred[index])
            index+=1
        input("Press ENTER key to see above result in graphical format")
        graph(X_train,Y_train,regressionObject,X_test,Y_test,Y_pred)
        r2=r2_score(Y_test,Y_pred)
        print("Our model is %2.2f%% accurate" %(r2*100))

        print("Now you can predict salary of an employee using our model")
        print("\nEnter grades, seperated by commas")

        grades=[float(e) for e in input().split(',')]
        g=[]
        for x in grades:
            g.append([x])
        grade=np.array(g)
        #print(grade.ndim)
        salaries=regressionObject.predict(grade)
        plt.scatter(grade, salaries ,color='black')
        plt.xlabel('Grades')
        plt.ylabel('Salaries')
        plt.show()

        d=pd.DataFrame({'Grades':grades,'Salaries':salaries})
        print(d)
        
    except FileNotFoundError:
        print('No csv file in directory')
        print("Press Enter key to exit")
        input()
        exit()

def checkcsv():
    csv_files=[]
    cur_dir=os.getcwd()
    content_list=os.listdir(cur_dir)
    print(content_list)
    for file in content_list:
        if file.split('.')[-1]=='csv':
            csv_files.append(file)

    if len(csv_files)==0:
        return 0
    else:
        return csv_files
    
def get_user_choice_csv(csv_files):
    index=1
    for file in csv_files:
        print(index,"  ",file )
        index+=1
    return csv_files[int(input("Select file to create ML model"))-1]

if  __name__=="__main__":
    main()
    input()
    
