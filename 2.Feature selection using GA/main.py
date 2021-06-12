import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def init_population(n,c):
    return np.array([[math.ceil(e) for e in pop] for pop in (np.random.rand(n,c)-0.5)]), np.zeros((2,c))-1

def single_point_crossover(population):
    r,c, n = population.shape[0], population.shape[1], np.random.randint(1,population.shape[1])
    for i in range(0,r,2):
        population[i], population[i+1] = np.append(population[i][0:n],population[i+1][n:c]),np.append(population[i+1][0:n],population[i][n:c])
    return population

def flip_mutation(population):
    return population.max() - population

def random_selection(population):
    r = population.shape[0]
    new_population = population.copy()
    #print(new_population)
    for i in range(r):
        new_population[i] = population[np.random.randint(0,r)]
        #print(new_population[i])
    return new_population

def get_fitness(data, feature_list, target, population):
    fitness = []
    #print(population.shape[1])
    #print('madhur')
    #print('madhur')
    for i in range(population.shape[0]):
        columns = [feature_list[j] for j in range(population.shape[1]) if population[i,j]==1]
        fitness.append(predictive_model(data[columns], data[target]))
    return fitness

def predictive_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7)
    lr = LogisticRegression(solver='liblinear', max_iter=100, random_state=7)
    lr.fit(X_train,y_train)
    return accuracy_score(y_test, lr.predict(X_test))


def replace_duplicate(population , memory):
    return population,memory


def ga(data , feature_list , target , n , max_iter):
    c = len(feature_list)
    #print(c)
    population , memory = init_population(n , c)
    #print(population)
    #print(population)
    #print(memory)
    population , memory = replace_duplicate(population , memory)

    fitness = get_fitness(data , feature_list , target , population)

    optimal_value = max(fitness)
    optimal_solution = population[np.where(fitness == optimal_value)][0]

    for i in range(max_iter):
        population = random_selection(population)
        population = single_point_crossover(population)
        if np.random.rand() < 0.3:
            population = flip_mutation(population)

        population , memory = replace_duplicate(population , memory)

        fitness = get_fitness(data , feature_list , target , population)

        if max(fitness) > optimal_value:
            optimal_value = max(fitness)
            optimal_solution = population[np.where(fitness == optimal_value)][0]

    return optimal_solution , optimal_value






def main():
    df= pd.read_csv('labelled_data.csv')
    #print(df)
    df.drop(columns='id' , axis=1 , inplace=True)
    df.drop(columns='Unnamed: 0' , axis=1 , inplace=True)
    #print(df.columns)
    target = 'diagnosis'
    feature_list = [i for i in df.columns if i not in target]
    # Execute Genetic Algorithm to obtain Important Feature
    feature_set , acc_score = ga(df , feature_list , target , 10 , 1000)

    # Filter Selected Features
    feature_set = [feature_list[i] for i in range(len(feature_list)) if feature_set[i] == 1]
    print('Optimal Feature Set\n' , feature_set , '\nOptimal Accuracy =' , round(acc_score * 100) , '%')
    return feature_set
    # Print List of Features
  #

if __name__ == "__main__":
    l=main()