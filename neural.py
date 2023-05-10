from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#Faz a divisão dos dados para teste e treino
def test_and_train_separation(charac,proac):
    x_train,x_test,y_train,y_test=train_test_split(
    charac,
    proac,
    #Tamanho do teste
    test_size=0.2,
    #Fixar as amostras
    random_state=10
    )   
    
    return x_train,x_test,y_train,y_test
    
def neural_train(charac,proac):
    
    pipeline=Pipeline([('mlp',MLPClassifier())])
    
    parameters={
    'mlp__hidden_layer_sizes':[(50,),(100,),(150,),(200,)],
    'mlp__activation':['identity', 'logistic', 'tanh', 'relu'],
    'mlp__solver':['lbfgs', 'sgd', 'adam'],
    'mlp__learning_rate':['constant', 'invscaling', 'adaptive'],
    }
    
    grid_search=GridSearchCV(pipeline,param_grid=parameters,cv=5,verbose=2)
    grid_search.fit(charac,proac)
    
    return grid_search.best_score_
    
     
   
     
def prediction(x_test,y_test,mlp):
    y_pred=mlp.predict(x_test)
    
    return accuracy_score(y_test,y_pred)
    