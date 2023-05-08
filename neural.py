from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    
def neural_train(x_train,y_train):
    mlp=MLPClassifier(
        hidden_layer_sizes=(30),
    )
    
    mlp.fit(x_train,y_train)
    
    return mlp
    
def prediction(x_test,y_test,mlp):
    y_pred=mlp.predict(x_test)
    
    return accuracy_score(y_test,y_pred)
    