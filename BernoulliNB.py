import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Definir a seed para reprodutibilidade
SEED = 5
np.random.seed(SEED)

# Dividir os dados em treino e teste (70% treino, 30% restante para validação e teste)
treino_x, teste_x, treino_y, teste_y = train_test_split(X, y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        random_state=SEED)

# Dividir os dados de teste em validação e teste (15% validação, 15% teste)
valid_x, teste_x, valid_y, teste_y = train_test_split(teste_x, teste_y,
                                                      test_size=0.5,
                                                      stratify=teste_y,
                                                      random_state=SEED)

# Criar e treinar o modelo Bernoulli Naive Bayes
modelo = BernoulliNB()
modelo.fit(treino_x, treino_y)

# Fazer previsões no conjunto de teste
predicoes = modelo.predict(teste_x)

# Avaliar o modelo
acuracia = accuracy_score(teste_y, predicoes) * 100
print(f'Acurácia é de {acuracia:.2f}%')
print('Treinamos com %d elementos, validamos com %d e testamos com %d elementos.' % (len(treino_x), len(valid_x),
                                                                                     len(teste_y)))

# Verificar a distribuição das classes nos conjuntos de treino, validação e teste
print("Distribuição de classes no conjunto de treino:")
print(np.bincount(treino_y))
print("\nDistribuição de classes no conjunto de validação:")
print(np.bincount(valid_y))
print("\nDistribuição de classes no conjunto de teste:")
print(np.bincount(teste_y))

# Salvar o modelo
filename = 'modelo_treinado_BernoulliNB.pkl'
pickle.dump(modelo, open(filename, 'wb'))
