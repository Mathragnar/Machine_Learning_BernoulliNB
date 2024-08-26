import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pickle
from scipy.stats import uniform

# Definir a seed para reprodutibilidade
SEED = 5
np.random.seed(SEED)

# Dividir os dados em treino e teste (70% treino, 30% restante para validação e teste)
treino_x, teste_x, treino_y, teste_y = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=SEED
)

# Dividir os dados de teste em validação e teste (15% validação, 15% teste)
valid_x, teste_x, valid_y, teste_y = train_test_split(
    teste_x, teste_y, test_size=0.5, stratify=teste_y, random_state=SEED
)

# --- Opção 1: Grid Search ---
# Definir grid de hiperparâmetros
parametros_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}  # Espaço de busca para alpha

# Criar objeto GridSearchCV
grid_search = GridSearchCV(BernoulliNB(), parametros_grid, cv=5, scoring='accuracy')
grid_search.fit(treino_x, treino_y)

# Imprimir melhores hiperparâmetros encontrados pelo Grid Search
print("Melhores parâmetros (Grid Search):", grid_search.best_params_)
print("Acurácia (Grid Search):", grid_search.best_score_)

# --- Opção 2: Random Search ---
# Definir espaço de busca para cada hiperparâmetro
parametros_random = {'alpha': uniform(loc=0.01, scale=10)}  # Distribuição uniforme para alpha

# Criar objeto RandomizedSearchCV
random_search = RandomizedSearchCV(BernoulliNB(), parametros_random, n_iter=10, cv=5, scoring='accuracy', random_state=SEED)
random_search.fit(treino_x, treino_y)

# Imprimir melhores hiperparâmetros encontrados pelo Random Search
print("Melhores parâmetros (Random Search):", random_search.best_params_)
print("Acurácia (Random Search):", random_search.best_score_)

# --- Escolher o melhor modelo e avaliar no conjunto de teste ---
melhor_modelo = grid_search.best_estimator_ if grid_search.best_score_ > random_search.best_score_ else random_search.best_estimator_

# Fazer previsões no conjunto de teste
predicoes = melhor_modelo.predict(teste_x)

# Avaliar o modelo
acuracia = accuracy_score(teste_y, predicoes) * 100
print(f'Acurácia do melhor modelo no conjunto de teste: {acuracia:.2f}%')

# Salvar o melhor modelo
filename = 'modelo_treinado.pkl'
pickle.dump(melhor_modelo, open(filename, 'wb'))