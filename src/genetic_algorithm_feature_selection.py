from deap import base, creator, tools, algorithms
import random

# Define GA fitness function (maximize fraud detection score)
def fitness(individual):
    selected_features = [i for i, val in enumerate(individual) if val == 1]
    if len(selected_features) == 0:
        return (0,)
    
    X_selected = X_train_resampled[:, selected_features]
    
    # Train SGD classifier on selected features
    model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_selected, y_train_resampled)
    
    score = model.score(X_selected, y_train_resampled)  
    return (score,)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)  
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def run_ga():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=30, halloffame=hof, verbose=True)

    return hof[0]  

best_features = run_ga()
selected_features = [i for i, val in enumerate(best_features) if val == 1]

print(f"Selected Feature Indices: {selected_features}")
