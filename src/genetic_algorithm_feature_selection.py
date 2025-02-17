import numpy as np
import pandas as pd
import random
from deap import base, creator, tools, algorithms

# Load processed data
df = pd.read_csv("data/processed_data.csv")

# Separate features and target
X = df.drop(columns=["isFraud"])
y = df["isFraud"]

# Define Genetic Algorithm components
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Initialize Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def fitness(individual):
    selected_features = [i for i, val in enumerate(individual) if val == 1]
    if len(selected_features) == 0:
        return (0,)
    
    # Updated fitness function: Encourage diverse feature selection
    score = len(selected_features) / len(individual)  # Simple proportion of selected features
    return (score,)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run Genetic Algorithm
def run_ga():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=30, halloffame=hof, verbose=True)
    return hof[0]

best_features = run_ga()
selected_features = [i for i, val in enumerate(best_features) if val == 1]

# Save selected features to file
with open("data/selected_features.txt", "w") as f:
    f.write("\n".join(map(str, selected_features)))

# Ensure only feature indices are written to the file
if selected_features:  # Check if features were selected
    with open("data/selected_features.txt", "w") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")  # Write only the index
    print("✅ Selected Features saved to data/selected_features.txt")
else:
    print("⚠️ No features were selected by the Genetic Algorithm!")
