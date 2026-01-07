from scipy.stats import wilcoxon, friedmanchisquare
import itertools

# Example accuracy values for 3 models over 5 test datasets
mistral_single = {'mistral_single amazon': [0.9084, 0.6191, 0.7451,0.7187],
                  'mistral_single dad jokes': [0.6882,	0.9404,	0.6811,	0.7091],
                  'mistral_single headlines': [0.6504,	0.5575,	0.9669,	0.6153],
                  'mistral_single one liners': [0.6367,0.5116,	0.6738,	0.9467]}

mistral_pair = {'mistral_pair amazon dad jokes': [0.8948,0.9498,0.735,	0.7394],
                 'mistral_pair amazon headlines': [0.8933,0.6693,	0.9647,	0.666],
                 'mistral_pair dad jokes headlines': [0.705,0.9051,	0.9467,	0.6964],
                 'mistral_pair dad jokes one liners': [0.6626,0.9544,	0.7009,	0.944],
                 'mistral_pair headlines one liners': [0.6758,0.524,	0.9673,	0.9404],
                 'mistral_pair one liners amazon': [00.9024,0.5342,	0.7407,	0.9436]}

mistral_loo = {'mistral_loo amazon': [0.6551,0.9433,	0.9589,	0.9356],
             'mistral_loo dad jokes': [0.898,0.5524,	0.9642,	0.9351],
             'mistral_loo headlines': [0.8953,0.948,	0.7338,	0.9344],
             'mistral_loo one liners': [0.8987,	0.956,	0.9604,	0.736]}


llama_single = {'llama_single amazon': [0.8823,	0.6515,	0.6539,	0.6111],
                  'llama_single dad jokes': [0.6263,	0.9279,	0.5932,	0.7009],
                  'llama_single headlines': [0.5807,	0.5678,	0.9021,	0.6543],
                  'llama_single one liners': [0.6233,	0.6184,	0.54,	0.8677]}

llama_pair = { 'llama_pair amazon dad jokes': [0.8247,	0.8953,	0.6704,	0.6926],
                 'llama_pair amazon headlines': [0.8208,	0.6197,	0.895,	0.686],
                 'llama_pair dad jokes headlines': [0.6271,	0.8942,	0.9169,	0.7083],
                 'llama_pair dad jokes one liners': [0.738,	0.8971,	0.6507,	0.9124],
                 'llama_pair headlines one liners': [0.6326,	0.549,	0.9278,	0.9202],
                 'llama_pair one liners amazon': [0.8613,	0.5559,	0.6529,	0.9133]}

llama_loo = {'llama_loo amazon': [0.6893,	0.8782,	0.8944,	0.8809],
               'llama_loo dad jokes': [0.8389,	0.5718,	0.8978,	0.8724],
               'llama_loo headlines': [0.8604,	0.8871,	0.6813,	0.8787],
               'llama_loo one liners': [0.8404,	0.8933,	0.9069,	0.6935]}
model2 = [0.79, 0.84, 0.77, 0.79, 0.81]
model3 = [0.85, 0.87, 0.80, 0.82, 0.86]

# models = {'M1': model1, 'M2': model2, 'M3': model3}
# models = {**mistral_single, **mistral_pair, **mistral_loo}
models = {**llama_single, **llama_pair, **llama_loo}
print(models)

stat, p = friedmanchisquare(*models.values())
print(f"Friedman test statistic: {stat}, p-value: {p}")

models = {mistral_single, mistral_pair, mistral_loo}
model_names = list(models.keys())

# Perform pairwise Wilcoxon tests and collect p-values
p_values = {}
for (name1, name2) in itertools.combinations(model_names, 2):
    stat, p = wilcoxon(models[name1], models[name2])
    p_values[(name1, name2)] = p

# Bonferroni correction
num_tests = len(p_values)
alpha = 0.05
adjusted_alpha = alpha / num_tests

# Display results
for pair, p in p_values.items():
    print(f"{pair[0]} vs {pair[1]}: p = {p:.4f} ", end='')
    if p < adjusted_alpha:
        print(f"-> Significant after Bonferroni (adjusted α = {adjusted_alpha:.4f})")
    else:
        print(f"-> Not significant (adjusted α = {adjusted_alpha:.4f})")