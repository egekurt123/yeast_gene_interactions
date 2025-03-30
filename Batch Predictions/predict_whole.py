from prediction_helpers import *

ExE = pd.read_csv('../extracted_data/ExE_imputed_5.csv', sep=',', index_col=0)
NxN = pd.read_csv('../extracted_data/NxN_imputed_10.csv', sep=',', index_col=0)
ExN = pd.read_csv('../extracted_data/ExN_filled_20.csv', sep=',', index_col=0)

predictive_models = [ 
    Ridge(alpha=5),
    Lasso(alpha=0.0001),
    RandomForestRegressor(n_estimators=10, max_depth=10),
]

random.seed(35)
proportion = 1


results_ExE = iterate_over_proportion(ExE, proportion, predictive_models)
results_NxN = iterate_over_proportion(NxN, proportion, predictive_models)
results_ExN = iterate_over_proportion(ExN, proportion, predictive_models)


results = [results_ExE, results_NxN, results_ExN]


output_file = "output/results.txt"

with open(output_file, "w") as f:

    for i, result in enumerate(results):

        f.write(f"Results for dataset {['ExE', 'NxN', 'ExN'][i]}\n")
        f.write("="*50 + "\n")
        f.write("="*50 + "\n")

        for model in predictive_models:
            top_genes_df = print_top_genes(results_ExE, model.__class__.__name__, 20)
        
            f.write(f"\nTop Genes for Model: {model.__class__.__name__}\n")
            f.write(top_genes_df.to_string(index=False))
            f.write("\n" + "="*50 + "\n")

print(f"Results written to {output_file}")

