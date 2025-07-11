from prediction_helpers import *

ExE = pd.read_csv('../../extracted_data/ExE_imputed_5.csv', sep=',', index_col=0)
NxN = pd.read_csv('../../extracted_data/NxN_imputed_10.csv', sep=',', index_col=0)
ExN = pd.read_csv('../../extracted_data/ExN_filled_20.csv', sep=',', index_col=0)

predictive_models = [ 
    Ridge(alpha=5)
]

random.seed(35)
proportion = 1


results_ExE = iterate_over_proportion_only_interactions(ExE, proportion, predictive_models)
results_NxN = iterate_over_proportion_only_interactions(NxN, proportion, predictive_models)
results_ExN = iterate_over_proportion_only_interactions(ExN, proportion, predictive_models)


results = [results_ExE, results_NxN, results_ExN]


output_file = "output/results.txt"

with open(output_file, "w") as f:

    datasets = [
        ("ExE", results_ExE),
        ("ExN", results_ExN), 
        ("NxN", results_NxN)
    ]
    
    for dataset_name, results in datasets:
        f.write(f"Results for dataset {dataset_name}\n")
        f.write("="*50 + "\n")
        f.write("="*50 + "\n")

        for model in predictive_models:
            top_genes_df, average_r2 = print_top_genes(results, model.__class__.__name__, 20)
            
            f.write(f"\nTop Genes for Model: {model.__class__.__name__}\n")
            f.write(f"Average R²: {average_r2:.4f}\n")
            f.write(top_genes_df.to_string(index=False))
            f.write("\n" + "="*50 + "\n")

print(f"Results written to {output_file}")

