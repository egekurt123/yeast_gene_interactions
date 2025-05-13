from prediction_helpers import *

ExE = pd.read_csv('../extracted_data/ExE_imputed_5.csv', sep=',', index_col=0)
NxN = pd.read_csv('../extracted_data/NxN_imputed_10.csv', sep=',', index_col=0)
ExN = pd.read_csv('../extracted_data/ExN_filled_20.csv', sep=',', index_col=0)

embeddings = pd.read_csv('../extracted_data/yeast_emb_embeddings_yeastnet_genex.csv', sep=',', index_col=0)
embeddings = embeddings.reset_index().set_index('gene_id')

predictive_models = [ 
    Ridge(alpha=5),
    Lasso(alpha=0.0001),
    RandomForestRegressor(n_estimators=10, max_depth=10),
]

random.seed(35)
proportion = 1


results_ExE = iterate_over_proportion_interactions_embeddings(ExE, embeddings, proportion, predictive_models)
results_NxN = iterate_over_proportion_interactions_embeddings(NxN, embeddings, proportion, predictive_models)
results_ExN = iterate_over_proportion_interactions_embeddings(ExN, embeddings, proportion, predictive_models)


results = [results_ExE, results_NxN, results_ExN]


output_file = "output/interaction_and_embedding/results.txt"

with open(output_file, "w") as f:

    #TODO extend for loop

    f.write(f"Results for dataset ExE\n")
    f.write("="*50 + "\n")
    f.write("="*50 + "\n")

    for model in predictive_models:
        top_genes_df = print_top_genes(results_ExE, model.__class__.__name__, 20)
        
        f.write(f"\nTop Genes for Model: {model.__class__.__name__}\n")
        f.write(top_genes_df.to_string(index=False))
        f.write("\n" + "="*50 + "\n")

    f.write(f"Results for dataset ExN\n")
    f.write("="*50 + "\n")
    f.write("="*50 + "\n")

    for model in predictive_models:
        top_genes_df = print_top_genes(results_ExN, model.__class__.__name__, 20)
        
        f.write(f"\nTop Genes for Model: {model.__class__.__name__}\n")
        f.write(top_genes_df.to_string(index=False))
        f.write("\n" + "="*50 + "\n")


    f.write(f"Results for dataset NxN\n")
    f.write("="*50 + "\n")
    f.write("="*50 + "\n")

    for model in predictive_models:
        top_genes_df = print_top_genes(results_NxN, model.__class__.__name__, 20)
        
        f.write(f"\nTop Genes for Model: {model.__class__.__name__}\n")
        f.write(top_genes_df.to_string(index=False))
        f.write("\n" + "="*50 + "\n")

print(f"Results written to {output_file}")

