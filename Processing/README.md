This folder has the code for performing data preprocessing based on data type in `Data` folder. This includes converting mutations into raw mutation data, obtaining variant annotations.

### Order of files
1. `3_train_test_eda_mutations_processing.ipynb` ensures all samples have atleast 1 mutation in the 324 genes
2. `4_get_annotated_mutation_matrices-vocab.ipynb` creates 7776 dimensional vectors used by methods other than PREDICT-AI and GANDALF.
3. `5_annotated_mutation_processing.ipynb` combines train-test splits with 7776 dimensional processed data from previous steps.
4. `5_predict-ai-data-processing.ipynb` combines train-test splits with tokenized mutations, as used by PREDICT-AI and GANDALF.
5. `6_gandalf_data_prep.ipynb` runs the input mutations through the pretrained PREDICT-AI transformer and saves it for later use.
