To save training time, all models used in the three different datasets are provided in the following path `/data/<dataset_name>/models` e.g., `/data/deception/models`. BERT parameters should be stored in the following path `data/<dataset_name>/bert_fine_tune`. Please download the folders from this link: https://tinyurl.com/bert-fine-tune-folder. Note that folders can be huge and may take time to download.

### 1. Generate top 10 features and their feature importance for `svm`, `svm_l1`, `xgb`, and `lstm`.
1. To save `svm`, `svm_l1`, `xgb`, and `lstm` features and their feature importance, run `save_combinations.py`.
    - Note that _only_ `save_combinations.py` uses the downloaded shap package instead of the local one. As such, before running `save_combinations.py`, remember to set package path to run the downloaded shap package. Otherwise, simply rename local `shap` folder to something else so that `save_combinations.py` does not read from the local package. If you renamed local shap folder, remember to revert to the original folder name after running `save_combinations.py` so other files will not be affected.
2. To save `lstm` attention weights, run `get_lstm_att_weights.py`.
3. To save `lstm` SHAP, run `python get_lstm_shap.py <dataset_name>`.

### 2. Generate top 10 features and their feature importance for `bert`.
1. Generate `tsv` files for `bert`:
    1. deception: run `python data_retrieval.py deception`
    2. yelp: run `python data_retrieval.py yelp`
    3. sst: run `python data_retrieval.py sst`
2. To save `bert` attention weights:
    1. deception: run `python bert_att_weight_retrieval.py --data_dir data/deception --bert_model data/deception/bert_fine_tune/ --task_name sst-2 --output_dir /data/temp_output_dir/deception/ --do_eval --max_seq_length 300 --eval_batch_size 1`
    2. yelp: run the above command, but replace `deception` with `yelp`, and change `max_seq_length` to `512`
    3. sst: run the above command, but replace `deception` with `sst`, and change `max_seq_length` to `128`
3. To save `bert` LIME:
    1. deception: run `python bert_lime.py --data_dir data/deception --bert_model data/deception/bert_fine_tune/ --task_name sst-2 --output_dir /data/temp_output_dir/deception/ --do_eval --max_seq_length 300 --eval_batch_size 1`
    2. yelp: run the above command, but replace `deception` with `yelp`, and change `max_seq_length` to `512`
    3. sst: run the above command, but replace `deception` with `sst`, and change `max_seq_length` to `128`
4. To save `bert` SHAP:
    1. deception: run `python bert_shap.py --data_dir data/deception --bert_model data/deception/bert_fine_tune/ --task_name sst-2 --output_dir /data/temp_output_dir/deception/ --do_eval --max_seq_length 300 --eval_batch_size 1`
    2. yelp: run the above command, but replace `deception` with `yelp`, and change `max_seq_length` to `512`
    3. sst: run the above command, but replace `deception` with `sst`, and change `max_seq_length` to `128`
5. Generate bert spans and white spans:
    1. deception: run `python tokenizer_alignment.py --data_dir data/deception --bert_model data/deception/bert_fine_tune --task_name sst-2 --output_dir /data/temp_output_dir/deception/ --do_eval --max_seq_length 300`
    2. yelp: run the above command, but replace `deception` with `yelp`, and change `max_seq_length` to `512`
    3. sst: run the above command, but replace `deception` with `sst`, and change `max_seq_length` to `128`
6. Align all `bert` features/tokens with correct weights, run `python get_bert.py`.
Note: to generate `bert` related feature and its feature importance, it is important to follow the above steps **in order**.

### 3. Recreate analysis plots.
1. To generate plots in the paper, refer to interactive notebook `main.ipynb`.

If met with any problems, please send an email to `vivian.lai@colorado.edu` and `jon.z.cai@colorado.edu`.