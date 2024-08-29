import pandas as pd

df = pd.read_parquet('https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/test_gen-00000'
                     '-of-00001-3d4cd8309148a71f.parquet')

df_train=df.sample(frac=0.95,random_state=200)
df_eval=df.drop(df_train.index)

df_train.to_json("ultrachat_chunk_train.jsonl", orient="records", lines=True)
df_eval.to_json("ultrachat_chunk_eval.jsonl", orient="records", lines=True)