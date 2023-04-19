# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Add description here
#
# *Note:* You can open this file as a notebook (JupyterLab: right-click on it in the side bar -> Open With -> Notebook)


# %%
# Uncomment the next two lines to enable auto reloading for imported modules
# # %load_ext autoreload
# # %autoreload 2
# For more info, see:
# https://docs.ploomber.io/en/latest/user-guide/faq_index.html#auto-reloading-code-in-jupyter

# %% tags=["parameters"]
# If this task has dependencies, list them them here
# (e.g. upstream = ['some_task']), otherwise leave as None.
upstream = ['data_processing']

# This is a placeholder, leave it as None
product = None


# %%
# your code here...
import pandas as pd
import gensim.downloader as gensim_api
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
import pickle

# %%
with open('../products/processed_data/openpol.pkl','rb') as f:
    clean = pickle.load(f)

# %%
lst_tokens_by_category = clean.groupby('category_number').field_clean.apply(lambda x: ' '.join(x).split())

# %%
token_counts_by_category = [pd.DataFrame(sorted(Counter(lst).items(), key=lambda x: x[1], reverse=True))  for lst in lst_tokens_by_category]
# %%
def create_dic_clusters(token_counts_by_category, method='greater_than_1pc_of_maxfreq',topn=100):
    '''
    '''
    assert method in ['greater_than_1pc_of_maxfreq', 'topn_most_freq']
    if method == 'greater_than_1pc_of_maxfreq':
        topn_words_by_category = [token_counts_df.loc[token_counts_df[1].apply(lambda x: x >= int(token_counts_df[1][0]/100.) and x > 1),0] for token_counts_df in token_counts_by_category]    
    else:
        topn_words_by_category = [token_counts_df.loc[:(topn-1),0] for token_counts_df in token_counts_by_category]
    categories = category_dict.values()
    return {c: topn_words_by_category[i] for i,c in enumerate(categories)}

dic_clusters = create_dic_clusters(token_counts_by_category, method='topn_most_freq', topn=10)

# %%
## word embedding
nlp = gensim_api.load("glove-twitter-200")
tot_words = [word for lst in dic_clusters.values() for word in lst]
X = nlp[tot_words]
# %%
## pca
pca = manifold.TSNE(perplexity=40, n_components=2, init='pca')
X = pca.fit_transform(X)
# %%
## create dtf
dtf = pd.DataFrame()
for k,v in dic_clusters.items():
    size = len(dtf) + len(v)
    dtf_group = pd.DataFrame(X[len(dtf):size], columns=["x","y"], 
                             index=v)
    dtf_group["cluster"] = k
    dtf = dtf.append(dtf_group)
# %%
## plot
fig, ax = plt.subplots(figsize=(15,10))
sns.scatterplot(data=dtf, x="x", y="y", hue="cluster", ax=ax)
ax.legend().texts[0].set_text(None)
ax.set(xlabel=None, ylabel=None, xticks=[], xticklabels=[], 
       yticks=[], yticklabels=[])
for i in range(len(dtf)):
    ax.annotate(dtf.index[i], 
               xy=(dtf["x"].iloc[i],dtf["y"].iloc[i]), 
               xytext=(5,2), textcoords='offset points', 
               ha='right', va='bottom')
# %%
