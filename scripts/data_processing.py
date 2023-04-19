# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
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
upstream = None

# This is a placeholder, leave it as None
product = None


# %%
# your code here...
import pandas as pd
import urllib
import re
import nltk
import gensim.downloader as gensim_api

# %%
nltk.download(['stopwords']) #,'wordnet','words','brown', 'webtext', 'abc', 'wordnet31'])
lst_stopwords = nltk.corpus.stopwords.words("english")
nlp = gensim_api.load("glove-twitter-200") #"glove-wiki-gigaword-300"

# %%
# download latest data
urllib.request.urlretrieve("https://openpolitics.au/openpol.csv", "../data/openpol.csv")

# %%
cols = pd.read_csv('../data/openpol.csv', nrows=0).columns.tolist()
dtypes = {c: 'str' for c in cols if c not in ['Date', 'OP Record ID', 'PARLIAMENT']}
raw = pd.read_csv('../data/openpol.csv', header=0, dtype=dtypes,parse_dates=['Date'])#.replace([pd.NA, 'Nil'], '',inplace=True)
raw.replace([pd.NA, 'Nil'], '',inplace=True)
raw['field'] = raw['Field 1'] + ' ' + raw['Field 2'] + ' ' + raw['Field 3']
raw.drop(columns = ['Field 1','Field 2','Field 3'], inplace=True)
raw

# %%
raw.dtypes


# %%
category_dict = {
        1: 'shareholdings',
        2: 'trusts',
        3: 'real estate',
        4: 'directorships',
        5: 'partnerships',
        6: 'liabilities',
        7: 'bonds and debentures',
        8: 'savings and insvestment accounts',
        9: 'other assets',
        10: 'other income',
        11: 'gifts',
        12: 'travel and hospitality',
        13: 'memberships/office holder or donor',
        14: 'other interests'
    }
def get_category(catnum, category_dict=category_dict):
    return category_dict.get(catnum, 'invalid category number')


uniqcats = raw['Interest category'].unique()
catvals  = [int(c.split('. ')[0]) for c in uniqcats]
raw['category_number'] = raw['Interest category'].replace({u: catvals[i] for i,u in enumerate(uniqcats)})
raw['category'] = raw.category_number.apply(get_category)
raw
# %%
exclude_strings = ['href', '<a', '/a>', 'https']
raw = raw[~raw.field.str.contains('|'.join(exclude_strings), regex=True)]
raw = raw[~raw.field.str.isdigit()]
raw

# %%
'''
Preprocess a string.
:parameter
    :param text: string - name of column containing text
    :param lst_stopwords: list - list of stopwords to remove
    :param flg_stemm: bool - whether stemming is to be applied
    :param flg_lemm: bool - whether lemmitisation is to be applied
:return
    cleaned text
'''
def utils_preprocess_text(text, flg_numm=True, flg_stemm=False, flg_lemm=True, lst_stopwords=None, flg_glve=True):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove numeric strings
    if flg_numm == True:
        lst_text = [word for word in lst_text if not word.isdigit()]

    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## keep only glove-twitter-200 understandable words
    if flg_glve == True:
        lst_text = [word for word in lst_text if word in nlp.key_to_index]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

# %%
raw['field_clean'] = raw.field.apply(lambda x: utils_preprocess_text(x, flg_stemm=False, flg_lemm=False, 
          lst_stopwords=lst_stopwords))

# %%
raw.to_pickle('../products/processed_data/openpol.pkl')


# %%
# raw.field[raw.field_clean.str.contains('jstw')]
# %%

# %%
