# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Legendary Pokemon Classification

# * dataset to use
#     * [kaggle Pokemon with stats](https://www.kaggle.com/abcsds/pokemon)

# ### 1. EDA

# **@libray import**

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

# **@load dataset**

# +
import os
csv_path = os.getcwd()+ '/Pokemon.csv'
original_data = pd.read_csv(csv_path)

'''
In the future, It will copy datasets because I will do various tasks in datasets
while dealing with data.
'''
# # copy pokemon dataset
pokemon = original_data.copy()
print(pokemon.shape)
pokemon.head()
# -

# : Datasets consist of 800 rows and 13 coulmns.
# <br/>
# In other words, there are 800 Pokemons and 13 features that explain each Pokemon.

# legendary pokemon dataset
legendary = pokemon[pokemon["Legendary"] == True]
print(legendary.shape)
legendary.head(2)

# originary pokemon dataset
ordinary = pokemon[pokemon["Legendary"] == False]
print(ordinary.shape)
ordinary.head(2)

# : Need to find the legendary Pokemon, <br/>
# <br/>
# "Legendary" == True values with Pokemon are stored in the legendary variable. <br/>
# <br/>
# "Legendary" == False values with Pokemon are stored in the ordinary variable. 

# **@Check missing value**

pokemon.isnull().sum()

print(len(pokemon.columns))
pokemon.columns

# : Type2(str): Second attribute; For Pokemon with only one attribute, Type 2 has NaN(missing value).

# +
# Double check with set()
len(set(pokemon["#"]))

# The id value of "#" isn't unique.
# -

pokemon[pokemon["#"]== 6]

len(set(pokemon["Name"]))
# The name is unique.

# check Type1 & Type2
pokemon.loc[[6, 10]]

# Check how many types of each attribute are in.
len(list(set(pokemon["Type 1"]))), len(list(set(pokemon["Type 2"])))

# The differece set of Type 2 and Type 1 is NaN
set(pokemon["Type 2"]) - set(pokemon["Type 1"])

# Save all types of Pokemon in types.
types = list(set(pokemon["Type 1"]))
print(len(types))
print(types)

pokemon["Type 2"].isna().sum()

# **@Type 1 data distribution plot**

# +
plt.figure(figsize=(15, 12))

plt.subplot(211)
sns.countplot(data=ordinary, x="Type 1", order=types).set_xlabel('')
plt.title("[Ordinary Pokemons]")

plt.subplot(212)
sns.countplot(data=legendary, x="Type 1", order=types).set_xlabel('')
plt.title("[Legendary Pokemons]")

plt.show()
# -

# Pivot table showing the ratio of Legendary by Type1
pd.pivot_table(pokemon, index="Type 1", values="Legendary").sort_values(by=["Legendary"], ascending=False)

# **@Type 2 data distribution plot**

# +
# NaN is automatically excluded when drawing a Countplot.
plt.figure(figsize=(12, 10))  

plt.subplot(211)
sns.countplot(data=ordinary, x="Type 2", order=types).set_xlabel('')
plt.title("[Ordinary Pokemons]")

plt.subplot(212)
sns.countplot(data=legendary, x="Type 2", order=types).set_xlabel('')
plt.title("[Legendary Pokemons]")

plt.show()
# -

# Pivot table showing the ratio of Legendary by Type 2.
pd.pivot_table(pokemon, index="Type 2", values="Legendary").sort_values(by=["Legendary"], ascending=False)

# **@Total of all stats**

stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
stats

# 0 index total of all stats validation.
print("#0 pokemon: ", pokemon.loc[0, "Name"])
print("total: ", int(pokemon.loc[0, "Total"]))
print("stats: ", list(pokemon.loc[0, stats]))
print("sum of all stats: ", sum(list(pokemon.loc[0, stats])))

# : For the first Pokemon, the total value is 318.

# Check Pokemon with Total Values and All Stats
sum(pokemon['Total'].values == pokemon[stats].values.sum(axis=1))
# Set to axis = 1, because it must be added in the transverse direction.

# **@Distribution plot according to Total value**

# +
fig, ax = plt.subplots()
fig.set_size_inches(16, 8)

sns.scatterplot(data=pokemon, x="Type 1", y="Total", hue="Legendary")
plt.show()
# -

# : Legendary Pokemon can be seen to have a high total stat value.

# **@Detail stats distribution**

# +
figure, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows=3, ncols=2)
figure.set_size_inches(22,18)

sns.scatterplot(data=pokemon, y="Total", x="HP", hue="Legendary", ax=ax1)
sns.scatterplot(data=pokemon, y="Total", x="Attack", hue="Legendary", ax=ax2)
sns.scatterplot(data=pokemon, y="Total", x="Defense", hue="Legendary", ax=ax3)
sns.scatterplot(data=pokemon, y="Total", x="Sp. Atk", hue="Legendary", ax=ax4)
sns.scatterplot(data=pokemon, y="Total", x="Sp. Def", hue="Legendary", ax=ax5)
sns.scatterplot(data=pokemon, y="Total", x="Speed", hue="Legendary", ax=ax6)
# -

# * HP, Defense, Sp. Def
#     * Legendary Pokemon has a high stat, but in these three, there are some Pokemons whose ordinary Pokemon is particularly than the legendary Pokemon.
#     * However, the total value of the Pokémon is not particularly high, so only certain stats are considered to be specially high, that is, Pokémon specialized in specific attributes.
# * Attack, Sp. Atk, Speed
#     * These three stats are almost proportional to Total.
#     * Legendary Pokemon takes the maximum of each stat.

# **@Generation of Pokemon**
#

# +
# Number of Pokemons in each generation. plt
plt.figure(figsize=(12, 10))   

plt.subplot(211)
sns.countplot(data=ordinary, x="Generation").set_xlabel('')
plt.title("[All Pkemons]")
plt.subplot(212)
sns.countplot(data=legendary, x="Generation").set_xlabel('')
plt.title("[Legendary Pkemons]")
plt.show()
# -

# **@Feature segmentation**

# +
# total of all stat values in Legendary Pokemons.
fig, ax = plt.subplots()
fig.set_size_inches(8, 4)

sns.scatterplot(data=legendary, y="Type 1", x="Total")
plt.show()
# -

print(sorted(list(set(legendary["Total"]))))

# : 9 Total Values.

# +
fig, ax = plt.subplots()
fig.set_size_inches(8, 4)

sns.countplot(data=legendary, x="Total")
plt.show()

# +
round(65 / 9, 2)

# About 7.22 have the same total stat value
# -

# total of all stat values in Ordinary Pokemons.
len(sorted(list(set(ordinary["Total"]))))

# +
round(735 / 195, 2)

# About 3.77 have the same total stat value
# -

# * Whether a Pokemon's total stat value is included in the legendary Pokemon's set of values is influenced by determining that it is a legendary Pokemon.
# * Among the total values of legendary Pokemon, there is a total value that ordinary Pokemon does not have.
#     * Total values are important columns for predicting whether or not they are legendary.

n1, n2, n3, n4, n5 = legendary[3:6], legendary[14:24], legendary[25:29], legendary[46:50], legendary[52:57]
names = pd.concat([n1, n2, n3, n4, n5]).reset_index(drop=True)
names

# * name feature of legendary pokemons
#     * Names tend to be similar.
#     * If there are several legends of Pokemon with forme in.

formes = names[13:23]
formes

# **@Name length comparision**

legendary["name_count"] = legendary["Name"].apply(lambda i: len(i))    
legendary.head()

ordinary["name_count"] = ordinary["Name"].apply(lambda i: len(i))    
ordinary.head()

# +
plt.figure(figsize=(16, 10))

plt.subplot(211)
sns.countplot(data=legendary, x="name_count").set_xlabel('')
plt.title("Legendary")

plt.subplot(212)
sns.countplot(data=ordinary, x="name_count").set_xlabel('')
plt.title("Ordinary")

plt.show()
# -

# Legendary pokemon name length is 10 or more probability. 
print(round(len(legendary[legendary["name_count"] > 9]) / len(legendary) * 100, 2), "%")

# Ordinary pokemon name length is 10 or more probability.
print(round(len(ordinary[ordinary["name_count"] > 9]) / len(ordinary) * 100, 2), "%")

# * If "Latios" is a legendary Pokemon, then "%%%Latios" is also a legendary Pokemon.
# * At least there is a high frequency of names in the legendary Pokemon.
# * Legendary Pokemon is likely to have a long name.

# **@Create a categorical column: Agter generating the name_count column, whether the length exceeds 10 or not.**

pokemon["name_count"] = pokemon["Name"].apply(lambda i : len(i))
pokemon.head()

pokemon["long_name"] = pokemon["name_count"] >= 10
pokemon.head()

# **@Token extractions often used in names**

# * Name type of Pokemon
#     * One word ex. Venusaur
#     * Two words, the preceding word has two capital words and is divided into two parts based on capital words ex. BenusaurMegaVenusaur
#     * Name is two words, and at the back, X, Y, the gender is displayed ex. CharizardMega Charizard X
#     * Includes characters that are not alphabets ex. Zygarde50% Forme

# * pre-process the name contains a non-alphathic words.

# +
# Make a column that dosen't have a spacing for checking alphabet.

pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())
pokemon.head()
# -

print(pokemon[pokemon["name_isalpha"] == False].shape)
pokemon[pokemon["name_isalpha"] == False]

# +
pokemon = pokemon.replace(to_replace="Nidoran♀", value="Nidoran X")
pokemon = pokemon.replace(to_replace="Nidoran♂", value="Nidoran Y")
pokemon = pokemon.replace(to_replace="Farfetch'd", value="Farfetchd")
pokemon = pokemon.replace(to_replace="Mr. Mime", value="Mr Mime")
pokemon = pokemon.replace(to_replace="Porygon2", value="Porygon")
pokemon = pokemon.replace(to_replace="Ho-oh", value="Ho Oh")
pokemon = pokemon.replace(to_replace="Mime Jr.", value="Mime Jr")
pokemon = pokemon.replace(to_replace="Porygon-Z", value="Porygon Z")
pokemon = pokemon.replace(to_replace="Zygarde50% Forme", value="Zygarde Forme")

pokemon.loc[[34, 37, 90, 131, 252, 270, 487, 525, 794]]
# -

pokemon["Name_nospace"] = pokemon["Name"].apply(lambda i: i.replace(" ", ""))
pokemon["name_isalpha"] = pokemon["Name_nospace"].apply(lambda i: i.isalpha())
pokemon[pokemon["name_isalpha"] == False]

# +
import re

# Separate and tokenize names based on spacing and capitalization.
def tokenize(name):
    name_split = name.split(" ")
    
    tokens = []
    for part_name in name_split:
        a = re.findall('[A-Z][a-z]*', part_name)
        tokens.extend(a)
        
    return np.array(tokens)

# tokenize legendary pokemons.
all_tokens = list(legendary["Name"].apply(tokenize).values)

token_set = []
for token in all_tokens:
    token_set.extend(token)

print(len(set(token_set)))
print(token_set)

# +
from collections import Counter

most_common = Counter(token_set).most_common(10)
most_common

# +
for token, _ in most_common:
    # pokemon[token] = ... 형식으로 사용하면 뒤에서 warning이 발생
    pokemon[f"{token}"] = pokemon["Name"].str.contains(token)

pokemon.head(10)
# -

# : If the column value of the token is True, the Pokemon is likely to be a legendary Pokemon.

# **@Type 1 & 2 categorical data preprocessing**

# * All 18 Types are one-hot encoding.
# * Pokemon with two attrivutes has a value of 1 in the right place for two types.

# +
# pandas one-hot encoding.
print(types)

for t in types:
    pokemon[t] = (pokemon["Type 1"] == t) | (pokemon["Type 2"] == t)
    
pokemon[[["Type 1", "Type 2"] + types][0]].head()
# -

# **@Baselines that make them the most basic data.**

print(original_data.shape)
original_data.head()

original_data.columns

features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']

target = 'Legendary'

X = original_data[features]
print(X.shape)
X.head()

y = original_data[target]
print(y.shape)
y.head()

# +
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# -

# ### 2. Model Learning

# +
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=25)
model

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# -

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

# * TN (True Negative): A case where a rightly judged Negative, or General Pokemon, is properly judged as a General Pokemon.
# * FP (False Positive): a wrongly judged Pokemon, or General Pokemon, is a case of misjudgment as a legendary Pokemon.
# * FN (False Negative): a wrongly judged Negative, or Legendary Pokemon It is a case of misjudgment as a general Pokémon.
# * TP (True Positive): A case of misjudgment of the rightly judged Positive, or Legendary Pokémon, as a legend's Pokémon.

print(152 / 160 * 100, "%")

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

print(len(pokemon.columns))
print(pokemon.columns)

# * "#": Data corresponding to ID, except because it is not a feature with special meaning other than the meaning of index.
# * "Name": String data, replaced by "name_count" and "long_name" through preprocessing, and 15 token columns.
# * "name_nospace", "name_isalpha": columns needed for preprocessing, not necessary for classification analysis.
# * "Type 1" & "Type 2": The attributes were processed with one-hot encoding.
# * "Legendary": This column is target data, so it does not put it in the "X" data the model learns, but uses it as "y" data.

# +
features = ['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 
            'name_count', 'long_name', 'Forme', 'Mega', 'Mewtwo', 'Kyurem', 'Deoxys', 'Hoopa', 
            'Latias', 'Latios', 'Kyogre', 'Groudon', 'Poison', 'Water', 'Steel', 'Grass', 
            'Bug', 'Normal', 'Fire', 'Fighting', 'Electric', 'Psychic', 'Ghost', 'Ice', 
            'Rock', 'Dark', 'Flying', 'Ground', 'Dragon', 'Fairy']

len(features)
# -

target = "Legendary"
target

X = pokemon[features]
print(X.shape)
X.head()

y = pokemon[target]
print(y.shape)
y.head()

# +
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
# -

model = DecisionTreeClassifier(random_state=25)
model

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

confusion_matrix(y_test, y_pred)

print(classification_report(y_test, y_pred))


