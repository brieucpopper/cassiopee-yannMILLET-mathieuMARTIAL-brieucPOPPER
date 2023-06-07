import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from stop_words import get_stop_words
from matplotlib import pyplot as plt
import numpy as np

#see the readme for an explanation of the following code
stop_words = get_stop_words('french')
stop_words += ['libé','joannsfar','cool','www','oeil','jour','bien','motion','design','bobika','linkinbio','profil','mouillait','concours', 'jeu', 'enduro', 'gagner', 'saviez','https', 'humanite', 'fr', 'semaine', 'photo', 'après', '2022',
               'santeplusmag', 'plus', 'lien', 'bio', 'cliquant','graphisme','récap','débat','lire','retrouvez','français','joannsfar'
               'endurocross', 'enduromagazine', 'endurolifestyle','aria','leseveilleurs','gettroffical','cocoboer','édition','libération'
               'joyeux', 'happy', 'tenter', 'ami', 'chance', 'comptes','terreurgraphique','end','week','valeurs','actuelles','story','dessin','dessindepresse','ans','article']


def extract_keywords_topic_analysis(data, num_topics, num_keywords):
    vectorizer = TfidfVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(data)

    nmf = NMF(n_components=num_topics,max_iter=1000)
    nmf.fit(X)

    keywords_per_topic = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(nmf.components_):
        top_keywords_idx = topic.argsort()[:-num_keywords - 1:-1]
        top_keywords = [feature_names[idx] for idx in top_keywords_idx]
        keywords_per_topic.append(top_keywords)

    return nmf.transform(X), keywords_per_topic


# Chargement des données depuis le fichier CSV dans un DataFrame
csv_file = '/home/cassiopee/mathieu/finalDb.csv'

df = pd.read_csv(csv_file, header=0)
#print the keys
print(df.keys())
df['engagement_score'] = df['nb_likes'] + df['nb_post_comments']*3
# Paramètres pour l'analyse des sujets
text_column = 'content'
num_topics = 20
num_keywords = 7

# Extraction des mots-clés des topics
X, keywords_per_topic = extract_keywords_topic_analysis(df[text_column], num_topics, num_keywords)

# Attribution des topics aux textes
df['topic'] = X.argmax(axis=1)

# Calcul de la moyenne de l'engagement_score par topic
engagement_mean_by_topic = df.groupby('topic')['engagement_score'].mean()
engagement_mean_by_topic.sort_values()

# Récupération des i premiers mots-clés de chaque topic
i = 7
top_keywords = [", ".join(keywords[:i]) for keywords in keywords_per_topic]

topicCount=0
for k in top_keywords:
    print(str(topicCount)+k)
    topicCount+=1
i = 3
top_keywords = [", ".join(keywords[:i]) for keywords in keywords_per_topic]
# Comptage du nombre d'occurrences de chaque topic
topic_counts = df['topic'].value_counts()

# Création de la palette de couleurs pour les barres
color_palette = plt.get_cmap('tab20')

# Tri des données par engagement moyen
engagement_mean_sorted, top_keywords_sorted = zip(*sorted(zip(engagement_mean_by_topic, top_keywords)))

# Création de la palette de couleurs pour les barres
color_palette = plt.get_cmap('tab20')

# Tracé du graphe engagement
fig, ax = plt.subplots()
bar_colors = [color_palette(i) for i in np.linspace(0, 1, num_topics)]
bar_positions = range(num_topics)
ax.barh(bar_positions, engagement_mean_by_topic, color=bar_colors)
ax.set_yticks(bar_positions)
ax.set_yticklabels(top_keywords)
ax.set_xlabel('Engagement moyen')
ax.set_ylabel('Topic')
ax.set_title('Engagement moyen par topic')
plt.tight_layout()
#save the plot
fig.savefig('/home/cassiopee/mathieu/engagement_20top_3kw.png')

# Tracé du graphe occurences
fig, ax = plt.subplots()
bar_colors = [color_palette(i) for i in np.linspace(0, 1, num_topics)]
bar_positions = range(num_topics)
ax.barh(bar_positions, topic_counts, color=bar_colors)
ax.set_yticks(bar_positions)
ax.set_yticklabels(top_keywords)
ax.set_xlabel('Nombre d\'occurences')
ax.set_ylabel('Topic')
ax.set_title('Nombre d\'occurrence pour chaque topic')
plt.tight_layout()
#save
fig.savefig('/home/cassiopee/mathieu/occurences_20top_3kw.png')

df.to_csv('topic.csv', sep=';')
