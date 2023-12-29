import matplotlib.pyplot as plt
import seaborn as sns

def hazardous_distribution(data):
    ax = sns.countplot(x = "Hazardous", data = data)
    plt.title('Distribution of Hazardous Asteroids')
    plt.xlabel('Not Hazardous                                           Hazardous')
    plt.show()

def heatmap(data):
    plt.figure(figsize = (30,40))
    sns.heatmap(data.corr(),annot=True)
    plt.show()