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

def train_loss_plot(history):
    plt.figure(figsize=(20,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
