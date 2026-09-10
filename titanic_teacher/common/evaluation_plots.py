import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

def show_norm_conf_mx(y, pred):
    norm_conf_mx = confusion_matrix(y, pred, normalize="true")

    plt.figure(figsize=(7,5))

    sns.heatmap(norm_conf_mx, annot=True, cmap="coolwarm", linewidth=0.5)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

