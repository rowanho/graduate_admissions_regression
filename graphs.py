import matplotlib.pyplot as plt

#input - tensorflow history object
def plot_history(title,histories):
    plt.figure(figsize=(20,10))
    for count,history in enumerate(histories):
        val = plt.plot(history.epoch,history.history['val_mean_squared_error'],'--',label="validation results " + str(count))
        plt.plot(history.epoch,history.history['mean_squared_error'],label="loss " + str(count),color = val[0].get_color())
    plt.xlabel('epoch')
    plt.ylabel('Mean squared error')
    plt.legend()
    plt.xlim([0,max(history.epoch)])
    plt.title(title)
    plt.show()
