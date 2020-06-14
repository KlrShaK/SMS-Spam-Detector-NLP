import matplotlib.pyplot as plt

def plot_accuracy(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Testing Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Training and validation Accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and validation Loss')
    plt.legend()
    plt.show()  # Todo To Show the Graphs