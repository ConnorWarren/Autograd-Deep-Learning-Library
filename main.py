import struct
import numpy as np
import numpy.typing as npt
import matplotlib
import matplotlib.pyplot as plt

from nn import NeuralNetwork
from functions import cross_entropy_loss

def parse_mnist(images_path: str, labels_path: str) \
    -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    with open(images_path, "rb") as f:
        _, n_items, _, _ = struct.unpack(">iiii", f.read(16))
        img_arr = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float64) / 255
        img_matrix = img_arr.reshape(n_items, 28*28)

    with open(labels_path, "rb") as f:
        header = f.read(8)
        _, n_items = struct.unpack(">ii", header)
        digits = f.read(n_items)
        label_arr = np.frombuffer(digits, dtype=np.uint8)

    return img_matrix, label_arr

def train(model: NeuralNetwork, images, labels, epochs: int, plot_loss=True) -> None:
    batch_size = len(images) // 100
    rng = np.random.default_rng()
    for epoch in range(epochs):
        print(f"Training epoch: {epoch}")
        for i in range(len(images) // batch_size):
            indices = rng.permutation(len(images))
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            raw_predictions = model.forward(images[batch_indices])
            expected = np.zeros((batch_size, 10))
            expected[range(batch_size), labels[batch_indices]] = 1
            model.backward(raw_predictions, expected)

def test(model: NeuralNetwork, images, labels) -> None:
    total_correct = 0
    total_incorrect = 0
    for image, label in zip(images, labels):
        prediction = np.argmax(model.forward(image[np.newaxis, :]))
        if prediction == label:
            total_correct += 1
        else:
            total_incorrect += 1

    print(f"Accuracy: {total_correct/(total_correct + total_incorrect) * 100}%")

def main() -> None:    
    model = NeuralNetwork() 
    images, labels = parse_mnist(
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte"
    )
    train(model, images, labels, 50, plot_loss=False)

    test_images, test_labels = parse_mnist(
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte"
    )
    test(model, test_images, test_labels)

if __name__ == "__main__":
    main()
