import struct
import numpy as np
import numpy.typing as npt
from nn import NeuralNetwork
from functions import cel_prime

def parse_mnist(images_path: str, labels_path: str) \
    -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    with open(images_path, "rb") as f:
        _, n_items, _, _ = struct.unpack(">iiii", f.read(16))
        img_arr = np.frombuffer(f.read(), dtype=np.int8)
        img_matrix = img_arr.reshape(n_items, 28*28)

    with open(labels_path, "rb") as f:
        header = f.read(8)
        _, n_items = struct.unpack(">ii", header)
        digits = f.read(n_items)
        label_arr = np.frombuffer(digits, dtype=np.int8)

    return img_matrix, label_arr

def train(model: NeuralNetwork, images, labels) -> None:
    epochs = 2
    for epoch in range(epochs):
        print(f"Training epoch: {epoch}")
        for image, label in zip(images, labels):
            prediction = model.forward(image)
            expected_arr = np.array([1 if n == label else 0 for n in range(10)])
            model.backward(expected_arr, prediction)

def test(model: NeuralNetwork, images, labels) -> None:
    total_correct = 0
    total_incorrect = 0
    for image, label in zip(images, labels):
        prediction = np.argmax(model.forward(image))
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
    train(model, images, labels)

    test_images, test_labels = parse_mnist(
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte"
    )
    test(model, test_images, test_labels)

if __name__ == "__main__":
    main()
