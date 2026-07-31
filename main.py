import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import struct

from nn import NeuralNetwork

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
        indices = rng.permutation(len(images))
        for i in range(len(images) // batch_size):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("path", default="model.npz", help="path to load/save trained model")
    parser.add_argument("--train", action="store_true", help="Train and save model")
    parser.add_argument("--load", action="store_true", help="Load model from file")
    parser.add_argument("--test", action="store_true", help="Test the loaded/trained model")
    parser.add_argument("--ui", action="store_true", help="Test the model with a drawing UI")
    parser.add_argument("--plot", action="store_true", help="Plot loss function from training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    args = parser.parse_args()

    model = NeuralNetwork()

    if args.load:
        model.load(args.path)

    if args.train:
        images, labels = parse_mnist(
            "train-images.idx3-ubyte",
            "train-labels.idx1-ubyte"
        )
        train(model, images, labels, args.epochs, plot_loss=False)
        model.save(args.path)

    if args.test:
        test_images, test_labels = parse_mnist(
            "t10k-images.idx3-ubyte",
            "t10k-labels.idx1-ubyte"
        )
        test(model, test_images, test_labels)

    if args.ui:
        ...

if __name__ == "__main__":
    main()
