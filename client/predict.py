import os
import sys

import torch
from data import load_data
from model import load_parameters

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(dir_path))


def predict(in_model_path, out_artifact_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_artifact_path: The path to save the predict output to.
    :type out_artifact_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Load data
    _, test_loader = load_data(data_path)

    # Load model
    model = load_parameters(in_model_path)
    model.eval()

    predictions = []
    # Predict
    with torch.no_grad():
        for inputs, labels in test_loader:
            output_raw = model(inputs)
            predictions.extend(output_raw)

    # Save prediction to file/artifact, the artifact will be
    # uploaded to the object store by the client
    torch.save(predictions, out_artifact_path)


if __name__ == "__main__":
    predict(sys.argv[1], sys.argv[2])
