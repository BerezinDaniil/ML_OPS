from functools import lru_cache

import numpy as np
import pandas as pd
from tritonclient.http import (InferenceServerClient, InferInput,
                               InferRequestedOutput)
from tritonclient.utils import np_to_triton_dtype


@lru_cache
def get_client():
    return InferenceServerClient(url="0.0.0.0:8500")


def call_triton(data):
    triton_client = get_client()

    input_data = InferInput(
        name="INPUT", shape=data.shape, datatype=np_to_triton_dtype(data.dtype)
    )
    input_data.set_data_from_numpy(data, binary_data=True)

    query_response = triton_client.infer(
        "onnx-danodel",
        [input_data],
        outputs=[
            InferRequestedOutput("PROBABILITY", binary_data=True),
        ],
    )

    prob = query_response.as_numpy("PROBABILITY")[0]
    return prob


def main():
    test = pd.read_csv("data/test.csv").head(5)
    X_test = np.array(test.drop(columns="round_winner"), dtype=np.float32)
    # y_test = test["round_winner"].values
    y_pred = [call_triton(x) for x in X_test]
    print(y_pred)
    # assert (y_test == y_pred).all()


if __name__ == "__main__":
    main()
