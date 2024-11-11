import numpy as np

if __name__ == "__main__":
    initial_model = np.full((17, 7, 4), 5200)

    np.save("initial_model_5200.npy", initial_model.T)

    print(initial_model.T[0])