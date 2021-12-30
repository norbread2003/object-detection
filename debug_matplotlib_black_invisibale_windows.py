import matplotlib.pyplot as plt

figure = {}
plot = {}
plot_handle = {}
center_color = {
    "C": "red",
    "D": "blue"
}

if __name__ == "__main__":
    for i in ["A", "B", "C", "D", "3d", "A_depth", "B_depth"]:
        if i == "3d":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig, ax = plt.subplots()
        figure[i] = fig
        plot[i] = ax

    plt.show()
