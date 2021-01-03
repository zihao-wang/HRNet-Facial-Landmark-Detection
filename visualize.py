import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

visfolder = "visualize"



def visualize_diff_tensor(input_folder, output_folder, label_filter=lambda s: int(s[-1]) % 5 == 0):
    os.makedirs(os.path.join(visfolder, output_folder), exist_ok=True)
    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('npy'):
            filetitle = os.path.basename(filename).split('.')[0]
            name, label = filetitle.split("@")
            diff = np.load(os.path.join(input_folder, filename))

            if label_filter(label) == False:
                continue

            print(filetitle)

            num_samples, num_points, coordinates = diff.shape
            for i in range(num_points):
                fig, ax = plt.subplots()
                ax.scatter(diff[:, i, 0], diff[:, i, 1])
                ax.set_title("{} @ {} point #{} diff scatter".format(name, label, i))
                ax.set_ylim([-5, 5])
                ax.set_xlim([-5, 5])

                os.makedirs(os.path.join(visfolder, output_folder, name, str(i)), exist_ok=True)
                fig.savefig(os.path.join(visfolder, output_folder, name, str(i), "{}.png".format(label)))

    return


if __name__ == "__main__":
    visualize_diff_tensor(input_folder='output/Dental/dental_landmark_all_R340x460',
                          output_folder='dental_landmark_all_R340x460')

