from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt

class Swissroll(Dataset):
    def __init__(self, tmin, tmax, N):
        # Generate a linearly spaced tensor t with N points between tmin and tmin + tmax
        t = tmin + torch.linspace(0, 1, N) * tmax
        # Create a tensor of 2D points in the shape of a Swiss roll
        self.vals = torch.stack([t * torch.cos(t) / tmax, t * torch.sin(t) / tmax]).T 

    def __len__(self):
        # Return the number of data points in the dataset
        return len(self.vals)

    def __getitem__(self, i):
        # Return the i-th data point from the dataset
        return self.vals[i]
    


class Single_Point(Dataset):
    """
    This dataset is for sake of performing sanity check on the model.
    To see whether the model is able to learn the mapping for a single point.
    """
    def __init__(self, N, point):
        self.points = torch.stack([point] * N)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        return self.points[i]


def plot_batch(batch):
    batch = batch.cpu().numpy()
    plt.scatter(batch[:,0], batch[:,1], marker='.')

    # Display the plot
    plt.show()
    
    # Close the plot to free up memory
    plt.close()


#dataset  = Swissroll(np.pi/2, 5*np.pi, 100)

#loader = DataLoader(dataset, batch_size=30)

#plot_batch(list(iter(loader))[3])

dataset = Single_Point(1000, torch.tensor([0.5, 0.5]))

loader = DataLoader(dataset, batch_size=50)

plot_batch(next(iter(loader)))