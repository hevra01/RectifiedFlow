import math
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
    
class linear_line(Dataset):
    def __init__(self, N, start, end):
        # Generate a linearly spaced tensor t with N points between start and end
        t = torch.linspace(start, end, N)
        # Create a tensor of 2D points forming a linear line
        self.points = torch.stack([t, t]).T

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        return self.points[i]


class FigureEight(Dataset):
    def __init__(self, tmin, tmax, N):
        # Generate a linearly spaced tensor t with N points between tmin and tmax
        t = torch.linspace(tmin, tmax, N)
        # Parametric equations for a figure-eight (lemniscate)
        self.vals = torch.stack([torch.sin(t), torch.sin(2 * t)]).T  # 2D figure-eight

    def __len__(self):
        # Return the number of data points in the dataset
        return len(self.vals)

    def __getitem__(self, i):
        # Return the i-th data point from the dataset
        return self.vals[i]

# Example usage:
dataset = FigureEight(0, 2 * math.pi, 1000)  # Creates a figure-eight with 1000 points

def plot_batch(batch):
    batch = batch.cpu().numpy()
    plt.scatter(batch[:,0], batch[:,1], marker='.')

    # Display the plot
    plt.show()
    
    # Close the plot to free up memory
    plt.close()


#dataset  = Swissroll(np.pi/2, 5*np.pi, 1000)

loader = DataLoader(dataset, batch_size=1000)

#plot_batch(list(iter(loader))[3])

#dataset = linear_line(1000, 0, 50)

#loader = DataLoader(dataset, batch_size=1000)

plot_batch(next(iter(loader)))