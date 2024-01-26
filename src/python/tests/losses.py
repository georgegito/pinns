import re
import matplotlib.pyplot as plt

def parse_losses(input_text):
    pattern = r"Epoch: (\d+).*?Total Loss: ([\d.]+).*?" \
              r"PDE Loss - Navier Stoker: ([\d.]+), PDE Loss - Poisson: ([\d.]+), " \
              r"BC Inlet Loss: ([\d.]+), BC Outlet Loss: ([\d.]+), " \
              r"BC Left Loss: ([\d.]+), BC Right Loss: ([\d.]+), " \
              r"BC Down Loss: ([\d.]+), BC Up Loss: ([\d.]+), " \
              r"Surface Loss: ([\d.]+), Real-Data Loss: ([\d.]+)"
    matches = re.findall(pattern, input_text, re.DOTALL)
    losses = []
    for match in matches:
        losses.append({key: float(value) for key, value in zip(
            ["Epoch", "PDE Loss - Navier Stoker", "PDE Loss - Poisson", 
             "BC Inlet Loss", "BC Outlet Loss", "BC Left Loss", "BC Right Loss", 
             "BC Down Loss", "BC Up Loss", "Surface Loss", "Real-Data Loss"], match)})
    return losses

def plot_losses(losses):
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['lines.markersize'] = 1

    epochs = [loss["Epoch"] for loss in losses]
    for key in losses[0].keys():
        if key != "Epoch":
            plt.plot(epochs, [loss[key] for loss in losses], marker='o', label=key)
    plt.title("Training Losses over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Replace 'path_to_file.txt' with your file path
file_path = '/Users/ggito/Desktop/out4/out/v3.txt'

with open(file_path, 'r') as file:
    input_text = file.read()

losses = parse_losses(input_text)
plot_losses(losses)