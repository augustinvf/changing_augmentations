import sys
import torch

chemin = str(sys.argv[1])
matrix = torch.load(chemin)

print(matrix)