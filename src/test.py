from graphviz import Digraph
import torch
from torch.autograd import Variable
from torchviz import make_dot
import matplotlib.pyplot as plt
from PIL import Image
import io

x1 = Variable(torch.Tensor([3]), requires_grad=True)
x2 = Variable(torch.Tensor([5]), requires_grad=True)

a = torch.mul(x1, x2)
y1 = torch.log(a)
y2 = torch.sin(x2)
w = torch.mul(y1, y2)
w = w**4
dot = make_dot(w)

# Converti il grafo in un'immagine
img_data = dot.pipe(format='png')
img = Image.open(io.BytesIO(img_data))

# Mostra l'immagine
plt.imshow(img)
plt.axis('off')
plt.show()