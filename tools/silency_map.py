import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W)
     giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Wrap the input tensors in Variables
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y)
    saliency = None

    # Forward pass.
    scores = model(X_var)

    # Get the correct class computed scores.
    scores = scores.gather(1, y_var.view(-1, 1)).squeeze()

    # Backward pass, need to supply initial gradients
    # of same tensor shape as scores.
    scores.backward(torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0]))

    # Get gradient for image.
    saliency = X_var.grad.data

    # Convert from 3d to 1d.
    saliency = saliency.abs()
    saliency, i = torch.max(saliency, dim=1)
    saliency = saliency.squeeze()

    return saliency


def show_saliency_maps(X, y, model, class_names):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat(X, dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.numpy()
    N = X.shape[0]

    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


# show_saliency_maps(X, y)
