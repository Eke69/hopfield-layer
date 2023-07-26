import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.model_selection import ParameterSampler, train_test_split

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST

from hflayers import Hopfield
from .transformer import HopfieldTransformerEncoder, HopfieldEncoderLayer

np.random.seed(0)
torch.manual_seed(0)

def patchify(images, num_patches):
    n, num_channels, height, width = images.shape

    assert height == width, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, num_patches ** 2, height * width * num_channels // num_patches ** 2)
    patch_size = height // num_patches

    for idx, image in enumerate(images):
        for i in range(num_patches):
            for j in range(num_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * num_patches + j] = patch.flatten()
    return patches

class ViT(nn.Module):
    def __init__(self, chw, num_patches=7, num_blocks=5, hidden_dim=8, num_heads=2, out_d=10):
        # Super constructor
        super(ViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( Channels , Height , Width )
        self.num_patches = num_patches
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Input and patches sizes
        assert chw[1] % num_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % num_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / num_patches, chw[2] / num_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_dim)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(num_patches ** 2 + 1, hidden_dim), persistent=False)

        input_size = self.hidden_dim  # Set input size to hidden_dim
        hidden_size = self.hidden_dim
        update_steps_max = 5
        scaling = 0.2

        encoder_layer = HopfieldEncoderLayer(Hopfield(input_size=input_size, hidden_size=hidden_size, update_steps_max=update_steps_max, scaling=scaling))
        self.blocks = HopfieldTransformerEncoder(encoder_layer, self.num_blocks)
        
        # 4) Transformer encoder blocks
        ## self.blocks = nn.ModuleList([MyViTBlock(hidden_dim, num_heads) for _ in range(num_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.num_patches).to(self.positional_embeddings.device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        hopfield_states = []
        out = self.blocks(out)
        hopfield_states.append(out.detach().cpu())
            
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out), hopfield_states, out # Map to output dimension, output category distribution, hopfield_states and representation
    
class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, image_shape):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.image_shape = image_shape

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(-1, *self.image_shape)
        return x
    
class ConvDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, image_shape):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, hidden_dim, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.image_shape = image_shape

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = x.view(-1, *self.image_shape)
        return x

class ViT_Autoencoder(nn.Module):
    def __init__(self, vit_model, decoder):
        super(ViT_Autoencoder, self).__init__()
        self.vit_model = vit_model
        self.decoder = decoder

    def forward(self, x):
        x = self.vit_model(x)
        x = self.decoder(x)
        return x

    
def get_positional_embeddings(sequence_length, dim):
    result = torch.ones(sequence_length, dim)
    for i in range(sequence_length):
        for j in range(dim):
            result[i][j] = np.sin(i / (10000 ** (j / dim))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / dim)))
    return result

def save_model(model, optimizer, epoch, hyperparameters, path):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'hyperparameters': hyperparameters
    }
    torch.save(state, path)

def load_model(model, optimizer, path):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    return model, optimizer, state['epoch'], state['hyperparameters']

def get_model(params, selected_model=None):
    if selected_model is not None:
        # Load saved model
        model, optimizer, epoch, hyperparameters = load_model(model, optimizer, selected_model)
        st.write(f"Loaded model from {selected_model}")
    else:
        model = ViT((1, 28, 28), num_patches=params['num_patches'], num_blocks=params['num_blocks'], hidden_dim=params['hidden_dim'], num_heads=params['num_heads'], out_d=params['out_d'])
        optimizer = Adam(model.parameters(), lr=params['lr'])
        st.write("Created new model")
    return model, optimizer

# def get_ae_model(params, selected_model=None):
#     if selected_model is not None:
#         model, optimizer, epoch, hyperparameters = load_model(model, optimizer, selected_model)
#         st.write(f"Loaded model from {selected_model}")
#     else:
#         model = ViT_Autoencoder()

def load_data():
    transform = ToTensor()
    train_set = MNIST(root='./../datasets', train=True, download=True, transform=transform)
    test_set = MNIST(root='./../datasets', train=False, download=True, transform=transform)
    train_set, val_set = train_test_split(train_set, test_size=0.2)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=128)
    return train_loader, test_loader, val_loader

def train_model(model, optimizer, train_loader, val_loader, device, N_EPOCHS, ae_model=None, auto_encoder=False):
    criterion = CrossEntropyLoss()
    criterion_ae = nn.MSELoss()
    progress_bar = st.progress(0)
    # Create a dataframe to store the loss and accuracy
    data = pd.DataFrame({
        'Validation Loss': [],
        'Validation Accuracy': []
    })
    # Create a line chart with the initial data
    chart = st.line_chart(data)
   
    for epoch in trange(N_EPOCHS, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            if isinstance(model, ViT_Autoencoder):
                output = model(x)
                loss = criterion_ae(output, x)
            else:
                y_hat, _, output = model(x)
                loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.progress((epoch + 1) / N_EPOCHS)
            val_loss, val_accuracy = validate_model(model, val_loader, device)
            # Update the data
            data = pd.concat([data, pd.DataFrame({
            'Validation Loss': [val_loss],
            'Validation Accuracy': [val_accuracy * 100]
            })], ignore_index=True)
            # Update the chart
            chart.line_chart(data)

        st.write(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f}")

def validate_model(model, val_loader, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    criterion = CrossEntropyLoss()
    criterion_ae = nn.MSELoss()

    with torch.no_grad():  # Disable gradient calculation
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            if isinstance(model, ViT_Autoencoder):
                output = model(x)
                loss = criterion_ae(output, x)
            else:
                y_hat, _, output = model(x)
                loss = criterion(y_hat, y)
            loss = criterion(y_hat, y)
            val_loss += loss.detach().cpu().item() / len(val_loader)

            # Calculate accuracy
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    val_accuracy = correct / total
    return val_loss, val_accuracy

def test_model(model, test_loader, device):
    criterion = CrossEntropyLoss()
    criterion_ae = nn.MSELoss()
    progress_bar = st.progress(0)
    correct, total = 0, 0
    test_loss = 0.0
    for i, (x, y) in enumerate(tqdm(test_loader, desc="Testing")):
        x, y = x.to(device), y.to(device)
        if isinstance(model, ViT_Autoencoder):
            output = model(x)
            loss = criterion_ae(output, x)
        else:
            y_hat, _, output = model(x)
            loss = criterion(y_hat, y)
        loss = criterion(y_hat, y)
        test_loss += loss.detach().cpu().item() / len(test_loader)
        progress_bar.progress((i + 1) / len(test_loader))

        correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
        total += len(x)

    st.write(f"Test loss: {test_loss:.2f}")
    st.write(f"Test accuracy: {correct / total * 100:.2f}%")
    return correct, total

def save_model_button(model, optimizer, N_EPOCHS, params, accuracy):
    if st.button('Save Model'):
        # Get current time to use in filename
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')

        # Create a unique filename for each model
        filename = f'model_{timestamp}_epochs_{N_EPOCHS}_accuracy_{accuracy * 100:.2f}.pth'

        directory = 'models'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the model
        save_model(model, optimizer, N_EPOCHS, params, f'{directory}/{filename}.pth')    
        st.write('Model saved successfully.')

def random_param_training(train_loader, val_loader, test_loader, device):
    # Define the hyperparameters and their possible values
    param_dist = {
        'num_patches': [7],
        'num_blocks': [2, 3, 5, 7],
        'hidden_dim': [8, 16, 32],
        'num_heads': [2, 3, 4, 5, 6],
        'out_d': [10],
        'lr': np.logspace(-3, -1, 100),
        'N_EPOCHS': [5, 6, 7, 8, 9, 10]
    }

    # Create a list of n randomly selected combinations of hyperparameters
    n_iter = 10
    sampler = list(ParameterSampler(param_dist, n_iter=n_iter))

    # For each combination of hyperparameters, train a model and evaluate it on the validation set
    best_model = None
    best_val_accuracy = 0
    for params in sampler:
        model_params = {k: v for k, v in params.items() if k != 'N_EPOCHS'}
        model, optimizer = get_model(model_params)
        model = model.to(device)
        train_model(model, optimizer, train_loader, val_loader, device, params['N_EPOCHS'])  # This function should train a model with the given hyperparameters and return it
        correct, total = test_model(model, test_loader, device)  # This function should evaluate the model on the validation set and return the accuracy
        val_accuracy = correct / total
        if val_accuracy > best_val_accuracy:
            best_model = model
            best_val_accuracy = val_accuracy
    return best_model, best_val_accuracy



def main():
    # Loading data
    train_loader, test_loader, val_loader = load_data()

    ## Render UI with streamlit

    st.title('Vision Transformer UI')

    # Sidebar for user inputs
    st.sidebar.header('Model Parameters')
    num_patches = st.sidebar.slider('num_patches', 1, 10, 7)
    num_blocks = st.sidebar.slider('num_blocks', 1, 10, 5)
    hidden_dim = st.sidebar.slider('hidden_dim', 1, 10, 8)
    num_heads = st.sidebar.slider('num_heads', 1, 10, 2)
    out_d = st.sidebar.slider('out_d', 1, 10, 10)
    N_EPOCHS = st.sidebar.slider('epochs', 1, 10, 5)
    LRV = st.sidebar.slider('LR', 1, 10, 5)
    LR = LRV/1000

    ## add sliders for hyperparameters

    params = {'num_patches': num_patches, 'num_blocks': num_blocks, 'hidden_dim': hidden_dim, 'num_heads': num_heads, 'out_d': out_d, 'lr': LR}

    # Get a list of all saved models
    saved_models = os.listdir('models') if os.path.exists('models') else []

    # Create a selectbox in the sidebar for selecting a model
    selected_model = st.sidebar.selectbox('Select a model', ['New model'] + saved_models)

    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = None
    if 'ae_mode' not in st.session_state:
        st.session_state.ae_mode = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if st.sidebar.button('Load/Initialize Model'):
        # If a model is selected, load it. Otherwise, create a new model.
        if selected_model != 'New model':
            st.session_state.model, st.session_state.optimizer = get_model(params, os.path.join('models', selected_model))
        else:
            st.session_state.model, st.session_state.optimizer = get_model(params)
    # if st.sidebar.button('Use autoencoder'):
        

    # Defining model and training options
    if st.session_state.model is not None:
        print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
        model = st.session_state.model.to(device)

        if st.button('Train Model'):
            train_model(model, st.session_state.optimizer, train_loader, val_loader, device, N_EPOCHS)

        correct, total = None, None
        if 'model_accuracy' not in st.session_state:
            st.session_state.model_accuracy = None

        if st.button('Test Model'):
            correct, total = test_model(model, test_loader, device)
            st.session_state.model_accuracy = correct / total

        if st.session_state.model_accuracy is not None:
            save_model_button(model, st.session_state.optimizer, N_EPOCHS, params, st.session_state.model_accuracy)
    
    if st.button('Random Param Testing'):
        best_model, best_val_accuracy = random_param_training(train_loader, val_loader, test_loader, device)
        st.session_state.model = best_model
        st.session_state.model_accuracy = best_val_accuracy
                


if __name__ == '__main__':
    main()
    
