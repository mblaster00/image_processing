import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)


class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        #image = Image.new('RGB', (self.imsize, self.imsize), (255, 255, 255))
        image = white_image = np.ones((self.imsize, self.imsize, 3), dtype=np.uint8) * 255
        ske.draw(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # key = cv2.waitKey(-1)
        return image



class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )


    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        reduced = True
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # prepreocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    
    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape( ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        # Convert normalized tensor to NumPy array
        numpy_image = normalized_image.detach().cpu().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))  # Rearrange (C, H, W) to (H, W, C)

        # Denormalize from [-1, 1] to [0, 255]
        denormalized_image = (numpy_image + 1) * 127.5
        denormalized_image = denormalized_image.astype(np.uint8)  # Convert to uint8

        # Convert to BGR format for OpenCV visualization
        denormalized_image = cv2.cvtColor(denormalized_image, cv2.COLOR_RGB2BGR)
        return denormalized_image




def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




class GenNNSkeToImage(nn.Module):
    """ Class that generates an image from a skeleton posture.
        Generator(Skeleton) -> Image
    """
    def __init__(self):
        super(GenNNSkeToImage, self).__init__()
        self.input_dim = Skeleton.reduced_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(self.input_dim, 256, 4, 1, 0),  # (B, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),            # (B, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),             # (B, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),              # (B, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),               # (B, 3, 64, 64)
            nn.Tanh()                                         # Output in range [-1, 1]
        )
        print(self.model)

    def forward(self, z):
        return self.model(z)





class GenNNSkeImToImage(nn.Module):
    """ class that Generate a new image from from THE IMAGE OF the new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()
        self.input_channels = 3  # RGB image
        self.model = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=3, stride=1, padding=1),  # Input convolution layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # Pooling to reduce spatial dimensions

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 2nd convolution layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 3rd convolution layer
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(256 * (64 // 8) * (64 // 8), 512),  # Fully connected layer
            nn.ReLU(inplace=True),
            nn.Linear(512, 64 * 64 * 3),  # Output layer (RGB image)
            nn.Tanh()  # Ensure the output values are between -1 and 1 (scaled to [0, 1] later)
        )

    def forward(self, z):
        flat_output = self.model(z)
        # Reshape to (C, H, W) for RGB image
        img = flat_output.view(-1, 3, 64, 64)  # Batch size x 3 x 64 x 64
        return img






class GenVanillaNN():
    """ class that Generate a new image from a new skeleton posture
        Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        image_size = 64
        if optSkeOrImage==1:
            self.netG = GenNNSkeToImage()
            src_transform = None
            self.filename = 'data/Dance/DanceGenVanillaFromSke.pth'
        else:
            self.netG = GenNNSkeImToImage()
            src_transform = transforms.Compose([ SkeToImageTransform(image_size),
                                                 transforms.ToTensor(),
                                                 #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                                 ])
            self.filename = 'data/Dance/DanceGenVanillaFromSkeim.pth'



        tgt_transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            # [transforms.Resize((64, 64)),
                            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=16, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            self.netG = torch.load(self.filename)


    def train(self, n_epochs=20, lr=0.0002, beta1=0.5):
        """Train the generator model."""
        print("Starting training...")

        # Loss function
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))

        # Move model to the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.netG.to(device)

        # Loop through epochs
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for i, (ske, image) in enumerate(self.dataloader):
                # Move data to the device
                ske = ske.to(device)
                image = image.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass: generate images
                generated_image = self.netG(ske)

                # Compute loss
                loss = criterion(generated_image, image)

                # Backward pass: update weights
                loss.backward()
                optimizer.step()

                # Accumulate loss for the epoch
                epoch_loss += loss.item()

                # Print progress every 10 batches
                if i % 10 == 0:
                    print(f"Epoch [{epoch+1}/{n_epochs}], Batch [{i+1}/{len(self.dataloader)}], Loss: {loss.item():.4f}")

            # Print epoch loss
            print(f"Epoch [{epoch+1}/{n_epochs}] completed with Loss: {epoch_loss/len(self.dataloader):.4f}")

        # Save the model after training
        print(f"Training completed. Saving model to {self.filename}")
        torch.save(self.netG, self.filename)



    def generate(self, ske):
        """Generate an image from a skeleton"""
        print("Generating image for skeleton:", ske)

        # Preprocess the skeleton into a tensor
        ske_t = self.dataset.preprocessSkeleton(ske)
        ske_t_batch = ske_t.unsqueeze(0)  # Add a batch dimension (batch size = 1)

        # Pass the skeleton through the generator network
        normalized_output = self.netG(ske_t_batch)

        # Convert the generated tensor to an image
        res = self.dataset.tensor2image(normalized_output[0])  # Get the first (and only) batch image
        return res





if __name__ == '__main__':
    force = False
    optSkeOrImage = 2           # use as input a skeleton (1) or an image with a skeleton drawed (2)
    n_epoch = 2000  # 200
    train = 1 #False
    #train = True

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    if train:
        # Train
        gen = GenVanillaNN(targetVideoSke, loadFromFile=False)
        gen.train(n_epoch)
    else:
        gen = GenVanillaNN(targetVideoSke, loadFromFile=True)    # load from file        


    # Test with a second video
    for i in range(targetVideoSke.skeCount()):
        image = gen.generate( targetVideoSke.ske[i] )
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        key = cv2.waitKey(-1)
