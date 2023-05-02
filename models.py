"""Definition of the Chroma Model. The model takes in Noisy Coordinates [B, N, 4, 3] of the backbone atoms and
outputs Denoised Coordinates [B, N, 4, 3]. The entire model is end-to-end trainable.
The main components include:
    1. Graph Sampling to create a random graph
    2. Graph Featurization
    3. Graph Neural Network Layer
    4. Inter-residue Geometry Prediction
    5. Backbone Solver
"""
