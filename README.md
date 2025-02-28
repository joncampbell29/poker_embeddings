# Analyzing Latent Representations in Poker Flops Using Autoencoders
This project analyzes poker flops using neural networks, 
specifically autoencoders, to learn meaningful representations of flop characteristics. 
The goal is to capture patterns and structures within all possible flops using various encoded attributes and statistical properties.

## Data Generation
The data for the project is aall 22,100 possible flops from a standard 52 card deck. To represent the cards, I used one-hot encoding for the rank (13 values: 2–10, J, Q, K, A) and the suit (4 values: Spades, Hearts, Clubs, Diamonds) and concated to create a "flop vector". Each flop is represented as a 51-dimentional vector. 
Other important properties from the flop included suitedness, pairness, connectedness, high_low_texture, high card, and straightness. The criteria an index mappings for these are in the definitions.txt file, and the functions are in the utils.py file.

### Example
- ['5s', '7h', 'Ts']
[0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.][ 1., 0., 0., 0.]+
- [0.,0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.][ 0., 1., 0., 0.]+
[0., 0.,0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.[ 1., 0., 0., 0.]
