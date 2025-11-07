'''
Each protein is a string of amino-acid letters:
- but neural networks can't process letters
They need numbers (tensors) as input

here we convert each amino acids letter into a numeric token ID
and pads all sequences to the same length
which makes them usable by Pytorch

Why Encode?

'''

