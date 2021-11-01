# Transfomer-Text-to-Image-GAN
This repo details the implementation of a Text to Image GAN trained on the Caltech-UCSD Birds Dataset.<br>

# Architecture
The architecture is based on a combination of GAN-CLS and MS-GAN referred from https://github.com/Yoan-D/text-to-image-synthesis. Instead of using a word2vec pretrained model, this model trains its own word embeddings and Transformer model to extract features from the caption text. The transformer model utilized is shown below: <br>
![transformer](https://user-images.githubusercontent.com/36445587/139666430-7a82b47c-1d18-4866-8981-cfbd42c02a9b.png)
