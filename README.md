# Transfomer-Text-to-Image-GAN
This repo details the implementation of a Text to Image GAN trained on the Caltech-UCSD Birds Dataset.<br>

# Architecture
The architecture is based on a combination of GAN-CLS and MS-GAN referred from https://github.com/Yoan-D/text-to-image-synthesis. Instead of using a word2vec pretrained model, this model trains its own word embeddings and Transformer model to extract features from the caption text. The transformer model utilized is shown below: <br>
![transformer](https://user-images.githubusercontent.com/36445587/139668651-a460214e-7245-4e50-8b93-796b8cc1fb57.png)

The output of the transformer of the shape (batch size, seq_length, d_model) has been average pooled over the temporal dimension and has been concatenated to the inputs of both the Generator and Discriminator of the GAN. The architecture is modelled as shown below:

![1_va0ul6e3xOAwlkxvWA3WoA](https://user-images.githubusercontent.com/36445587/139667752-0939ee78-4b21-4cf0-b011-e986b84d8ee3.png)

# Usage
Generate the text vectors:<br>
```bash
$ python3 gen_text_vectors.py
```

Generate the image vectors:<br>
```bash
$ python3 gen_image_vectors.py
```

Train the model:
```bash
$ python3 train.py
```
