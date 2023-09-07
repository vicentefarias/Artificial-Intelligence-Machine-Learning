# -*- coding: utf-8 -*-
"""flickr_caption_generator.ipynb
# COMS W4705 - Homework 4
## Image Captioning with Conditioned LSTM Generators

Follow the instructions in this notebook step-by step. Much of the code is provided, but some sections are marked with **todo**.

Specifically, you will build the following components:

* Create matrices of image representations using an off-the-shelf image encoder.
* Read and preprocess the image captions.
* Write a generator function that returns one training instance (input/output sequence pair) at a time.
* Train an LSTM language generator on the caption data.
* Write a decoder function for the language generator.
* Add the image input to write an LSTM caption generator.
* Implement beam search for the image caption generator.

Please see the special submission instructions at the bottom.

This notebook assumes that you are running it on a machine with the possibility of using a GPU (like the VM on GCP as instructed in tutorial).

### Getting Started

First, run the following commands to make sure you have all required packages.
"""

# Commented out IPython magic to ensure Python compatibility.
import os
from collections import defaultdict
import numpy as np
import PIL
from matplotlib import pyplot as plt


from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Input, Bidirectional, RepeatVector, Concatenate, Activation
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.optimizers import Adam

"""### Access to the flickr8k data

We will use the flickr8k data set, described here in more detail:

> M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artificial Intelligence Research, Volume 47, pages 853-899 http://www.jair.org/papers/paper3994.html when discussing our results

The data is available here: https://storage.googleapis.com/4705-hw5-data/hw5data-20220809T182644Z-001.zip

N.B.: Usage of this data is limited to this homework assignment. If you would like to experiment with the data set beyond this course, I suggest that you submit your own download request here (it's free): https://forms.illinois.edu/sec/1713398
"""

"""If you are using a separate persistent disk to store the data, ssh into your VM instance and copy the data directory there. Either way, you want the following variable to point to the location of the data directory."""

FLICKR_PATH="hw5data"

"""## Part I: Image Encodings (14 pts)

The files Flickr_8k.trainImages.txt Flickr_8k.devImages.txt Flickr_8k.testImages.txt, contain a list of training, development, and test images, respectively. Let's load these lists.
"""

def load_image_list(filename):
    with open(filename,'r') as image_list_f:
        return [line.strip() for line in image_list_f]

train_list = load_image_list(os.path.join(FLICKR_PATH, 'Flickr_8k.trainImages.txt'))
dev_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.devImages.txt'))
test_list = load_image_list(os.path.join(FLICKR_PATH,'Flickr_8k.testImages.txt'))

"""Let's see how many images there are"""

len(train_list), len(dev_list), len(test_list)

"""Each entry is an image filename."""

dev_list[20]

"""The images are located in a subdirectory.  """

IMG_PATH = os.path.join(FLICKR_PATH, "Flickr8k_Dataset")

"""We can use PIL to open the image and matplotlib to display it."""

image = PIL.Image.open(os.path.join(IMG_PATH, dev_list[20]))
image

"""if you can't see the image, try"""

plt.imshow(image)

"""We are going to use an off-the-shelf pre-trained image encoder, the Inception V3 network. The model is a version of a convolution neural network for object detection. Here is more detail about this model (not required for this project):

> Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2818-2826).
> https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html

The model requires that input images are presented as 299x299 pixels, with 3 color channels (RGB). The individual RGB values need to range between 0 and 1.0. The flickr images don't fit.
"""

np.asarray(image).shape

"""The values range from 0 to 255."""

np.asarray(image)

"""We can use PIL to resize the image and then divide every value by 255."""

new_image = np.asarray(image.resize((299,299))) / 255.0
plt.imshow(new_image)

new_image.shape

"""Let's put this all in a function for convenience."""

def get_image(image_name):
    image = PIL.Image.open(os.path.join(IMG_PATH, image_name))
    return np.asarray(image.resize((299,299))) / 255.0

plt.imshow(get_image(dev_list[25]))

"""Next, we load the pre-trained Inception model."""

img_model = InceptionV3(weights='imagenet') # This will download the weight files for you and might take a while.

img_model.summary() # this is quite a complex model.

"""This is a prediction model,so the output is typically a softmax-activated vector representing 1000 possible object types. Because we are interested in an encoded representation of the image we are just going to use the second-to-last layer as a source of image encodings. Each image will be encoded as a vector of size 2048.

We will use the following hack: hook up the input into a new Keras model and use the penultimate layer of the existing model as output.
"""

new_input = img_model.input
new_output = img_model.layers[-2].output
img_encoder = Model(new_input, new_output) # This is the final Keras image encoder model we will use.

"""Let's try the encoder. At this point, you may want to add a GPU to the VM you are using (if not using already)."""

encoded_image = img_encoder.predict(np.array([new_image]))

encoded_image

"""**TODO:** We will need to create encodings for all images and store them in one big matrix (one for each dataset, train, dev, test).
We can then save the matrices so that we never have to touch the bulky image data again.

To save memory (but slow the process down a little bit) we will read in the images lazily using a generator. We will encounter generators again later when we train the LSTM. If you are unfamiliar with generators, take a look at this page: https://wiki.python.org/moin/Generators

Write the following generator function, which should return one image at a time.
`img_list` is a list of image file names (i.e. the train, dev, or test set). The return value should be a numpy array of shape (1,299,299,3).
"""

def img_generator(img_list):
    ret = np.zeros((1,299,299,3))
    for img in img_list:
        image = PIL.Image.open(os.path.join(IMG_PATH, img))
        new_image = np.asarray(image.resize((299,299))) / 255.0
        ret[0] = new_image
        yield ret

"""Now we can encode all images (this takes a few minutes)."""

enc_train = img_encoder.predict_generator(img_generator(train_list), steps=len(train_list), verbose=1)

enc_train[11]

enc_dev = img_encoder.predict_generator(img_generator(dev_list), steps=len(dev_list), verbose=1)

enc_test = img_encoder.predict_generator(img_generator(test_list), steps=len(test_list), verbose=1)

"""It's a good idea to save the resulting matrices, so we do not have to run the encoder again."""

# Choose a suitable location here, please do NOT attempt to write your output files to the shared data directory.
OUTPUT_PATH = "hw5output"
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

np.save(os.path.join(OUTPUT_PATH,"encoded_images_train.npy"), enc_train)
np.save(os.path.join(OUTPUT_PATH,"encoded_images_dev.npy"), enc_dev)
np.save(os.path.join(OUTPUT_PATH,"encoded_images_test.npy"), enc_test)

"""## Part II Text (Caption) Data Preparation (14 pts)

Next, we need to load the image captions and generate training data for the generator model.

### Reading image descriptions

**TODO**: Write the following function that reads the image descriptions from the file `filename` and returns a dictionary in the following format. Take a look at the file `Flickr8k.token.txt` for the format of the input file.
The keys of the dictionary should be image filenames. Each value should be a list of 5 captions. Each caption should be a list of tokens.  

The captions in the file are already tokenized, so you can just split them at white spaces. You should convert each token to lower case. You should then pad each caption with a START token on the left and an END token on the right.
"""

def read_image_descriptions(filename):
    image_descriptions = defaultdict(list)
    f = open(filename, 'r')
    lines = f.readlines()
    count = 0
    captions = []
    for l in lines:
        l = l[0:len(l)-1]
        tokens = l.split('#')
        img_name = tokens[0]
        caption = ['<START>']
        cap = tokens[1].split('\t')[1].lower()
        caption += cap.split(' ')
        caption += ['<END>']
        captions.append(caption)
        count += 1
        if count == 5:
            image_descriptions[img_name] = captions
            count = 0
            captions = []
    f.close()
    return image_descriptions

descriptions = read_image_descriptions(f"{FLICKR_PATH}/Flickr8k.token.txt")

print(descriptions[dev_list[0]])

"""Running the previous cell should print:     
`[['<START>', 'the', 'boy', 'laying', 'face', 'down', 'on', 'a', 'skateboard', 'is', 'being', 'pushed', 'along', 'the', 'ground', 'by', 'another', 'boy', '.', '<END>'], ['<START>', 'two', 'girls', 'play', 'on', 'a', 'skateboard', 'in', 'a', 'courtyard', '.', '<END>'], ['<START>', 'two', 'people', 'play', 'on', 'a', 'long', 'skateboard', '.', '<END>'], ['<START>', 'two', 'small', 'children', 'in', 'red', 'shirts', 'playing', 'on', 'a', 'skateboard', '.', '<END>'], ['<START>', 'two', 'young', 'children', 'on', 'a', 'skateboard', 'going', 'across', 'a', 'sidewalk', '<END>']]
`

### Creating Word Indices
"""

tokens = set()
for cList in descriptions.values():
    for caption in cList:
        for tok in caption:
            tokens.add(tok)
tokens = sorted(list(set(tokens)))
id_to_word = {v: k for v, k in enumerate(tokens)}

"""Next, we need to create a lookup table from the **training** data mapping words to integer indices, so we can encode input
and output sequences using numeric representations. **TODO** create the dictionaries id_to_word and word_to_id, which should map tokens to numeric ids and numeric ids to tokens.  
Hint: Create a set of tokens in the training data first, then convert the set into a list and sort it. This way if you run the code multiple times, you will always get the same dictionaries.
"""

word_to_id = {k: v for v, k in enumerate(tokens)}

word_to_id['dog'] # should print an integer

id_to_word[1985] # should print a token

"""Note that we do not need an UNK word token because we are generating. The generated text will only contain tokens seen at training time.

## Part III Basic Decoder Model (24 pts)

For now, we will just train a model for text generation without conditioning the generator on the image input.

There are different ways to do this and our approach will be slightly different from the generator discussed in class.

The core idea here is that the Keras recurrent layers (including LSTM) create an "unrolled" RNN. Each time-step is represented as a different unit, but the weights for these units are shared. We are going to use the constant MAX_LEN to refer to the maximum length of a sequence, which turns out to be 40 words in this data set (including START and END).
"""

max(len(description) for image_id in train_list for description in descriptions[image_id])

"""In class, we discussed LSTM generators as transducers that map each word in the input sequence to the next word.
<img src="http://www.cs.columbia.edu/~bauer/4705/lstm1.png" width="480px">

Instead, we will use the model to predict one word at a time, given a partial sequence. For example, given the sequence ["START","a"], the model might predict "dog" as the most likely word. We are basically using the LSTM to encode the input sequence up to this point.
<img src="http://www.cs.columbia.edu/~bauer/4705/lstm2.png" width="480px">

To train the model, we will convert each description into a set of input output pairs as follows. For example, consider the sequence

`['<START>', 'a', 'black', 'dog', '.', '<END>']`

We would train the model using the following input/output pairs

| i | input                        | output |
|---|------------------------------|--------|
| 0 |[`START`]                     | `a`    |  
| 1 |[`START`,`a`]                 | `black`|
| 2 |[`START`,`a`, `black`]        | `dog`  |
| 3 |[`START`,`a`, `black`, `dog`] | `END`  |

Here is the model in Keras Keras. Note that we are using a Bidirectional LSTM, which encodes the sequence from both directions and then predicts the output.
Also note the `return_sequence=False` parameter, which causes the LSTM to return a single output instead of one output per state.

Note also that we use an embedding layer for the input words. The weights are shared between all units of the unrolled LSTM. We will train these embeddings with the model.
"""

MAX_LEN = 40
EMBEDDING_DIM=300
vocab_size = len(word_to_id)

# Text input
text_input = Input(shape=(MAX_LEN,))
embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(text_input)
x = Bidirectional(LSTM(512, return_sequences=False))(embedding)
pred = Dense(vocab_size, activation='softmax')(x)
model = Model(inputs=[text_input],outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.summary()

"""The model input is a numpy ndarray (a tensor) of size `(batch_size, MAX_LEN)`. Each row is a vector of size MAX_LEN in which each entry is an integer representing a word (according to the `word_to_id` dictionary). If the input sequence is shorter than MAX_LEN, the remaining entries should be padded with 0.

For each input example, the model returns a softmax activated vector (a probability distribution) over possible output words. The model output is a numpy ndarray of size `(batch_size, vocab_size)`. vocab_size is the number of vocabulary words.

### Creating a Generator for the Training Data

**TODO**:

We could simply create one large numpy ndarray for all the training data. Because we have a lot of training instances (each training sentence will produce up to MAX_LEN input/output pairs, one for each word), it is better to produce the training examples *lazily*, i.e. in batches using a generator (recall the image generator in part I).

Write the function `text_training_generator` below, that takes as a paramater the batch_size and returns an `(input, output)` pair. `input` is a `(batch_size, MAX_LEN)` ndarray of partial input sequences, `output` contains the next words predicted for each partial input sequence, encoded as a `(batch_size, vocab_size)` ndarray.

Each time the next() function is called on the generator instance, it should return a new batch of the *training* data. You can use `train_list` as a list of training images. A batch may contain input/output examples extracted from different descriptions or even from different images.

You can just refer back to the variables you have defined above, including `descriptions`, `train_list`, `vocab_size`, etc.

Hint: To prevent issues with having to reset the generator for each epoch and to make sure the generator can always return exactly `batch_size` input/output pairs in each step, wrap your code into a `while True:` loop. This way, when you reach the end of the training data, you will just continue adding training data from the beginning into the batch.
"""

def text_training_generator(batch_size=128):
    i = 0
    inputs = np.zeros((batch_size, MAX_LEN))
    outputs = np.zeros((batch_size, vocab_size))
    while True:
        for img_id in train_list:
            for description in descriptions[img_id]:
                tokenized = [word_to_id[word] for word in description]
                for j in range(1, len(description)):
                    inRow = np.zeros(MAX_LEN)
                    outRow = np.zeros(vocab_size)
                    inRow[:j] = tokenized[:j]
                    nxt = tokenized[j]
                    outRow[nxt] = 1
                    inputs[i] = inRow
                    outputs[i] = outRow
                    i += 1
                    if i%(batch_size)==0:
                        yield((inputs, outputs))
                        i = 0
                        inp = np.zeros((batch_size, MAX_LEN))
                        out = np.zeros((batch_size, vocab_size))

"""### Training the Model

We will use the `fit_generator` method of the model to train the model. fit_generator needs to know how many iterator steps there are per epoch.

Because there are len(train_list) training samples with up to `MAX_LEN` words, an upper bound for the number of total training instances is `len(train_list)*MAX_LEN`. Because the generator returns these in batches, the number of steps is len(train_list) * MAX_LEN // batch_size
"""

batch_size = 128
generator = text_training_generator(batch_size)
steps = len(train_list) * MAX_LEN // batch_size

model.fit_generator(generator, steps_per_epoch=steps, verbose=True, epochs=10)

"""### Greedy Decoder

**TODO** Next, you will write a decoder. The decoder should start with the sequence `["<START>"]`, use the model to predict the most likely word, append the word to the sequence and then continue until `"<END>"` is predicted or the sequence reaches `MAX_LEN` words.
"""

def decoder():
    input = np.zeros((1,MAX_LEN))
    input[0][0] = word_to_id['<START>']
    for i in range(1, MAX_LEN):
      output = model.predict(input)[0]
      id = output.argmax()
      input[0][i] = id
      if id == word_to_id['<END>']:
        return [id_to_word[id] for id in input[0]]
    return [id_to_word[id] for id in input[0]]

print(decoder())

"""This simple decoder will of course always predict the same sequence (and it's not necessarily a good one).

Modify the decoder as follows. Instead of choosing the most likely word in each step, sample the next word from the distribution (i.e. the softmax activated output) returned by the model. Take a look at the [np.random.multinomial](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.multinomial.html) function to do this.
"""

def sample_decoder():
    input = np.zeros((1,MAX_LEN))
    input[0][0] = word_to_id['<START>']
    for i in range(1, MAX_LEN):
      out = model.predict(input)[0]
      out = np.asarray(out).astype('float64')
      out = out / np.sum(out)
      prob = np.random.multinomial(1, out, 1)
      id = prob.argmax()
      input[0][i] = id
      if id == word_to_id['<END>']:
        return [id_to_word[id] for id in input[0]]
    return [id_to_word[id] for id in input[0]]

"""You should now be able to see some interesting output that looks a lot like flickr8k image captions -- only that the captions are generated randomly without any image input."""

for i in range(10):
    print(sample_decoder())

"""## Part III - Conditioning on the Image (24 pts)

We will now extend the model to condition the next word not only on the partial sequence, but also on the encoded image.

We will project the 2048-dimensional image encoding to a 300-dimensional hidden layer.
We then concatenate this vector with each embedded input word, before applying the LSTM.

Here is what the Keras model looks like:
"""

MAX_LEN = 40
EMBEDDING_DIM=300
IMAGE_ENC_DIM=300

# Image input
img_input = Input(shape=(2048,))
img_enc = Dense(300, activation="relu") (img_input)
images = RepeatVector(MAX_LEN)(img_enc)

# Text input
text_input = Input(shape=(MAX_LEN,))
embedding = Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN)(text_input)
x = Concatenate()([images,embedding])
y = Bidirectional(LSTM(256, return_sequences=False))(x)
pred = Dense(vocab_size, activation='softmax')(y)
model = Model(inputs=[img_input,text_input],outputs=pred)
model.compile(loss='categorical_crossentropy', optimizer="RMSProp", metrics=['accuracy'])

model.summary()

"""The model now takes two inputs:
    
   1. a `(batch_size, 2048)` ndarray of image encodings.
   2. a `(batch_size, MAX_LEN)` ndarray of partial input sequences.
    
And one output as before: a `(batch_size, vocab_size)` ndarray of predicted word distributions.

**TODO**: Modify the training data generator to include the image with each input/output pair.
Your generator needs to return an object of the following format: `([image_inputs, text_inputs], next_words)`. Where each element is an ndarray of the type described above.  

You need to find the image encoding that belongs to each image. You can use the fact that the index of the image in `train_list` is the same as the index in enc_train and enc_dev.

If you have previously saved the image encodings, you can load them from disk:
"""

enc_train = np.load(f"{OUTPUT_PATH}/encoded_images_train.npy")
enc_dev = np.load(f"{OUTPUT_PATH}/encoded_images_dev.npy")

def training_generator(batch_size=128):
    i = 0
    imgIn = np.zeros((batch_size, 2048))
    textIn = np.zeros((batch_size, MAX_LEN))
    outputs = np.zeros((batch_size, vocab_size))
    while True:
        img_idx = 0
        for img_id in train_list:
            img = get_image(img_id)
            for description in descriptions[img_id]:
                tokenized = [word_to_id[word] for word in description]
                for j in range(1, len(description)):
                    inRow = np.zeros(MAX_LEN)
                    outRow = np.zeros(vocab_size)
                    inRow[:j] = tokenized[:j]
                    nxt = tokenized[j]
                    outRow[nxt] = 1
                    imgIn[i] = enc_train[img_idx]
                    textIn[i] = inRow
                    outputs[i] = outRow
                    i += 1
                    if i%(batch_size)==0:
                        yield(([imgIn, textIn], outputs))
                        i = 0
                        inp = np.zeros((batch_size, MAX_LEN))
                        out = np.zeros((batch_size, vocab_size))
            img_idx += 1

"""You should now be able to train the model as before:"""

batch_size = 128
generator = training_generator(batch_size)
steps = len(train_list) * MAX_LEN // batch_size

model.fit_generator(generator, steps_per_epoch=steps, verbose=True, epochs=20)

"""Continue to train the model until you hit an accuracy of about 40%. This may take a while. I strongly encourage you to experiment with cloud GPUs using the GCP voucher for the class.

You can save your model weights to disk and continue at a later time.
"""

model.save_weights(f"{OUTPUT_PATH}/model.h5")

"""to load the model:"""

model.load_weights(f"{OUTPUT_PATH}/model.h5")

"""**TODO**: Now we are ready to actually generate image captions using the trained model. Modify the simple greedy decoder you wrote for the text-only generator, so that it takes an encoded image (a vector of length 2048) as input, and returns a sequence."""

def image_decoder(enc_image):
    img = np.zeros((1, 2048))
    input = np.zeros((1,MAX_LEN))
    input[0][0] = word_to_id['<START>']
    img[0] = enc_image
    for i in range(1, MAX_LEN):
      out = model.predict([img, input])[0]
      out = np.asarray(out).astype('float64')
      out = out / np.sum(out)
      prob = np.random.multinomial(1, out, 1)
      id = prob.argmax()
      input[0][i] = id
      if id == word_to_id['<END>']:
        return [id_to_word[id] for id in input[0]]
    return [id_to_word[id] for id in input[0]]

"""As a sanity check, you should now be able to reproduce (approximately) captions for the training images."""

plt.imshow(get_image(train_list[0]))
image_decoder(enc_train[0])

"""You should also be able to apply the model to dev images and get reasonable captions:"""

plt.imshow(get_image(dev_list[1]))
image_decoder(enc_dev[1])

"""For this assignment we will not perform a formal evaluation.

Feel free to experiment with the parameters of the model or continue training the model. At some point, the model will overfit and will no longer produce good descriptions for the dev images.

## Part IV - Beam Search Decoder (24 pts)

**TODO** Modify the simple greedy decoder for the caption generator to use beam search.
Instead of always selecting the most probable word, use a *beam*, which contains the n highest-scoring sequences so far and their total probability (i.e. the product of all word probabilities). I recommend that you use a list of `(probability, sequence)` tuples. After each time-step, prune the list to include only the n most probable sequences.

Then, for each sequence, compute the n most likely successor words. Append the word to produce n new sequences and compute their score. This way, you create a new list of n*n candidates.

Prune this list to the best n as before and continue until `MAX_LEN` words have been generated.

Note that you cannot use the occurence of the `"<END>"` tag to terminate generation, because the tag may occur in different positions for different entries in the beam.

Once `MAX_LEN` has been reached, return the most likely sequence out of the current n.
"""

def beam_decoder(beam_width, enc_image):
    img = np.zeros((1, 2048))
    input = np.zeros((1, MAX_LEN))
    input[0][0] = word_to_id['<START>']
    img[0] = enc_image

    beam = [(0.0, input)]
    for i in range(1, MAX_LEN):
        candidates = []
        for prob, seq in beam:
            out = model.predict([img, seq])[0]
            out = np.asarray(out).astype('float64')
            out = out / np.sum(out)
            word_ids = np.argsort(out)[-beam_width:]  # Select top beam_width words
            for word_id in word_ids:
                candidate_seq = seq.copy()
                candidate_seq[0][i] = word_id
                candidates.append((prob + np.log(out[word_id]), candidate_seq))

        candidates.sort(reverse=True, key=lambda x: x[0])
        beam = candidates[:beam_width]

    best_seq = beam[0][1]
    caption = []
    for word_id in best_seq[0]:
        caption.append(id_to_word[word_id])
        if id_to_word[word_id] == '<END>':
            break

    return caption

beam_decoder(3, enc_dev[1])

"""**TODO** Finally, before you submit this assignment, please show 5 development images, each with 1) their greedy output, 2) beam search at n=3 3) beam search at n=5."""

plt.imshow(get_image(dev_list[2]))

image_decoder(enc_dev[2])

beam_decoder(3, enc_dev[2])

beam_decoder(5, enc_dev[2])

plt.imshow(get_image(dev_list[3]))

image_decoder(enc_dev[3])

beam_decoder(3, enc_dev[3])

beam_decoder(5, enc_dev[3])

plt.imshow(get_image(dev_list[4]))

image_decoder(enc_dev[4])

beam_decoder(3, enc_dev[4])

beam_decoder(5, enc_dev[4])

plt.imshow(get_image(dev_list[7]))

image_decoder(enc_dev[7])

beam_decoder(3, enc_dev[7])

beam_decoder(5, enc_dev[7])

plt.imshow(get_image(dev_list[11]))

image_decoder(enc_dev[11])

beam_decoder(3, enc_dev[11])

beam_decoder(5, enc_dev[11])
