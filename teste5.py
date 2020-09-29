import numpy
from keras.utils import np_utils

epochs = 100
stepsPerEpochs=10
VALIDATION_SIZE = 1
BATCH_SIZE = 128
checkpoint ="weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"


# load ascii text and covert to lowercase
filename = "wonderlandFull.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)


# prepare the dataset of input to output pairs encoded as integers
seq_length = 100


# Load data
def generate_arrays_from_file(path, batchsize):
    dataX = []
    dataY = []
    batchcount = 0
    seq_length = 100
    while True:
        with open(path) as f:

            for i in range(0, n_chars - seq_length, 1):
                seq_in = raw_text[i:i + seq_length]
                seq_out = raw_text[i + seq_length]
                dataX.append([char_to_int[char] for char in seq_in])
                dataY.append(char_to_int[seq_out])

                batchcount += 1
                if batchcount > batchsize:
                    #zeros = numpy.zeros((batchsize+1, 1))
                    #ys = numpy.array(dataY).reshape(-1,1)
                    #teste3 = numpy.column_stack((ys, zeros,zeros))

                    #y = np_utils.to_categorical(dataY)

                    #y = numpy.array(dataY)
                    #y = [char_to_int[char] for char in dataY]

                    onehot_encoded = list()

                    for data in dataY:
                        letter = [0 for _ in range(n_vocab)]
                        letter[data] = 1

                        onehot_encoded.append(letter)

                    yield (numpy.reshape(dataX, (batchsize+1, seq_length, 1)), onehot_encoded)
                    dataX = []
                    dataY = []
                    batchcount = 0

# Load data
def generate_arrays_from_file2(path, batchsize):
    dataX = []
    dataY = []
    batchcount = 0
    seq_length = 100
    while True:
        with open(path) as f:

            for i in range(0, n_chars - seq_length, 1):
                seq_in = raw_text[i:i + seq_length]
                seq_out = raw_text[i + seq_length]
                dataX.append([char_to_int[char] for char in seq_in])
                dataY.append(char_to_int[seq_out])

                batchcount += 1
                if batchcount > batchsize:
                    onehot_encoded = list()

                    for data in dataY:
                        letter = [0 for _ in range(n_vocab)]
                        letter[data] = 1

                        onehot_encoded.append(letter)


                    #y = [char_to_int[char] for char in dataY]


                    dataX = []
                    dataY = []
                    batchcount = 0

a = generate_arrays_from_file2(filename,128)
b = next(a)
for c in b[1]:
    print(c)
