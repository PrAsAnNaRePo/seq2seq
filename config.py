import data

DEVICE = 'cuda'
EPOCHS = 30
BATCH_SIZE = 64
LR = 0.0001
ENCODER_IN_CH = len(data.title_field.vocab)
DECODER_IN_CH = len(data.desc_field.vocab)
ENCODER_EMBED_DIM = 100
OUTPUT_SIZE = len(data.desc_field.vocab)
ENCODER_HIDDEN_SIZE = 1024
DECODER_HIDDEN_SIZE = 1024
ENCODER_NUM_LAYERS = 1
DECODER_NUM_LAYERS = 1
E_DROP = 0.5
D_DROP = 0.4

