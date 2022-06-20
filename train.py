from operator import mod
import torch
from tqdm import tqdm
import data
from models import Encoder, Decoder, Seq2Seq
from data import train_iterator, test_iterator
import config

# torch.backends.cudnn.benchmark = True

encoder = Encoder(
    config.ENCODER_IN_CH,
    config.ENCODER_EMBED_DIM,
    config.ENCODER_HIDDEN_SIZE,
    config.ENCODER_NUM_LAYERS,
    config.E_DROP)

decoder = Decoder(
    config.DECODER_IN_CH,
    config.ENCODER_EMBED_DIM,
    config.DECODER_HIDDEN_SIZE,
    config.DECODER_NUM_LAYERS,
    config.D_DROP,
    config.OUTPUT_SIZE,
)

model = Seq2Seq(encoder, decoder).to('cuda')
optim = torch.optim.Adam(model.parameters(), lr=config.LR)
loss_func = torch.nn.CrossEntropyLoss(ignore_index=data.desc_field.vocab.stoi['<pad>'])

sent = 'Wall St. Bears Claw Back Into the Black'

def translate_sentence(model, sentence, device, max_length=360):
    tokens = [token.lower() for token in sentence]
    tokens.insert(0, data.title_field.init_token)
    tokens.append(data.title_field.eos_token)
    text_to_indices = [data.title_field.vocab.stoi[token] for token in tokens]
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [data.desc_field.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to('cuda')

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == data.title_field.vocab.stoi["<eos>"]:
            break

    translated_sentence = [data.desc_field.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def train_step():
    loop = tqdm(train_iterator)
    
    for idx, batch in enumerate(loop):
        x = batch.t.to('cuda')
        y = batch.d.to('cuda')

        output = model(x, y)

        output = output[1:].reshape(-1, output.shape[2])
        y = y[1:].reshape(-1)

        optim.zero_grad()
        loss = loss_func(output, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
        optim.step()
    print(f'train loss : {loss.item()}')

def test_step():
    loop = tqdm(test_iterator)
    for idx, batch in enumerate(loop):
        x = batch.t.to('cuda')
        y = batch.d.to('cuda')
        output = model(x, y)

        output = output[1:].reshape(-1, output.shape[2])
        y = y[1:].reshape(-1)

        optim.zero_grad()
        loss = loss_func(output, y)
    print(f'validation loss : {loss}')
    print('Translated word : ', translate_sentence(model, sent, 'cuda'))
    y1 = y.T
    y1 = y1[0]
    print('Expected : ', [data.desc_field.vocab.itos[i] for i in y1])

for e in range(1, config.EPOCHS):
    print(f'[{e}/{config.EPOCHS}]======>')
    model.train()
    train_step()
    model.eval()
    test_step()
