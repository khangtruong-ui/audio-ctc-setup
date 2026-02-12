import os

max_wav_length = 222621
data_dir = './LJSpeech-1.1/'
metadata_dir = data_dir + 'metadata.csv'
wavs_dir = data_dir + 'wavs/'
chars = ' abcdefghijklmnopqrstuvwxyz'
max_chars = 187


import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

def tree_map(f):

    def wrap_func(pytree, *args, **kwargs):
        return jax.tree.map(lambda x: f(x, *args, **kwargs), pytree)

    return wrap_func

mesh = Mesh(np.array(jax.devices()).reshape((-1, 1)), ('data', 'kernel',))
kernel_sharding = NamedSharding(mesh, P(None, 'kernel'))
conv_sharding = NamedSharding(mesh, P(None, None, None, 'kernel'))
sharding = NamedSharding(mesh, P('data'))
non_sharding = no_sharding = NamedSharding(mesh, P())

@tree_map
def distribute_device(tensor, sharding, replicate=False):
    if type(tensor) in [int, float, str]:
        return tensor
    global_shape = (tensor.shape[0] * jax.process_count(),) + tensor.shape[1:] if not replicate else tensor.shape
    global_array = jax.make_array_from_process_local_data(sharding, tensor, global_shape)
    return global_array


from scipy.io.wavfile import read as read_audio
import scipy
import jax
import jax.numpy as jnp
import numpy as np

import pandas as pd

# Read metadata file and parse it
metadata_df = pd.read_csv(metadata_dir, sep="|", header=None, quoting=3)
metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
metadata_df = metadata_df[["file_name", "normalized_transcription"]]
metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
metadata_df.head(3)


from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class TorchDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds) * 1000000

    def __getitem__(self, i):
        item = self.ds[i % len(self.ds)]
        output = jax.tree.map(lambda x: np.array(x), item)
        return output


class DataSource:
    def __init__(self, df, datapath='LJSpeech-1.1'):
        self.df = df
        self.datapath = datapath

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = self.df.iloc[i]
        fname, transcript = item['file_name'], item['normalized_transcription']
        spectrogram, spectrogram_mask = self.get_spectrogram(fname)
        pad_token, label_mask = self.get_tokens_and_mask(transcript)
        return dict(
            spectrogram=spectrogram, 
            label=pad_token, 
            mask_label=label_mask, 
            mask_spectrogram=spectrogram_mask
        )

    def get_tokens_and_mask(self, transcript):
        tokenized = np.array([chars.index(c) for c in transcript if c in chars])
        pad_length = max_chars - tokenized.shape[-1]
        pad_token = np.pad(tokenized, (0, pad_length), mode='constant', constant_values=0)
        mask = (np.arange(max_chars) >= tokenized.shape[-1]).astype(np.float32)
        return pad_token, mask
    
    def get_spectrogram(self, fname):
        _, wav = read_audio(f'{self.datapath}/wavs/{fname}.wav')
        pad_length = max_wav_length - wav.shape[-1]
        spectrogram_mask = (np.arange(max_wav_length) >= wav.shape[-1]).astype(np.float32)
        pad_wav = np.pad(wav, (0, max(0, pad_length)), 'constant', constant_values=0)
        _, _, spectrogram = scipy.signal.stft(pad_wav, nfft=256)
        return np.abs(spectrogram), spectrogram_mask


BATCH_SIZE = 128

datasource = DataSource(metadata_df, data_dir)
dataset = TorchDataset(datasource)

torch_sampler = DistributedSampler(
    dataset=dataset,
    num_replicas=jax.process_count(),   # == num_programs
    rank=jax.process_index(),                 # == program_index
    shuffle=True,
)

torch_loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE * jax.local_device_count(),
    sampler=torch_sampler,
    drop_last=True,
    num_workers=0
)

print('CREATE LOADER')
iter_loader = iter(torch_loader)
print('LOADER CREATED')


import flax.linen as nn


class AudioToText(nn.Module):
    output_dim: int 
    rnn_layers=5
    rnn_units=128

    @nn.compact
    def __call__(self, input_spectrogram):
        x = input_spectrogram.transpose((0, 2, 1))[..., None]
        x = nn.Conv(
            32, 
            kernel_size=(11, 41), 
            strides=(2, 2), 
            name='conv_1'
        )(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(
            32, 
            kernel_size=(11, 21),
            strides=(1, 2),
            name='conv_2'
        )(x)
        x = nn.LayerNorm()(x)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[-2] * x.shape[-1]))
        for _ in range(self.rnn_layers):
            rnn_cell = nn.GRUCell(features=self.rnn_units, gate_fn=nn.sigmoid, activation=nn.tanh)
            x = nn.Bidirectional(rnn_cell, rnn_cell)(x)

        x = nn.Dense(self.rnn_units * 2, name='dense_1')(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x



import optax

model = AudioToText(len(chars))
opt = optax.adam(learning_rate=1e-4)

print('CREATE VARIABLES')
variables = model.init(jax.random.key(0), jnp.zeros((4, 129, 1741)))
opt_state = opt.init(variables['params'])
print('VARIABLES CREATED')

from tqdm import tqdm

@jax.jit
def train_step(variables, opt_state, batch):
    
    @jax.value_and_grad
    def compute_loss(param):
        input_spectrogram = batch['spectrogram']
        logits = model.apply({'params': param}, input_spectrogram)
        loss = optax.losses.ctc_loss(logits, batch['mask_spectrogram'], batch['label'], label_paddings=batch['mask_label'])
        return loss 
    
    loss, grad = compute_loss(variables['params'])
    updates, opt_state = opt.update(grad, opt_state)
    params = optax.apply_updates(variables['params'], updates)
    variables = variables | {'params': params}
    return loss, variables, opt_state


def train_epoch(variables, opt_state, dataset, epoch, length):
    pbar = tqdm(desc=f'Epoch {epoch}', total=length)
    for _, batch in zip(range(length), dataset):
        batch = distribute_device(batch, sharding)
        loss, variables, opt_state = train_step(variables, opt_state, batch)
        pbar.set_postfix({'loss': loss})
        pbar.update(1)
    
    return variables, opt_state

def train_loop(variables, opt_state, dataset, epoches):
    datalength = len(datasource) // BATCH_SIZE 
    
    for epoch in range(epoches):
        variables, opt_state = train_epoch(variables, opt_state, dataset, epoch, datalength)
    
    return variables, opt_state

variables, opt_state = train_loop(variables, opt_state, iter_loader, 10)



