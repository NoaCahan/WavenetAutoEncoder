from audio_func import mu_law_decode, mu_law_encode
from collections import OrderedDict
from train import load_model
import json
import numpy as np 
from torch.autograd import Variable
import torch.nn.functional as F 
import torch
import os
from model import WavenetAutoencoder
import librosa
#from faster_audio_data import one_hot_encode

def predict_next(net,input_wav,quantization_channel = 256):
    out_wav = net(input_wav)
    out = out_wav.view(-1,quantization_channel)
    last = out[-1,:]
    last = last.view(-1)
    _,predict = torch.topk(last, 1)
    return int(predict)

def generate(model_path,model_name,generate_path,generate_name,sr = 16000,duration=10):
    if os.path.exists(generate_path) is False:
        os.makedirs(generate_path)
    with open('./params/model_params.json') as f :
        model_params = json.load(f)
    f.close()
    net = WavenetAutoencoder(**model_params)
    net = load_model(net,model_path,model_name)
    cuda_available = torch.cuda.is_available()
    if cuda_available is True:
        net = net.cuda()
    net.eval()

    start_piece = torch.zeros(1, 256, net.receptive_field+512)
    start_piece[:, 128, :] = 1.0
    start_piece = Variable(start_piece, volatile=True)
    
    if cuda_available is True:
        start_piece = start_piece.cuda()
    note_num = duration * sr
    note = start_piece
    generated_piece = []
    input_wav = start_piece
    

    for i in range(note_num):
        print(i)
        predict_note = predict_next(net, input_wav)
        generated_piece.append(note)
        temp = torch.zeros(net.quantization_channel,1)
        temp[predict_note] =1
        temp = temp.view(1,net.quantization_channel,1)
        #		temp = torch.zeros(1, net.quantization_channel, 1)
        #		temp[:, predict_note, :] = 1.0
        note = Variable(temp)
        if cuda_available is True:
            note = note.cuda()
        #		print(note.size())
        #		print(input_wav.size())
        input_wav = torch.cat((input_wav[:,-net.receptive_field-511:],note), 2)
    print(generated_piece)
    generated_piece = torch.LongTensor(generated_piece)
    generated_piece = mu_law_decode(generated_piece,
                            net.quantization_channel)
    generated_piece = generated_piece.numpy()
    wav_name = generate_path + generate_name
    librosa.output.write_wav(wav_name, generated_piece, sr=sr)

def decode_next(net,input_wav ,encoding,output_width ,quantization_channel = 256):
    out_wav = net.decoder(input_wav,encoding,output_width)
    out = out_wav.view(-1,quantization_channel)
    last = out[-1,:]
    last = last.view(-1)
    _,predict = torch.topk(last, 1)
    return int(predict)
    
def encode(model_path,model_name,encoding_path,encoding_name, piece, sr = 16000,duration=10):
    if os.path.exists(encoding_path) is False:
        os.makedirs(encoding_path)
    with open('./params/model_params.json') as f :
        model_params = json.load(f)
    f.close()
    net = WavenetAutoencoder(**model_params)
    net = load_model(net,model_path,model_name)
    cuda_available = torch.cuda.is_available()
    if cuda_available is True:
        net = net.cuda()
    net.eval()
    
    audio = librosa.load(piece, sr=16000, mono=True)
    audio = audio[0]
    length = len(audio)
    audio = torch.from_numpy(audio)
    audio = mu_law_encode(audio)

    piece_one_hot = np.zeros((length, net.quantization_channel))
    piece_one_hot[np.arange(length), audio.numpy()] = 1.0
    piece_one_hot = piece_one_hot.reshape(1,net.quantization_channel, length)
    start_piece = torch.FloatTensor(piece_one_hot)

    start_piece = Variable(start_piece, volatile=True)
    print(start_piece.data.size())

    if cuda_available is True:
        start_piece = start_piece.cuda()
    input_wav = start_piece
    piece_encoding = net.encoder(input_wav)
    
    print(piece_encoding.data.size())
    piece_encoding = piece_encoding.data.cpu().numpy()
    encoding_ndarray = encoding_path + encoding_name+ '.npy'
    np.save(encoding_ndarray, piece_encoding)


def sample_categorical(pmf):

    """Sample from a categorical distribution.
    Args:
    pmf: Probablity mass function. Output of a softmax over categories.
      Array of shape [batch_size, number of categories]. Rows sum to 1.
    Returns:
    idxs: Array of size [batch_size, 1]. Integer of category sampled.

    Code from: 

    https://github.com/tensorflow/magenta/blob/master/magenta/models/nsynth/wavenet/fastgen.py

    """
    batch_size = 1
    print(pmf.ndim)
    if pmf.ndim == 1:
        pmf = np.expand_dims(pmf, 0)
        batch_size = pmf.shape[0]
        cdf = np.cumsum(pmf, axis=1)
        rand_vals = np.random.rand(batch_size)
        idxs = np.zeros([batch_size, 1])
    for i in range(batch_size):
        idxs[i] = cdf[i].searchsorted(rand_vals[i])
    return idxs

def decode1(model_path,model_name, encoding, decoder_path, decoder_name, sr = 16000,duration=10):
    
    """Synthesize audio from an array of embeddings.
    
    Args:
    encodings: Numpy array with shape [batch_size, time, dim].
    save_paths: Iterable of output file names.
    checkpoint_path: Location of the pretrained model. [model.ckpt-200000]
    samples_per_save: Save files after every amount of generated samples.

    """
    if os.path.exists(decoder_path) is False:
        os.makedirs(decoder_path)
    with open('./params/model_params.json') as f :
        model_params = json.load(f)
    f.close()
    net = WavenetAutoencoder(**model_params)
    net = load_model(net,model_path,model_name)
    cuda_available = torch.cuda.is_available()
    if cuda_available is True:
        net = net.cuda()
    net.eval()

    start_piece = torch.zeros(1, 256, net.receptive_field+512)
    start_piece[:, 128, :] = 1.0
    start_piece = Variable(start_piece, volatile=True)
    
    # Load Encoding
    encoding_ndarray = np.load(encoding)
    _, encoding_channels, encoding_length = encoding_ndarray.shape
    hop_size = model_params['en_pool_kernel_size']
    total_length = encoding_length * hop_size
    batch_size = 1

    encoding = torch.from_numpy(encoding_ndarray).contiguous()
    encoding = Variable(encoding, volatile=True)
    #generated_piece = np.zeros(( batch_size, total_length,), dtype=np.float32)
    #audio = np.zeros([batch_size, 1])
    
    if cuda_available is True:
        start_piece = start_piece.cuda()
    note = start_piece
    generated_piece = []
    
    input_wav = start_piece
    generated_piece = np.zeros(( batch_size, total_length,), dtype=np.float32)
    audio = np.zeros([batch_size, 1])
    # Should be torch.Size([1, 256, 4606])
    #print(input_wav.data.size())
    
    for i in range(total_length):
        print(i)
        #enc_i = encoding[:, :, i]
        #predict_note = decode_next(net, input_wav, enc_i, output_width)
        #temp = torch.zeros(net.quantization_channel,1)
        #temp[predict_note] =1
        #temp = temp.view(1,net.quantization_channel,1)
        #note = Variable(temp, volatile=True)
        
        enc_i = encoding[:, :, i].unsqueeze(2)
        #pmf = net.decoder(input_wav, enc_i, output_width)
        #sample_bin = sample_categorical(pmf.data.cpu().numpy())
        
        sample_bin = decode_next(net, audio, enc_i, hop_size)
        print(sample_bin)
        audio = mu_law_decode(sample_bin)
        #print(audio.data.size())
        generated_piece[:, i] = audio[:, 0]
        #if cuda_available is True:
        #    note = note.cuda()
        #input_wav = torch.cat((input_wav[:,-net.receptive_field-511:],note), 2)
        
    print(generated_piece.data.size())
    generated_piece = torch.LongTensor(generated_piece)
    generated_piece = mu_law_decode(generated_piece, net.quantization_channel)
    generated_piece = generated_piece.numpy()
    wav_name = generate_path + generate_name
    librosa.output.write_wav(wav_name, generated_piece, sr=sr)
    
    #for sample_i in range(total_length):
    #    enc_i = sample_i // hop_length
    #    pmf = sess.run(
    #      [net["predictions"], net["push_ops"]],
    #      feed_dict={
    #          net["X"]: audio,
    #          net["encoding"]: encodings[:, enc_i, :]
    #      })[0]
    #    sample_bin = sample_categorical(pmf)
    #    audio = utils.inv_mu_law_numpy(sample_bin - 128)
    #    audio_batch[:, sample_i] = audio[:, 0]
    #    if sample_i % 100 == 0:
    #        tf.logging.info("Sample: %d" % sample_i)
    #    if sample_i % samples_per_save == 0:
    #        save_batch(audio_batch, save_paths)
    #save_batch(audio_batch, save_paths)
    

def decode(model_path,model_name, encoding, decoder_path, decoder_name, sr = 16000,duration=10):
    
    """Synthesize audio from an array of embeddings.
    
    Args:
    encodings: Numpy array with shape [batch_size, time, dim].
    save_paths: Iterable of output file names.
    checkpoint_path: Location of the pretrained model. [model.ckpt-200000]
    samples_per_save: Save files after every amount of generated samples.

    """
    
    if os.path.exists(decoder_path) is False:
        os.makedirs(decoder_path)
    with open('./params/model_params.json') as f :
        model_params = json.load(f)
    f.close()
    net = WavenetAutoencoder(**model_params)
    net = load_model(net,model_path,model_name)
    cuda_available = torch.cuda.is_available()
    if cuda_available is True:
        net = net.cuda()
    net.eval()

    start_piece = torch.zeros(1, 256, net.receptive_field+512)
    start_piece[:, 128, :] = 1.0
    start_piece = Variable(start_piece, volatile=True)
    
    # Load Encoding
    encoding_ndarray = np.load(encoding)
    _, encoding_channels, encoding_length = encoding_ndarray.shape
    hop_size = model_params['en_pool_kernel_size']
    total_length = encoding_length * hop_size

    encoding = torch.from_numpy(encoding_ndarray).contiguous()
    encoding = Variable(encoding, volatile=True)
    #generated_piece = np.zeros(( batch_size, total_length,), dtype=np.float32)
    #audio = np.zeros([batch_size, 1])
    
    if cuda_available is True:
        start_piece = start_piece.cuda()
    note = start_piece
    generated_piece = []
    input_wav = start_piece
    
    # Should be torch.Size([1, 256, 4606])
    #print(input_wav.data.size())
    
    for i in range(total_length):
        print(i)
        enc_i = encoding[:, :, i].unsqueeze(2)
        predict_note = decode_next(net, input_wav, enc_i, hop_size)
        generated_piece.append(note)
        temp = torch.zeros(net.quantization_channel,1)
        temp[predict_note] =1
        temp = temp.view(1,net.quantization_channel,1)
        #		temp = torch.zeros(1, net.quantization_channel, 1)
        #		temp[:, predict_note, :] = 1.0
        note = Variable(temp)
        if cuda_available is True:
            note = note.cuda()
        #		print(note.size())
        print(input_wav.data.size())
        input_wav = torch.cat((input_wav[:,-net.receptive_field-511:],note), 2)
    print(generated_piece)
    generated_piece = torch.LongTensor(generated_piece)
    generated_piece = mu_law_decode(generated_piece,
                            net.quantization_channel)
    generated_piece = generated_piece.numpy()
    wav_name = generate_path + generate_name
    librosa.output.write_wav(wav_name, generated_piece, sr=sr)
    
    
if __name__ =='__main__':
    # Generate seems to be working
    #generate('./restore/', 'wavenet_autoencoder1.model','./generate/','generate_noa' , './samples/noa.wav')
    #generate('./restore/', 'wavenet_autoencoder_cpu.model','./generate/','generate' )

    # Encoding dies in the middle
    #encode('./restore/', 'wavenet_autoencoder_cpu.model','./encoding/','encoding_noa', './samples/noa.wav')
    decode('./restore/' ,'wavenet_autoencoder_cpu.model','./encoding/encoding_noa.npy', './decoding/','decoding_noa' )
    decode1('./restore/','wavenet_autoencoder_cpu.model','./encoding/encoding_noa.npy', './decoding/','decoding_noa' )
