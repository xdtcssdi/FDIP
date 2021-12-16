import torch.nn as nn
import  torch

class Encoder(nn.Module):
    def __init__(self,  input_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.gru = nn.GRU(input_dim, enc_units,bidirectional=False)

    def forward(self, enc_input):
        # x: batch_size,seq_len,input_dim
        self.batch_sz = enc_input.shape[0]
        # x transformed = seq_len X batch_size X input_dim
        enc_input = enc_input.permute(1, 0, 2)
        self.hidden = self.initialize_hidden_state()

        # output: seq_len, batch, num_directions * enc_units
        # self.hidden: num_layers * num_directions, batch, enc_units
        output, self.hidden = self.gru(enc_input, self.hidden) # gru returns hidden state of all timesteps as well as hidden state at last timestep

        # outputs are always from the top hidden layer. self.hidden is hidden state for t = seq_len
        return output, self.hidden

    def initialize_hidden_state(self):
        # h_0  of shape(num_layers * num_directions, batch, enc_units)
        return torch.zeros((1*1 , self.batch_sz, self.enc_units)).cuda()


class Decoder(nn.Module):
    def __init__(self, output_dim, dec_units, enc_units):
        super(Decoder, self).__init__()

        self.dec_units = dec_units
        self.enc_units = enc_units

        self.output_dim = output_dim

        self.gru = nn.GRU(self.output_dim + self.enc_units,self.dec_units,batch_first=True)
        self.fc = nn.Linear(self.dec_units, self.output_dim)

        #dec_units, enc_units : hidden units for encoder and decoder
        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.dec_units, 1)

    def forward(self, dec_input, enc_hidden, enc_output):
        # enc_output original: seq_len, batch, num_directions * hidden_size
        # enc_output converted == batch, seq_len, num_directions * hidden_size
        enc_output = enc_output.permute(1, 0, 2)
        self.batch_sz = enc_output.shape[0]
        # hidden shape == num_layers * num_directions, batch, hidden_size
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = enc_hidden.permute(1, 0, 2)

        # score: (batch_size, seq_len, hidden_size) # Bahdanaus's
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        # It doesn't matter which FC we pick for each of the inputs
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, seq_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)

        # context_vector shape after sum == (batch_size, enc_units)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        # dec_input shape == (batch_size, 1, output_dim)

        # dec_input shape after concatenation == (batch_size, 1, output_dim + enc_units)
        dec_input = torch.cat((context_vector.unsqueeze(1), dec_input), -1)

        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, dec_units)
        output, state = self.gru(dec_input)

        # output shape == (batch_size * 1, dec_units)
        output = output.view(-1, output.size(2))

        # output shape == (batch_size * 1, output_dim)
        x = self.fc(output)

        return x, state

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.dec_units))