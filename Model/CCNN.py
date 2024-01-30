# Designer:Ethan Pan
# Coder:God's hand
# Time:2024/1/30 17:03
from torch import nn
import numpy as np
def complex_spectrum_features(segmented_data, FFT_PARAMS):
    sample_freq = FFT_PARAMS[0]
    time_len = FFT_PARAMS[1]
    resolution, start_freq, end_freq = 0.2930, 3, 35
    NFFT = round(sample_freq / resolution)
    fft_index_start = int(round(start_freq / resolution))
    fft_index_end = int(round(end_freq / resolution)) + 1
    sample_point = int(sample_freq * time_len)
    fft_result = np.fft.fft(segmented_data, axis=-1, n=NFFT) / (sample_point / 2)
    real_part = np.real(fft_result[:, :, :, fft_index_start:fft_index_end])
    imag_part = np.imag(fft_result[:, :, :, fft_index_start:fft_index_end])
    features_data = np.concatenate([real_part, imag_part], axis=-1)
    return features_data


class CNN(nn.Module):
      def __init__(self, Nch, Nfc, Nk):
          super(CNN, self).__init__()
          self.F = 2 * Nch
          self.D = (Nfc - 10 + 1)
          self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=self.F,
                                                kernel_size=(Nch, 1), stride=(Nch, 1)),
                                      nn.BatchNorm2d(num_features=self.F),
                                      nn.ReLU(),
                                      nn.Dropout(0.25))

          self.layer2 = nn.Sequential(nn.Conv2d(in_channels=self.F, out_channels=self.F,
                                                kernel_size=(1, 10), stride=(1, 1)),
                                      nn.BatchNorm2d(num_features=self.F),
                                      nn.ReLU(),
                                      nn.Dropout(0.25))

          self.dense_layer = nn.Sequential(nn.Flatten(),
                                           nn.Linear(self.F * self.D, Nk))


      def forward(self, X):
          X = self.layer1(X)
          out = self.layer2(X)
          # features = out.reshape(-1, self.D * self.F)
          out = self.dense_layer(out)
          # return features, out
          return out