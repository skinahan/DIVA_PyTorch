% Read in the audio files produced by each model
diva_aud = audioread('DIVA_HAPPY_TRAINED_OUT.wav');
torch_diva_aud = audioread('1661288163_TORCHDIVA_OUT.wav');
% Trim trailing silence from the TorchDIVA production
torch_diva_aud = torch_diva_aud(1:6119,:);
% Calculate the difference in the two signals
aud_diff = diva_aud - torch_diva_aud;
% Plot the spectrograms
subplot(3,1,1);
spectrogram(diva_aud);
title('DIVA - Happy');
subplot(3,1,2);
spectrogram(torch_diva_aud);
title('TorchDIVA - Happy');
subplot(3,1,3);
spectrogram(aud_diff);
title('Difference - Happy');