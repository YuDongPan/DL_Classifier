# Aim & Scope
Steady state visual evoked potential(SSVEP) refers to the electrophysiological signals related to stimulus frequency that are evoked in the occipital frontal lobe region of the brain when the subject continuously fixates on flashing or flipping stimuli at a fixed frequency. Compared to P300, MI motor imagery and other EEG signals, SSVEP has a higher signal-to-noise ratio and can produce higher ITR, making it one of the most promising EEG paradigms for a long time. However, traditional signal processing algorithms rely on manual feature extraction in decoding SSVEP signals, and have poor performance in various extreme situations (short data length, large number of stimuli, and small number of calibration data), which cannot meet the practical needs of the industry. Deep learning technology, one of the main branches of artificial intelligence, has been used in research in various fields such as computer vision, natural language processing, recommendation systems, etc. due to its powerful feature expression ability and extremely high flexibility. It has overturned the algorithm design ideas in these fields and achieved remarkable results. 

In recent years, deep learning technology has gradually been favored by researchers in the field of BCI. This repository is provided for replicating the deep learning-based recognition methods of SSVEP signals. The replicated methods include EEGNet [1]-[2], C-CNN [3], FBtCNN [4], ConvCA [5], SSVEPNet [6], and SSVEPformer [7]. The file distribution follow the code desgin of <a href="https://github.com/YuDongPan/SSVEPNet">SSVEPNet</a>, and a 12-class public dataset [6] was used to conduct evaluation.

# Introduction
- EEGNet: EEGNet is a convolutional neural network model specifically designed for processing EEG signal data, which receives time-domain EEG data as network input. EEGNet consists of 4 layers. The first layer is a convolutional layer that simulates bandpass filtering for each channel. The second layer is a spatial filtering layer that weights the data from each channel, achieved through depth-wise convolution. The third layer is a separate convolutional layer for extracting category information. The fourth layer is a fully connected layer for classification. Since its proposal, EEGNet has been used in various EEG tasks, such as motor imagery, P300, SSVEP, etc [1]-[2].
  
![image](show_img/EEGNet.jpg)

- CCNN:
  
- FBtCNN:
  
- ConvCA:
  
- SSVEPNet:
  
- SSVEPformer:


# Reference
[1] Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013. <a href="https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta">https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta</a>

[2] Waytowich N, Lawhern V J, Garcia J O, et al. Compact convolutional neural networks for classification of asynchronous steady-state visual evoked potentials[J]. Journal of neural engineering, 2018, 15(6): 066031. <a href="https://iopscience.iop.org/article/10.1088/1741-2552/aae5d8/meta">https://iopscience.iop.org/article/10.1088/1741-2552/aae5d8/meta</a>

[3] Ravi A, Beni N H, Manuel J, et al. Comparing user-dependent and user-independent training of CNN for SSVEP BCI[J]. Journal of neural engineering, 2020, 17(2): 026028. <a href="https://iopscience.iop.org/article/10.1088/1741-2552/ab6a67/meta">https://iopscience.iop.org/article/10.1088/1741-2552/ab6a67/meta</a>

[4] Ding W, Shan J, Fang B, et al. Filter bank convolutional neural network for short time-window steady-state visual evoked potential classification[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2021, 29: 2615-2624. <a href="https://ieeexplore.ieee.org/abstract/document/9632600/">https://ieeexplore.ieee.org/abstract/document/9632600/</a>

[5] Li Y, Xiang J, Kesavadas T. Convolutional correlation analysis for enhancing the performance of SSVEP-based brain-computer interface[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2020, 28(12): 2681-2690. <a href="https://ieeexplore.ieee.org/abstract/document/9261605/">https://ieeexplore.ieee.org/abstract/document/9261605/</a>

[6] Pan Y, Chen J, Zhang Y, et al. An efficient CNN-LSTM network with spectral normalization and label smoothing technologies for SSVEP frequency recognition[J]. Journal of Neural Engineering, 2022, 19(5): 056014. <a href="https://iopscience.iop.org/article/10.1088/1741-2552/ac8dc5/meta">https://iopscience.iop.org/article/10.1088/1741-2552/ac8dc5/meta</a>

[7] Chen J, Zhang Y, Pan Y, et al. A Transformer-based deep neural network model for SSVEP classification[J]. Neural Networks, 2023, 164: 521-534. <a href="https://www.sciencedirect.com/science/article/abs/pii/S0893608023002319">https://www.sciencedirect.com/science/article/abs/pii/S0893608023002319</a>


