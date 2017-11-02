# Wavelet-EMD

The earth mover’s distance (EMD) is an important perceptually meaningful metric for comparing histograms,
but it suffers from high (O(N3 logN)) computational complexity. This is an implementation of novel linear time algorithm for
approximating the EMD for low dimensional histograms using the sum of absolute values of the weighted wavelet
coefficients of the difference histogram which we call as Wavelet EMD.
