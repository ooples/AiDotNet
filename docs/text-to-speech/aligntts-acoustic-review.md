# AlignTTS acoustic integration review

The primary source is [Zeng et al., AlignTTS](https://arxiv.org/abs/2003.01950), including its four training phases. `aligntts_acoustic_reference.py` executes the full reduced-width acoustic graph with independent PyTorch operators: character encoder, three-hidden-layer Gaussian density network, length regulation, mel decoder, and separately embedded Transformer duration predictor.

Every positive-duration monotonic alignment is explicitly enumerated for four characters and six frames. Log-sum-exp of those path scores supplies the sequence negative log likelihood; the maximum-scoring path supplies durations. Neither the native alignment dynamic program nor Coqui's modified AlignTTS loss is used as the oracle.

`AlignTtsAcousticReferenceTests` compares the actual production graph's encoder output, every Gaussian likelihood, exact Viterbi durations, decoder mel, predicted log duration, all four phase objectives, and every connected parameter gradient. All 59 native learned tensors are accounted for; disconnected parameters must have zero expected gradients. The phase-owned subsets are checked separately: 28 tensors for alignment, 15 for decoder training, 43 for joint fine-tuning, and 16 for duration training. Every comparison passes at 1e-9 against Build41.

The graph uses reduced widths and one FFT block per stack. Three density hidden layers are selected to exercise configurable stacking; the paper does not disclose their count and the native default is two. This establishes arithmetic, gradient, and training-phase integration; it is not an original author-checkpoint or corpus convergence result. Existing tests separately verify published defaults, phase updates, and strict native converted-checkpoint loading.
