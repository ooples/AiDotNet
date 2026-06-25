---
title: "Wavelet Functions"
description: "All 22 public types in the AiDotNet.waveletfunctions namespace, organized by kind."
section: "API Reference"
---

**22** public types in this namespace, organized by kind.

## Models & Types (20)

| Type | Summary |
|:-----|:--------|
| [`BSplineWavelet<T>`](/docs/reference/wiki/waveletfunctions/bsplinewavelet/) | Implements a B-spline wavelet, which is a smooth wavelet constructed from B-spline functions. |
| [`BattleLemarieWavelet<T>`](/docs/reference/wiki/waveletfunctions/battlelemariewavelet/) | Implements the Battle-Lemarie wavelet function, which is a smooth, orthogonal wavelet based on B-splines. |
| [`BiorthogonalWavelet<T>`](/docs/reference/wiki/waveletfunctions/biorthogonalwavelet/) | Implements biorthogonal wavelets, which offer symmetry and linear phase properties while maintaining perfect reconstruction capabilities. |
| [`CoifletWavelet<T>`](/docs/reference/wiki/waveletfunctions/coifletwavelet/) | Implements Coiflet wavelets, which are compactly supported wavelets with a high number of vanishing moments for both the wavelet and scaling functions. |
| [`ComplexGaussianWavelet<T>`](/docs/reference/wiki/waveletfunctions/complexgaussianwavelet/) | Implements a Complex Gaussian wavelet, which is based on the derivative of a Gaussian function and can handle complex-valued signals. |
| [`ComplexMorletWavelet<T>`](/docs/reference/wiki/waveletfunctions/complexmorletwavelet/) | Implements a Complex Morlet wavelet, which is a complex exponential modulated by a Gaussian window, making it well-suited for time-frequency analysis of signals. |
| [`ContinuousMexicanHatWavelet<T>`](/docs/reference/wiki/waveletfunctions/continuousmexicanhatwavelet/) | Implements the Mexican Hat wavelet (also known as the Ricker wavelet or the second derivative of a Gaussian), which is commonly used for continuous wavelet transforms and feature detection. |
| [`DOGWavelet<T>`](/docs/reference/wiki/waveletfunctions/dogwavelet/) | Implements the Derivative of Gaussian (DOG) wavelet, which is based on the nth derivative of the Gaussian function and is useful for detecting changes in signals. |
| [`DaubechiesWavelet<T>`](/docs/reference/wiki/waveletfunctions/daubechieswavelet/) | Implements Daubechies wavelets, which are a family of orthogonal wavelets characterized by a maximal number of vanishing moments for a given support width. |
| [`FejérKorovkinWavelet<T>`](/docs/reference/wiki/waveletfunctions/fejrkorovkinwavelet/) | Represents a Fejér-Korovkin wavelet function implementation for signal processing and analysis. |
| [`GaborWavelet<T>`](/docs/reference/wiki/waveletfunctions/gaborwavelet/) | Represents a Gabor wavelet function implementation for time-frequency analysis and signal processing. |
| [`GaussianWavelet<T>`](/docs/reference/wiki/waveletfunctions/gaussianwavelet/) | Represents a Gaussian wavelet function implementation for signal processing and analysis. |
| [`HaarWavelet<T>`](/docs/reference/wiki/waveletfunctions/haarwavelet/) | Represents a Haar wavelet function implementation for signal processing and analysis. |
| [`MexicanHatWavelet<T>`](/docs/reference/wiki/waveletfunctions/mexicanhatwavelet/) | Represents a Mexican Hat wavelet function implementation for signal processing and analysis. |
| [`MeyerWavelet<T>`](/docs/reference/wiki/waveletfunctions/meyerwavelet/) | Represents a Meyer wavelet function implementation for frequency domain analysis and signal processing. |
| [`MorletWavelet<T>`](/docs/reference/wiki/waveletfunctions/morletwavelet/) | Represents a Morlet wavelet function implementation for time-frequency analysis and signal processing. |
| [`PaulWavelet<T>`](/docs/reference/wiki/waveletfunctions/paulwavelet/) | Represents a Paul wavelet function implementation for complex signal analysis and processing. |
| [`ReverseBiorthogonalWavelet<T>`](/docs/reference/wiki/waveletfunctions/reversebiorthogonalwavelet/) | Represents a Reverse Biorthogonal wavelet function implementation for signal processing and analysis. |
| [`ShannonWavelet<T>`](/docs/reference/wiki/waveletfunctions/shannonwavelet/) | Represents a Shannon wavelet function implementation for signal processing and frequency analysis. |
| [`SymletWavelet<T>`](/docs/reference/wiki/waveletfunctions/symletwavelet/) | Implements Symlet wavelets, which are nearly symmetric wavelets proposed by Daubechies as modifications to the Daubechies wavelets with increased symmetry. |

## Base Classes (2)

| Type | Summary |
|:-----|:--------|
| [`ComplexWaveletFunctionBase<T>`](/docs/reference/wiki/waveletfunctions/complexwaveletfunctionbase/) | Base class for all complex wavelet function implementations providing common functionality. |
| [`WaveletFunctionBase<T>`](/docs/reference/wiki/waveletfunctions/waveletfunctionbase/) | Base class for all wavelet function implementations providing common functionality. |

