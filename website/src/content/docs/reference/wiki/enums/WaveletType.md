---
title: "WaveletType"
description: "Defines the different types of biorthogonal wavelets that can be used for signal processing and analysis."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the different types of biorthogonal wavelets that can be used for signal processing and analysis.

## How It Works

**For Beginners:** Wavelets are mathematical functions that cut up data into different frequency components.

Think of wavelets like special lenses that let you look at your data in different ways:

- They can zoom in to see fine details (high frequencies)
- They can zoom out to see the big picture (low frequencies)

Biorthogonal wavelets (Bior) are a special family of wavelets that have useful properties for 
signal processing. The "Reverse" prefix indicates these are the reconstruction filters.

The numbers in each wavelet name (like "11" in ReverseBior11) represent:

- First number: Decomposition filter length
- Second number: Reconstruction filter length

Different wavelets are better suited for different types of data and applications.

## Fields

| Field | Summary |
|:-----|:--------|
| `ReverseBior11` | Reverse Biorthogonal 1.1 wavelet - the simplest biorthogonal wavelet with one vanishing moment in both decomposition and reconstruction. |
| `ReverseBior13` | Reverse Biorthogonal 1.3 wavelet - has one vanishing moment for decomposition and three for reconstruction. |
| `ReverseBior22` | Reverse Biorthogonal 2.2 wavelet - has two vanishing moments in both decomposition and reconstruction. |
| `ReverseBior24` | Reverse Biorthogonal 2.4 wavelet - has two vanishing moments for decomposition and four for reconstruction. |
| `ReverseBior26` | Reverse Biorthogonal 2.6 wavelet - has two vanishing moments for decomposition and six for reconstruction. |
| `ReverseBior28` | Reverse Biorthogonal 2.8 wavelet - has two vanishing moments for decomposition and eight for reconstruction. |
| `ReverseBior31` | Reverse Biorthogonal 3.1 wavelet - has three vanishing moments for decomposition and one for reconstruction. |
| `ReverseBior33` | Reverse Biorthogonal 3.3 wavelet - has three vanishing moments in both decomposition and reconstruction. |
| `ReverseBior35` | Reverse Biorthogonal 3.5 wavelet - has three vanishing moments for decomposition and five for reconstruction. |
| `ReverseBior37` | Reverse Biorthogonal 3.7 wavelet - has three vanishing moments for decomposition and seven for reconstruction. |
| `ReverseBior39` | Reverse Biorthogonal 3.9 wavelet - has three vanishing moments for decomposition and nine for reconstruction. |
| `ReverseBior44` | Reverse Biorthogonal 4.4 wavelet - has four vanishing moments in both decomposition and reconstruction. |
| `ReverseBior46` | Reverse Biorthogonal 4.6 wavelet - has four vanishing moments for decomposition and six for reconstruction. |
| `ReverseBior48` | Reverse Biorthogonal 4.8 wavelet - has four vanishing moments for decomposition and eight for reconstruction. |
| `ReverseBior55` | Reverse Biorthogonal 5.5 wavelet - has five vanishing moments in both decomposition and reconstruction. |
| `ReverseBior68` | Reverse Biorthogonal 6.8 wavelet - has six vanishing moments for decomposition and eight for reconstruction. |

