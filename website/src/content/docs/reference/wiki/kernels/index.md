---
title: "Kernels"
description: "All 60 public types in the AiDotNet.kernels namespace, organized by kind."
section: "API Reference"
---

**60** public types in this namespace, organized by kind.

## Models & Types (57)

| Type | Summary |
|:-----|:--------|
| [`ANOVAKernel<T>`](/docs/reference/wiki/kernels/anovakernel/) | Implements the ANOVA (Analysis of Variance) kernel function for measuring similarity between data points. |
| [`ARDKernel<T>`](/docs/reference/wiki/kernels/ardkernel/) | Implements the Automatic Relevance Determination (ARD) kernel with per-dimension length scales. |
| [`AdditiveChiSquaredKernel<T>`](/docs/reference/wiki/kernels/additivechisquaredkernel/) | Implements the Additive Chi-Squared kernel function for measuring similarity between data points. |
| [`AdditiveStructureKernel<T>`](/docs/reference/wiki/kernels/additivestructurekernel/) | Additive Structure Kernel that decomposes the function into additive components. |
| [`ArcKernel<T>`](/docs/reference/wiki/kernels/arckernel/) | Arc (Angular) Kernel based on the angle between vectors. |
| [`BSplineKernel<T>`](/docs/reference/wiki/kernels/bsplinekernel/) | Implements the B-Spline kernel function for measuring similarity between data points. |
| [`BesselKernel<T>`](/docs/reference/wiki/kernels/besselkernel/) | Implements the Bessel kernel function for measuring similarity between data points. |
| [`CauchyKernel<T>`](/docs/reference/wiki/kernels/cauchykernel/) | Implements the Cauchy kernel function for measuring similarity between data points. |
| [`ChiSquareKernel<T>`](/docs/reference/wiki/kernels/chisquarekernel/) | Implements the Chi-Square kernel function for measuring similarity between data points. |
| [`CircularKernel<T>`](/docs/reference/wiki/kernels/circularkernel/) | Implements the Circular kernel function for measuring similarity between data points. |
| [`ConstantKernel<T>`](/docs/reference/wiki/kernels/constantkernel/) | Implements the Constant kernel, which returns a constant value regardless of input. |
| [`CosineKernel<T>`](/docs/reference/wiki/kernels/cosinekernel/) | Cosine Similarity Kernel that measures angular distance between vectors. |
| [`CylindricalKernel<T>`](/docs/reference/wiki/kernels/cylindricalkernel/) | Cylindrical Kernel for Bayesian optimization with periodic/angular dimensions. |
| [`DeepNeuralNetworkKernel<T>`](/docs/reference/wiki/kernels/deepneuralnetworkkernel/) | Implements a deep (multi-layer) Neural Network kernel. |
| [`DotProductKernel<T>`](/docs/reference/wiki/kernels/dotproductkernel/) | Implements the Dot Product (Linear) kernel with optional inhomogeneity. |
| [`ExpSineSquaredKernel<T>`](/docs/reference/wiki/kernels/expsinesquaredkernel/) | Implements the Exp-Sine-Squared (Periodic) kernel for modeling repeating patterns. |
| [`ExponentialKernel<T>`](/docs/reference/wiki/kernels/exponentialkernel/) | Implements the Exponential kernel function for measuring similarity between data points. |
| [`GaussianKernel<T>`](/docs/reference/wiki/kernels/gaussiankernel/) | Implements the Gaussian (Radial Basis Function) kernel for measuring similarity between data points. |
| [`GeneralizedHistogramIntersectionKernel<T>`](/docs/reference/wiki/kernels/generalizedhistogramintersectionkernel/) | Implements the Generalized Histogram Intersection kernel for measuring similarity between data points. |
| [`GeneralizedTStudentKernel<T>`](/docs/reference/wiki/kernels/generalizedtstudentkernel/) | Implements the Generalized T-Student kernel for measuring similarity between data points. |
| [`GibbsKernel<T>`](/docs/reference/wiki/kernels/gibbskernel/) | Implements the Gibbs kernel with input-dependent length scales for non-stationary covariance. |
| [`GradientKernel<T>`](/docs/reference/wiki/kernels/gradientkernel/) | Kernel that incorporates gradient observations for GPs with derivative information. |
| [`GridInterpolationKernel<T>`](/docs/reference/wiki/kernels/gridinterpolationkernel/) | Grid Interpolation Kernel (KISS-GP) for scalable Gaussian Process inference. |
| [`GridKernel<T>`](/docs/reference/wiki/kernels/gridkernel/) | Grid Kernel for exploiting Kronecker structure in regularly-spaced data. |
| [`HellingerKernel<T>`](/docs/reference/wiki/kernels/hellingerkernel/) | Implements the Hellinger kernel for measuring similarity between probability distributions. |
| [`HistogramIntersectionKernel<T>`](/docs/reference/wiki/kernels/histogramintersectionkernel/) | Implements the Histogram Intersection kernel for measuring similarity between data points. |
| [`IndexKernel<T>`](/docs/reference/wiki/kernels/indexkernel/) | Index kernel for multi-task/multi-output Gaussian Processes. |
| [`InducingPointKernel<T>`](/docs/reference/wiki/kernels/inducingpointkernel/) | Inducing Point Kernel for sparse Gaussian Process approximations. |
| [`InverseMultiquadricKernel<T>`](/docs/reference/wiki/kernels/inversemultiquadrickernel/) | Implements the Inverse Multiquadric kernel for measuring similarity between data points. |
| [`LCMKernel<T>`](/docs/reference/wiki/kernels/lcmkernel/) | Linear Coregionalization Model (LCM) kernel for multi-output Gaussian Processes. |
| [`LaplacianKernel<T>`](/docs/reference/wiki/kernels/laplaciankernel/) | Implements the Laplacian kernel for measuring similarity between data points. |
| [`LinearKernel<T>`](/docs/reference/wiki/kernels/linearkernel/) | Implements the Linear kernel for measuring similarity between data points. |
| [`LocallyPeriodicKernel<T>`](/docs/reference/wiki/kernels/locallyperiodickernel/) | Implements the Locally Periodic kernel for measuring similarity between data points with periodic patterns. |
| [`LogKernel<T>`](/docs/reference/wiki/kernels/logkernel/) | Implements the Log kernel for measuring similarity between data points. |
| [`MaternKernel<T>`](/docs/reference/wiki/kernels/maternkernel/) | Implements the Matérn family of kernels with configurable smoothness parameter. |
| [`MultiquadricKernel<T>`](/docs/reference/wiki/kernels/multiquadrickernel/) | Implements the Multiquadric kernel for measuring similarity between data points. |
| [`NeuralNetworkKernel<T>`](/docs/reference/wiki/kernels/neuralnetworkkernel/) | Implements the Neural Network (Arc-Cosine) kernel that corresponds to infinitely wide neural networks. |
| [`PiecewisePolynomialKernel<T>`](/docs/reference/wiki/kernels/piecewisepolynomialkernel/) | Implements the Piecewise Polynomial kernel for measuring similarity between data points. |
| [`PolynomialKernel<T>`](/docs/reference/wiki/kernels/polynomialkernel/) | Implements the Polynomial kernel for measuring similarity between data points. |
| [`PowerKernel<T>`](/docs/reference/wiki/kernels/powerkernel/) | Implements the Power kernel for measuring dissimilarity between data points. |
| [`ProbabilisticKernel<T>`](/docs/reference/wiki/kernels/probabilistickernel/) | Implements the Probabilistic kernel for measuring similarity between data points. |
| [`ProductKernel<T>`](/docs/reference/wiki/kernels/productkernel/) | Implements a Product kernel that combines multiple kernels by multiplying their outputs. |
| [`ProductStructureKernel<T>`](/docs/reference/wiki/kernels/productstructurekernel/) | Product Structure Kernel for modeling multiplicative interactions between feature groups. |
| [`RFFKernel<T>`](/docs/reference/wiki/kernels/rffkernel/) | Random Fourier Features (RFF) kernel for scalable approximation of shift-invariant kernels. |
| [`RationalQuadraticKernel<T>`](/docs/reference/wiki/kernels/rationalquadratickernel/) | Implements the Rational Quadratic kernel, equivalent to an infinite mixture of RBF kernels. |
| [`ScaleKernel<T>`](/docs/reference/wiki/kernels/scalekernel/) | A wrapper kernel that scales another kernel by a constant factor (output scale/variance). |
| [`SigmoidKernel<T>`](/docs/reference/wiki/kernels/sigmoidkernel/) | Implements the Sigmoid kernel for measuring similarity between data points. |
| [`SpectralDeltaKernel<T>`](/docs/reference/wiki/kernels/spectraldeltakernel/) | Spectral Delta Kernel representing a single spectral component. |
| [`SpectralMixtureKernel<T>`](/docs/reference/wiki/kernels/spectralmixturekernel/) | Implements the Spectral Mixture (SM) kernel for discovering and exploiting patterns in data. |
| [`SphericalKernel<T>`](/docs/reference/wiki/kernels/sphericalkernel/) | Implements the Spherical kernel for measuring similarity between data points. |
| [`SplineKernel<T>`](/docs/reference/wiki/kernels/splinekernel/) | Implements the Spline kernel for measuring similarity between data points. |
| [`StringKernel<T>`](/docs/reference/wiki/kernels/stringkernel/) | Implements various string kernels for comparing text/sequence data in Gaussian Processes. |
| [`SumKernel<T>`](/docs/reference/wiki/kernels/sumkernel/) | Implements a Sum kernel that combines multiple kernels by adding their outputs. |
| [`TanimotoKernel<T>`](/docs/reference/wiki/kernels/tanimotokernel/) | Implements the Tanimoto kernel (also known as the Jaccard kernel) for measuring similarity between data points. |
| [`WaveKernel<T>`](/docs/reference/wiki/kernels/wavekernel/) | Implements the Wave kernel for measuring similarity between data points. |
| [`WaveletKernel<T>`](/docs/reference/wiki/kernels/waveletkernel/) | Implements the Wavelet kernel for measuring similarity between data points using wavelet functions. |
| [`WhiteNoiseKernel<T>`](/docs/reference/wiki/kernels/whitenoisekernel/) | Implements the White Noise kernel, which adds independent noise to each observation. |

## Enums (3)

| Type | Summary |
|:-----|:--------|
| [`GradientKernelType<T>`](/docs/reference/wiki/kernels/gradientkerneltype/) | Supported base kernel types for gradient computation. |
| [`KernelType<T>`](/docs/reference/wiki/kernels/kerneltype/) | The type of string kernel to use. |
| [`RFFKernelType<T>`](/docs/reference/wiki/kernels/rffkerneltype/) | Types of kernels that can be approximated with RFF. |

