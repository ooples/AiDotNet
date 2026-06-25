---
title: "Decomposition Methods"
description: "All 32 public types in the AiDotNet.decompositionmethods namespace, organized by kind."
section: "API Reference"
---

**32** public types in this namespace, organized by kind.

## Models & Types (30)

| Type | Summary |
|:-----|:--------|
| [`AdditiveDecomposition<T>`](/docs/reference/wiki/decompositionmethods/additivedecomposition/) | Implements additive time series decomposition, breaking a time series into trend, seasonal, and residual components. |
| [`BeveridgeNelsonDecomposition<T>`](/docs/reference/wiki/decompositionmethods/beveridgenelsondecomposition/) | Implements the Beveridge-Nelson decomposition method for time series analysis. |
| [`BidiagonalDecomposition<T>`](/docs/reference/wiki/decompositionmethods/bidiagonaldecomposition/) | Implements the Bidiagonal Decomposition of a matrix, which factors a matrix into U*B*V^T, where U and V are orthogonal matrices and B is a bidiagonal matrix. |
| [`CholeskyDecomposition<T>`](/docs/reference/wiki/decompositionmethods/choleskydecomposition/) | Implements the Cholesky decomposition for symmetric positive definite matrices. |
| [`ComplexMatrixDecomposition<T>`](/docs/reference/wiki/decompositionmethods/complexmatrixdecomposition/) | A wrapper class that adapts a real-valued matrix decomposition to work with complex numbers. |
| [`CramerDecomposition<T>`](/docs/reference/wiki/decompositionmethods/cramerdecomposition/) | Implements Cramer's rule for solving systems of linear equations and matrix inversion. |
| [`EMDDecomposition<T>`](/docs/reference/wiki/decompositionmethods/emddecomposition/) | Implements the Empirical Mode Decomposition (EMD) method for time series decomposition. |
| [`EigenDecomposition<T>`](/docs/reference/wiki/decompositionmethods/eigendecomposition/) | Performs eigenvalue decomposition of a matrix, breaking it down into its eigenvalues and eigenvectors. |
| [`GramSchmidtDecomposition<T>`](/docs/reference/wiki/decompositionmethods/gramschmidtdecomposition/) | Implements the Gram-Schmidt orthogonalization process to decompose a matrix into an orthogonal matrix Q and an upper triangular matrix R. |
| [`HessenbergDecomposition<T>`](/docs/reference/wiki/decompositionmethods/hessenbergdecomposition/) | Implements Hessenberg decomposition, which transforms a matrix into a form that is almost triangular. |
| [`HodrickPrescottDecomposition<T>`](/docs/reference/wiki/decompositionmethods/hodrickprescottdecomposition/) | Implements the Hodrick-Prescott filter for decomposing time series data into trend and cyclical components. |
| [`IcaDecomposition<T>`](/docs/reference/wiki/decompositionmethods/icadecomposition/) | Implements Independent Component Analysis (ICA) for blind source separation. |
| [`LdlDecomposition<T>`](/docs/reference/wiki/decompositionmethods/ldldecomposition/) | Performs LDL decomposition on a symmetric matrix, factoring it into a lower triangular matrix L and a diagonal matrix D such that A = LDL^T. |
| [`LqDecomposition<T>`](/docs/reference/wiki/decompositionmethods/lqdecomposition/) | Performs LQ decomposition on a matrix, factoring it into a lower triangular matrix L and an orthogonal matrix Q. |
| [`LuDecomposition<T>`](/docs/reference/wiki/decompositionmethods/ludecomposition/) | Implements LU decomposition for matrices, which factorizes a matrix into a product of lower and upper triangular matrices. |
| [`MultiplicativeDecomposition<T>`](/docs/reference/wiki/decompositionmethods/multiplicativedecomposition/) | Performs multiplicative decomposition of time series data into trend, seasonal, and residual components. |
| [`NmfDecomposition<T>`](/docs/reference/wiki/decompositionmethods/nmfdecomposition/) | Implements Non-negative Matrix Factorization (NMF) for matrices with non-negative elements. |
| [`NormalDecomposition<T>`](/docs/reference/wiki/decompositionmethods/normaldecomposition/) | Implements the Normal Equation method for solving linear systems, especially useful for overdetermined systems. |
| [`PolarDecomposition<T>`](/docs/reference/wiki/decompositionmethods/polardecomposition/) | Implements the Polar Decomposition of a matrix, which factors a matrix A into the product of an orthogonal matrix U and a positive semi-definite matrix P. |
| [`QrDecomposition<T>`](/docs/reference/wiki/decompositionmethods/qrdecomposition/) | Performs QR decomposition on a matrix, factoring it into an orthogonal matrix Q and an upper triangular matrix R. |
| [`SEATSDecomposition<T>`](/docs/reference/wiki/decompositionmethods/seatsdecomposition/) | Implements the SEATS (Seasonal Extraction in ARIMA Time Series) decomposition method for time series data. |
| [`SSADecomposition<T>`](/docs/reference/wiki/decompositionmethods/ssadecomposition/) | Implements Singular Spectrum Analysis (SSA) for time series decomposition. |
| [`STLTimeSeriesDecomposition<T>`](/docs/reference/wiki/decompositionmethods/stltimeseriesdecomposition/) | Implements the Seasonal-Trend decomposition using LOESS (STL) algorithm for time series analysis. |
| [`SchurDecomposition<T>`](/docs/reference/wiki/decompositionmethods/schurdecomposition/) | Performs Schur decomposition on a matrix, factoring it into the product of a unitary matrix and an upper triangular matrix. |
| [`SvdDecomposition<T>`](/docs/reference/wiki/decompositionmethods/svddecomposition/) | Implements Singular Value Decomposition (SVD) for matrices. |
| [`TakagiDecomposition<T>`](/docs/reference/wiki/decompositionmethods/takagidecomposition/) | Implements the Takagi factorization for complex symmetric matrices. |
| [`TridiagonalDecomposition<T>`](/docs/reference/wiki/decompositionmethods/tridiagonaldecomposition/) | Represents a tridiagonal decomposition of a matrix, which decomposes a matrix A into Q*T*Q^T, where Q is orthogonal and T is tridiagonal. |
| [`UduDecomposition<T>`](/docs/reference/wiki/decompositionmethods/ududecomposition/) | Represents a UDU' decomposition of a matrix, which factorizes a symmetric matrix A into U*D*U', where U is an upper triangular matrix with ones on the diagonal, D is a diagonal matrix, and U' is the transpose of U. |
| [`WaveletDecomposition<T>`](/docs/reference/wiki/decompositionmethods/waveletdecomposition/) | Implements wavelet-based decomposition methods for time series data. |
| [`X11Decomposition<T>`](/docs/reference/wiki/decompositionmethods/x11decomposition/) | Implements the X-11 method for time series decomposition, which breaks down a time series into trend, seasonal, and irregular components. |

## Base Classes (2)

| Type | Summary |
|:-----|:--------|
| [`MatrixDecompositionBase<T>`](/docs/reference/wiki/decompositionmethods/matrixdecompositionbase/) | Base class for matrix decomposition algorithms that break down matrices into simpler components. |
| [`TimeSeriesDecompositionBase<T>`](/docs/reference/wiki/decompositionmethods/timeseriesdecompositionbase/) | Base class for time series decomposition algorithms that break down time series data into component parts. |

