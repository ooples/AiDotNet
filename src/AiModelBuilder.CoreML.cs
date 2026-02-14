using AiDotNet.Clustering.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;

namespace AiDotNet;

/// <summary>
/// Core machine learning component configuration extensions for AiModelBuilder.
/// Provides Configure methods for fundamental ML building blocks: loss functions,
/// activation functions, kernels, regression, classification, clustering, layers,
/// anomaly detection, interpolation, window/wavelet functions, learning rate schedulers,
/// link functions, matrix decompositions, and Gaussian processes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private ILossFunction<T>? _configuredLossFunction;
    private IActivationFunction<T>? _configuredActivationFunction;
    private IKernelFunction<T>? _configuredKernelFunction;
    private IRegression<T>? _configuredRegression;
    private IClassifier<T>? _configuredClassifier;
    private IClustering<T>? _configuredClustering;
    private ILayer<T>? _configuredLayer;
    private IAnomalyDetector<T>? _configuredAnomalyDetector;
    private IInterpolation<T>? _configuredInterpolation;
    private I2DInterpolation<T>? _configured2DInterpolation;
    private IWindowFunction<T>? _configuredWindowFunction;
    private IWaveletFunction<T>? _configuredWaveletFunction;
    private ILearningRateScheduler? _configuredLearningRateScheduler;
    private ILinkFunction<T>? _configuredLinkFunction;
    private IMatrixDecomposition<T>? _configuredMatrixDecomposition;
    private IGaussianProcess<T>? _configuredGaussianProcess;

    /// <summary>
    /// Configures the loss function used to measure prediction error during training.
    /// </summary>
    /// <param name="lossFunction">The loss function implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A loss function measures how far off your model's predictions
    /// are from the correct answers. Lower loss means better predictions. Different loss functions
    /// are suited for different tasks (e.g., MSE for regression, CrossEntropy for classification).</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureLossFunction(ILossFunction<T> lossFunction)
    {
        _configuredLossFunction = lossFunction;
        return this;
    }

    /// <summary>
    /// Configures the activation function for neural network layers.
    /// </summary>
    /// <param name="activationFunction">The activation function implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Activation functions add non-linearity to neural networks,
    /// enabling them to learn complex patterns. Common choices include ReLU (fast, general-purpose),
    /// Sigmoid (0-1 output), and Tanh (-1 to 1 output).</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureActivationFunction(IActivationFunction<T> activationFunction)
    {
        _configuredActivationFunction = activationFunction;
        return this;
    }

    /// <summary>
    /// Configures the kernel function for kernel-based methods (SVM, Gaussian processes, etc.).
    /// </summary>
    /// <param name="kernelFunction">The kernel function implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Kernel functions measure similarity between data points in a
    /// higher-dimensional space. They enable algorithms like SVM to find complex decision boundaries
    /// without explicitly computing the high-dimensional transformation.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureKernelFunction(IKernelFunction<T> kernelFunction)
    {
        _configuredKernelFunction = kernelFunction;
        return this;
    }

    /// <summary>
    /// Configures a regression algorithm for predicting continuous numeric values.
    /// </summary>
    /// <param name="regression">The regression implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Regression predicts numeric values (e.g., house prices, temperature).
    /// Available algorithms include Linear Regression, Ridge, Lasso, Polynomial Regression,
    /// and many more specialized methods.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureRegression(IRegression<T> regression)
    {
        _configuredRegression = regression;
        return this;
    }

    /// <summary>
    /// Configures a classification algorithm for categorizing data into discrete classes.
    /// </summary>
    /// <param name="classifier">The classifier implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Classification assigns data points to categories (e.g., spam/not-spam,
    /// cat/dog/bird). Available algorithms include Logistic Regression, SVM, Decision Trees,
    /// Random Forest, and many more.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureClassifier(IClassifier<T> classifier)
    {
        _configuredClassifier = classifier;
        return this;
    }

    /// <summary>
    /// Configures a clustering algorithm for grouping similar data points together.
    /// </summary>
    /// <param name="clustering">The clustering implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Clustering finds natural groups in your data without needing
    /// labeled examples. For instance, it can group customers by purchasing behavior or
    /// documents by topic. Popular algorithms include K-Means, DBSCAN, and Hierarchical Clustering.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureClustering(IClustering<T> clustering)
    {
        _configuredClustering = clustering;
        return this;
    }

    /// <summary>
    /// Configures a neural network layer for building custom network architectures.
    /// </summary>
    /// <param name="layer">The layer implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Layers are the building blocks of neural networks. Each layer
    /// transforms data in a specific way. Common types include Dense (fully connected), Convolutional
    /// (for images), LSTM/GRU (for sequences), and Attention layers (for transformers).</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureLayer(ILayer<T> layer)
    {
        _configuredLayer = layer;
        return this;
    }

    /// <summary>
    /// Configures an anomaly detection algorithm for identifying unusual data points.
    /// </summary>
    /// <param name="anomalyDetector">The anomaly detector implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Anomaly detection finds data points that don't fit the normal
    /// pattern, like fraudulent transactions, network intrusions, or equipment failures.
    /// Available methods include Isolation Forest, One-Class SVM, Autoencoders, and more.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAnomalyDetector(IAnomalyDetector<T> anomalyDetector)
    {
        _configuredAnomalyDetector = anomalyDetector;
        return this;
    }

    /// <summary>
    /// Configures a 1D interpolation method for estimating values between known data points.
    /// </summary>
    /// <param name="interpolation">The interpolation implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Interpolation estimates values between known data points.
    /// For example, if you know temperatures at noon and 2pm, interpolation can estimate
    /// the temperature at 1pm. Methods include Linear, Cubic Spline, and Polynomial interpolation.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureInterpolation(IInterpolation<T> interpolation)
    {
        _configuredInterpolation = interpolation;
        return this;
    }

    /// <summary>
    /// Configures a 2D interpolation method for estimating values on a surface between known data points.
    /// </summary>
    /// <param name="interpolation">The 2D interpolation implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> 2D interpolation estimates values across a surface (two dimensions).
    /// For example, estimating elevation at any point on a map given known elevation measurements
    /// at specific locations. Methods include Bilinear and Bicubic interpolation.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureInterpolation2D(I2DInterpolation<T> interpolation)
    {
        _configured2DInterpolation = interpolation;
        return this;
    }

    /// <summary>
    /// Configures a window function for signal processing and spectral analysis.
    /// </summary>
    /// <param name="windowFunction">The window function implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Window functions shape a signal before frequency analysis (FFT),
    /// reducing spectral leakage and improving frequency resolution. Common choices include
    /// Hamming, Hanning, Blackman, and Kaiser windows.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureWindowFunction(IWindowFunction<T> windowFunction)
    {
        _configuredWindowFunction = windowFunction;
        return this;
    }

    /// <summary>
    /// Configures a wavelet function for time-frequency analysis and signal decomposition.
    /// </summary>
    /// <param name="waveletFunction">The wavelet function implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Wavelets decompose signals into components at different scales,
    /// capturing both time and frequency information. They're useful for denoising, compression,
    /// and feature extraction. Common wavelets include Haar, Daubechies, and Morlet.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureWaveletFunction(IWaveletFunction<T> waveletFunction)
    {
        _configuredWaveletFunction = waveletFunction;
        return this;
    }

    /// <summary>
    /// Configures a learning rate scheduler that adjusts the learning rate during training.
    /// </summary>
    /// <param name="scheduler">The learning rate scheduler implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A learning rate scheduler changes how fast your model learns
    /// over the course of training. Typically you start with a higher learning rate and reduce it
    /// over time, allowing the model to make large improvements early and fine-tune later.
    /// Common schedulers include StepLR, CosineAnnealing, and ReduceOnPlateau.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureLearningRateScheduler(ILearningRateScheduler scheduler)
    {
        _configuredLearningRateScheduler = scheduler;
        return this;
    }

    /// <summary>
    /// Configures a link function for generalized linear models (GLMs).
    /// </summary>
    /// <param name="linkFunction">The link function implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Link functions connect the linear predictor in a GLM to the
    /// expected response. For example, the Logit link transforms probabilities to the real line
    /// for logistic regression, while the Log link is used for count data (Poisson regression).</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureLinkFunction(ILinkFunction<T> linkFunction)
    {
        _configuredLinkFunction = linkFunction;
        return this;
    }

    /// <summary>
    /// Configures a matrix decomposition method for linear algebra operations.
    /// </summary>
    /// <param name="decomposition">The matrix decomposition implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Matrix decomposition breaks a matrix into simpler components,
    /// enabling efficient solutions to linear systems, dimensionality reduction, and data compression.
    /// Common methods include SVD, QR, LU, and Cholesky decomposition.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureMatrixDecomposition(IMatrixDecomposition<T> decomposition)
    {
        _configuredMatrixDecomposition = decomposition;
        return this;
    }

    /// <summary>
    /// Configures a Gaussian process model for probabilistic predictions with uncertainty estimates.
    /// </summary>
    /// <param name="gaussianProcess">The Gaussian process implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Gaussian processes provide predictions along with uncertainty
    /// estimates, telling you not just what the prediction is but how confident the model is.
    /// They are ideal for small datasets and situations where knowing prediction confidence matters,
    /// such as Bayesian optimization and active learning.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureGaussianProcess(IGaussianProcess<T> gaussianProcess)
    {
        _configuredGaussianProcess = gaussianProcess;
        return this;
    }
}
