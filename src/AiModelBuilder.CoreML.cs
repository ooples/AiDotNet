using AiDotNet.Clustering.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;

namespace AiDotNet;

/// <summary>
/// Core machine learning component configuration extensions for AiModelBuilder.
/// Provides Configure methods for fundamental ML building blocks: loss functions,
/// learning rate schedulers, anomaly detection, and Gaussian processes.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private ILossFunction<T>? _configuredLossFunction;
    private ILearningRateScheduler? _configuredLearningRateScheduler;

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

    // ConfigureKernelFunction removed: a kernel is a constructor parameter of the specific
    // kernel-based model (SVM, GP, etc.). Set it on that model's options — the one door. A
    // builder-level setter cannot reach an already-constructed model whose kernel is read in its ctor.

    // ConfigureClustering removed: IClustering<T> IS an IFullModel; pass it via ConfigureModel(...).

    // ConfigureLayer removed: a layer is a constructor input to a NeuralNetworkArchitecture, which
    // the model is built from. Compose the architecture and pass the model via ConfigureModel(...).

    // ConfigureAnomalyDetector removed: IAnomalyDetector<T> is now an IFullModel<T, Matrix<T>,
    // Vector<T>> (unsupervised, like clustering), so pass it via ConfigureModel(...).

    // ConfigureInterpolation removed: an interpolation method is a constructor parameter of the
    // consuming model/estimator. Set it on that model's options — the one door.

    // ConfigureInterpolation2D removed: same as ConfigureInterpolation, for the 2D case.

    // ConfigureWindowFunction removed: a window function is a constructor parameter of the consuming
    // spectral/signal model. Set it on that model's options — the one door.

    // ConfigureWaveletFunction removed: a wavelet is a constructor parameter of the consuming
    // wavelet-transform model. Set it on that model's options — the one door.

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

    // ConfigureLinkFunction removed: a link function is a constructor parameter of the GLM that uses
    // it. Set it on that model's options — the one door.

    // ConfigureMatrixDecomposition removed: a decomposition is a constructor parameter of the
    // consuming linear-algebra model/solver. Set it on that model's options — the one door.

    // ConfigureGaussianProcess removed: IGaussianProcess<T> is now an IFullModel<T, Matrix<T>,
    // Vector<T>>, so pass it via ConfigureModel(...). Its (mean, variance) Predict overload and
    // uncertainty API remain available directly on the model instance.
}
