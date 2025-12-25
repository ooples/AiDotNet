using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Meta;

/// <summary>
/// Base class for meta classifiers that wrap other classifiers.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Meta classifiers use other classifiers as base estimators to provide
/// enhanced functionality like multi-class support, multi-label support,
/// or ensemble voting.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Meta classifiers are "classifiers about classifiers":
///
/// Examples:
/// - OneVsRest: Trains one classifier per class
/// - OneVsOne: Trains one classifier per pair of classes
/// - Voting: Combines predictions from multiple classifiers
/// - Stacking: Uses classifier outputs as features for another classifier
///
/// They extend the capabilities of base classifiers.
/// </para>
/// </remarks>
public abstract class MetaClassifierBase<T> : ProbabilisticClassifierBase<T>
{
    /// <summary>
    /// Gets the meta classifier specific options.
    /// </summary>
    protected new MetaClassifierOptions<T> Options => (MetaClassifierOptions<T>)base.Options;

    /// <summary>
    /// The base estimator factory function.
    /// </summary>
    protected Func<IClassifier<T>>? EstimatorFactory { get; set; }

    /// <summary>
    /// Initializes a new instance of the MetaClassifierBase class.
    /// </summary>
    /// <param name="options">Configuration options for the meta classifier.</param>
    /// <param name="estimatorFactory">Factory function to create base estimators.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    protected MetaClassifierBase(
        MetaClassifierOptions<T>? options = null,
        Func<IClassifier<T>>? estimatorFactory = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new MetaClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
        EstimatorFactory = estimatorFactory;
    }

    /// <summary>
    /// Creates a new base estimator instance.
    /// </summary>
    /// <returns>A new classifier instance.</returns>
    protected IClassifier<T> CreateBaseEstimator()
    {
        if (EstimatorFactory is null)
        {
            throw new InvalidOperationException("Estimator factory is not set.");
        }
        return EstimatorFactory();
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        // Meta classifiers typically don't have a simple parameter vector
        return new Vector<T>(0);
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("Meta classifiers do not support setting parameters directly.");
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        throw new NotSupportedException("Meta classifiers do not support setting parameters directly.");
    }

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Meta classifiers delegate to base estimators
        return new Vector<T>(0);
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // No-op for meta classifiers
    }
}

/// <summary>
/// Configuration options for meta classifiers.
/// </summary>
/// <typeparam name="T">The data type used for calculations.</typeparam>
public class MetaClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of parallel jobs for fitting (when supported).
    /// </summary>
    /// <value>Number of parallel jobs. -1 means use all processors. Default is 1.</value>
    public int NumJobs { get; set; } = 1;
}
