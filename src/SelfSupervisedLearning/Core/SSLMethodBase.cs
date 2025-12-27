using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.SelfSupervisedLearning.Core.Interfaces;

namespace AiDotNet.SelfSupervisedLearning.Core;

/// <summary>
/// Abstract base class for self-supervised learning methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This base class provides common functionality shared by all
/// SSL methods, including parameter management, training mode control, and configuration handling.</para>
///
/// <para>Derived classes (SimCLR, MoCo, BYOL, etc.) implement the specific training logic
/// in the <see cref="TrainStepCore"/> method.</para>
/// </remarks>
public abstract class SSLMethodBase<T> : ISSLMethod<T>
{
    /// <summary>
    /// Numeric operations for generic type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The main encoder neural network.
    /// </summary>
    protected readonly INeuralNetwork<T> _encoder;

    /// <summary>
    /// The projection head for SSL embeddings.
    /// </summary>
    protected readonly IProjectorHead<T>? _projector;

    /// <summary>
    /// The SSL configuration.
    /// </summary>
    protected readonly SSLConfig _config;

    /// <summary>
    /// Whether the method is in training mode.
    /// </summary>
    protected bool _isTraining = true;

    /// <summary>
    /// Current training step counter.
    /// </summary>
    protected int _currentStep;

    /// <summary>
    /// Current epoch counter.
    /// </summary>
    protected int _currentEpoch;

    /// <inheritdoc />
    public abstract string Name { get; }

    /// <inheritdoc />
    public abstract SSLMethodCategory Category { get; }

    /// <inheritdoc />
    public abstract bool RequiresMemoryBank { get; }

    /// <inheritdoc />
    public abstract bool UsesMomentumEncoder { get; }

    /// <inheritdoc />
    public int ParameterCount
    {
        get
        {
            int count = 0;
            var encoderParams = _encoder.GetParameters();
            count += encoderParams.Length;

            if (_projector is not null)
            {
                count += _projector.ParameterCount;
            }

            count += GetAdditionalParameterCount();
            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the SSLMethodBase class.
    /// </summary>
    /// <param name="encoder">The encoder neural network.</param>
    /// <param name="projector">Optional projection head.</param>
    /// <param name="config">SSL configuration.</param>
    protected SSLMethodBase(
        INeuralNetwork<T> encoder,
        IProjectorHead<T>? projector,
        SSLConfig? config)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _projector = projector;
        _config = config ?? new SSLConfig();
    }

    /// <inheritdoc />
    public INeuralNetwork<T> GetEncoder() => _encoder;

    /// <inheritdoc />
    public SSLStepResult<T> TrainStep(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext = null)
    {
        if (batch is null)
        {
            throw new ArgumentNullException(nameof(batch));
        }

        _currentStep++;

        // Set training mode
        _encoder.SetTrainingMode(true);
        _projector?.SetTrainingMode(true);

        // Delegate to implementation-specific training step
        return TrainStepCore(batch, augmentationContext);
    }

    /// <summary>
    /// Implementation-specific training step logic.
    /// </summary>
    /// <param name="batch">The input batch tensor.</param>
    /// <param name="augmentationContext">Optional augmentation context.</param>
    /// <returns>The result of the training step.</returns>
    protected abstract SSLStepResult<T> TrainStepCore(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext);

    /// <inheritdoc />
    public virtual Tensor<T> Encode(Tensor<T> input)
    {
        if (input is null)
        {
            throw new ArgumentNullException(nameof(input));
        }

        // Set evaluation mode for encoding
        _encoder.SetTrainingMode(false);

        // Get encoder output
        var encoded = _encoder.Predict(input);

        return encoded;
    }

    /// <summary>
    /// Encodes input and projects it to the SSL embedding space.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The projected embedding.</returns>
    protected virtual Tensor<T> EncodeAndProject(Tensor<T> input)
    {
        var encoded = _encoder.ForwardWithMemory(input);

        if (_projector is not null)
        {
            return _projector.Project(encoded);
        }

        return encoded;
    }

    /// <inheritdoc />
    public virtual void Reset()
    {
        _currentStep = 0;
        _currentEpoch = 0;
        _projector?.Reset();
    }

    /// <inheritdoc />
    public virtual Vector<T> GetParameters()
    {
        var encoderParams = _encoder.GetParameters();
        var projectorParams = _projector?.GetParameters();
        var additionalParams = GetAdditionalParameters();

        // Calculate total length
        int totalLength = encoderParams.Length;
        if (projectorParams is not null) totalLength += projectorParams.Length;
        if (additionalParams is not null) totalLength += additionalParams.Length;

        // Create combined parameter vector
        var combined = new T[totalLength];
        int offset = 0;

        // Copy encoder parameters
        for (int i = 0; i < encoderParams.Length; i++)
        {
            combined[offset++] = encoderParams[i];
        }

        // Copy projector parameters
        if (projectorParams is not null)
        {
            for (int i = 0; i < projectorParams.Length; i++)
            {
                combined[offset++] = projectorParams[i];
            }
        }

        // Copy additional parameters
        if (additionalParams is not null)
        {
            for (int i = 0; i < additionalParams.Length; i++)
            {
                combined[offset++] = additionalParams[i];
            }
        }

        return new Vector<T>(combined);
    }

    /// <inheritdoc />
    public virtual void SetParameters(Vector<T> parameters)
    {
        if (parameters is null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }

        int offset = 0;

        // Set encoder parameters
        var encoderParams = _encoder.GetParameters();
        var encoderLength = encoderParams.Length;
        var encoderParamArray = new T[encoderLength];
        for (int i = 0; i < encoderLength; i++)
        {
            encoderParamArray[i] = parameters[offset++];
        }
        _encoder.UpdateParameters(new Vector<T>(encoderParamArray));

        // Set projector parameters
        if (_projector is not null)
        {
            var projectorLength = _projector.ParameterCount;
            var projectorParamArray = new T[projectorLength];
            for (int i = 0; i < projectorLength; i++)
            {
                projectorParamArray[i] = parameters[offset++];
            }
            _projector.SetParameters(new Vector<T>(projectorParamArray));
        }

        // Set additional parameters
        SetAdditionalParameters(parameters, ref offset);
    }

    /// <summary>
    /// Gets additional parameters specific to this SSL method.
    /// </summary>
    /// <returns>Additional parameters, or null if none.</returns>
    protected virtual Vector<T>? GetAdditionalParameters() => null;

    /// <summary>
    /// Gets the count of additional parameters.
    /// </summary>
    /// <returns>The number of additional parameters.</returns>
    protected virtual int GetAdditionalParameterCount() => 0;

    /// <summary>
    /// Sets additional parameters specific to this SSL method.
    /// </summary>
    /// <param name="parameters">The full parameter vector.</param>
    /// <param name="offset">The current offset into the parameter vector.</param>
    protected virtual void SetAdditionalParameters(Vector<T> parameters, ref int offset)
    {
        // Default implementation does nothing
    }

    /// <summary>
    /// Sets the training mode for the SSL method.
    /// </summary>
    /// <param name="isTraining">True for training mode, false for evaluation.</param>
    public virtual void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
        _encoder.SetTrainingMode(isTraining);
        _projector?.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Signals the start of a new epoch.
    /// </summary>
    /// <param name="epochNumber">The current epoch number.</param>
    public virtual void OnEpochStart(int epochNumber)
    {
        _currentEpoch = epochNumber;
    }

    /// <summary>
    /// Signals the end of an epoch.
    /// </summary>
    /// <param name="epochNumber">The completed epoch number.</param>
    public virtual void OnEpochEnd(int epochNumber)
    {
        // Subclasses can override for epoch-end operations
    }

    /// <summary>
    /// Gets the effective temperature based on configuration and scheduling.
    /// </summary>
    /// <returns>The current temperature value.</returns>
    protected virtual double GetEffectiveTemperature()
    {
        return _config.Temperature ?? 0.07;
    }

    /// <summary>
    /// Gets the effective learning rate based on configuration and scheduling.
    /// </summary>
    /// <returns>The current learning rate.</returns>
    public virtual double GetEffectiveLearningRate()
    {
        var baseLr = _config.LearningRate ?? 0.3;
        var batchSize = _config.BatchSize ?? 256;

        // Linear scaling rule
        var effectiveLr = baseLr * batchSize / 256.0;

        // Apply cosine decay if enabled
        if (_config.UseCosineDecay == true)
        {
            var totalSteps = (_config.PretrainingEpochs ?? 100) * 1000; // Approximate
            var progress = Math.Min(1.0, (double)_currentStep / totalSteps);
            effectiveLr *= 0.5 * (1.0 + Math.Cos(Math.PI * progress));
        }

        // Apply warmup if in warmup phase
        var warmupEpochs = _config.WarmupEpochs ?? 10;
        if (_currentEpoch < warmupEpochs)
        {
            var warmupProgress = (double)_currentEpoch / warmupEpochs;
            effectiveLr *= warmupProgress;
        }

        return effectiveLr;
    }

    /// <summary>
    /// Creates a default step result with common metrics.
    /// </summary>
    /// <param name="loss">The loss value.</param>
    /// <returns>A step result with populated common fields.</returns>
    protected SSLStepResult<T> CreateStepResult(T loss)
    {
        return new SSLStepResult<T>
        {
            Loss = loss,
            CurrentLearningRate = GetEffectiveLearningRate(),
            CurrentTemperature = GetEffectiveTemperature(),
            Metrics = []
        };
    }

    /// <summary>
    /// Computes cosine similarity between two tensors.
    /// </summary>
    /// <param name="a">First tensor [batch, dim].</param>
    /// <param name="b">Second tensor [batch, dim].</param>
    /// <returns>Cosine similarity values [batch].</returns>
    protected virtual Tensor<T> CosineSimilarity(Tensor<T> a, Tensor<T> b)
    {
        // Normalize both tensors
        var aNorm = L2Normalize(a);
        var bNorm = L2Normalize(b);

        // Compute dot product along last dimension
        var batchSize = a.Shape[0];
        var dim = a.Shape[1];
        var result = new T[batchSize];

        for (int i = 0; i < batchSize; i++)
        {
            T sum = NumOps.Zero;
            for (int j = 0; j < dim; j++)
            {
                sum = NumOps.Add(sum, NumOps.Multiply(aNorm[i, j], bNorm[i, j]));
            }
            result[i] = sum;
        }

        return new Tensor<T>(result, [batchSize]);
    }

    /// <summary>
    /// L2-normalizes a tensor along the last dimension.
    /// </summary>
    /// <param name="tensor">The tensor to normalize [batch, dim].</param>
    /// <returns>The normalized tensor.</returns>
    protected virtual Tensor<T> L2Normalize(Tensor<T> tensor)
    {
        var batchSize = tensor.Shape[0];
        var dim = tensor.Shape[1];
        var result = new T[batchSize * dim];

        for (int i = 0; i < batchSize; i++)
        {
            // Compute L2 norm
            T sumSquared = NumOps.Zero;
            for (int j = 0; j < dim; j++)
            {
                var val = tensor[i, j];
                sumSquared = NumOps.Add(sumSquared, NumOps.Multiply(val, val));
            }

            var norm = NumOps.Sqrt(NumOps.Add(sumSquared, NumOps.FromDouble(1e-8)));

            // Normalize
            for (int j = 0; j < dim; j++)
            {
                result[i * dim + j] = NumOps.Divide(tensor[i, j], norm);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }
}
