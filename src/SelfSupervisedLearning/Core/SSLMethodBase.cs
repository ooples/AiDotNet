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
    /// Gets the global execution engine for vector operations and GPU/CPU acceleration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The engine handles hardware-accelerated computations.
    /// It automatically selects the best available hardware (GPU if available, otherwise CPU)
    /// for matrix operations, making SSL training much faster.</para>
    /// </remarks>
    protected IEngine Engine => AiDotNetEngine.Current;

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

        if (parameters.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Parameter vector length ({parameters.Length}) does not match expected count ({ParameterCount}).",
                nameof(parameters));
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
            // Use Engine for accelerated vector operations when available
            var row = new T[dim];
            for (int j = 0; j < dim; j++)
            {
                row[j] = tensor[i, j];
            }

            var vec = new Vector<T>(row);

            // Compute L2 norm using engine's dot product
            var normSquared = Engine.DotProduct(vec, vec);
            var norm = NumOps.Sqrt(NumOps.Add(normSquared, NumOps.FromDouble(1e-8)));

            // Normalize
            for (int j = 0; j < dim; j++)
            {
                result[i * dim + j] = NumOps.Divide(tensor[i, j], norm);
            }
        }

        return new Tensor<T>(result, [batchSize, dim]);
    }

    /// <summary>
    /// Computes matrix multiplication with engine-accelerated dot products.
    /// </summary>
    /// <param name="a">First matrix [M, K].</param>
    /// <param name="b">Second matrix [K, N].</param>
    /// <returns>Result matrix [M, N].</returns>
    protected virtual Tensor<T> MatMul(Tensor<T> a, Tensor<T> b)
    {
        var m = a.Shape[0];
        var k = a.Shape[1];
        var n = b.Shape[1];
        var result = new T[m * n];

        // Transpose b for better cache locality: [K, N] -> column vectors of length K
        for (int i = 0; i < m; i++)
        {
            // Extract row i from a
            var rowA = new T[k];
            for (int j = 0; j < k; j++)
            {
                rowA[j] = a[i, j];
            }
            var vecA = new Vector<T>(rowA);

            for (int j = 0; j < n; j++)
            {
                // Extract column j from b
                var colB = new T[k];
                for (int l = 0; l < k; l++)
                {
                    colB[l] = b[l, j];
                }
                var vecB = new Vector<T>(colB);

                // Use engine for accelerated dot product
                result[i * n + j] = Engine.DotProduct(vecA, vecB);
            }
        }

        return new Tensor<T>(result, [m, n]);
    }

    /// <summary>
    /// Computes similarity matrix between two sets of embeddings.
    /// </summary>
    /// <param name="embeddings1">First set of embeddings [N, D].</param>
    /// <param name="embeddings2">Second set of embeddings [M, D].</param>
    /// <param name="normalize">Whether to L2-normalize before computing similarity.</param>
    /// <returns>Similarity matrix [N, M].</returns>
    protected virtual Tensor<T> ComputeSimilarityMatrix(
        Tensor<T> embeddings1,
        Tensor<T> embeddings2,
        bool normalize = true)
    {
        var n = embeddings1.Shape[0];
        var m = embeddings2.Shape[0];
        var d = embeddings1.Shape[1];

        var e1 = normalize ? L2Normalize(embeddings1) : embeddings1;
        var e2 = normalize ? L2Normalize(embeddings2) : embeddings2;

        var result = new T[n * m];

        // Compute dot products using engine
        for (int i = 0; i < n; i++)
        {
            var row1 = new T[d];
            for (int k = 0; k < d; k++)
            {
                row1[k] = e1[i, k];
            }
            var vec1 = new Vector<T>(row1);

            for (int j = 0; j < m; j++)
            {
                var row2 = new T[d];
                for (int k = 0; k < d; k++)
                {
                    row2[k] = e2[j, k];
                }
                var vec2 = new Vector<T>(row2);

                result[i * m + j] = Engine.DotProduct(vec1, vec2);
            }
        }

        return new Tensor<T>(result, [n, m]);
    }

    /// <summary>
    /// Computes the pairwise squared distances between embeddings.
    /// </summary>
    /// <param name="embeddings">Embeddings [N, D].</param>
    /// <returns>Distance matrix [N, N].</returns>
    protected virtual Tensor<T> ComputePairwiseDistances(Tensor<T> embeddings)
    {
        var n = embeddings.Shape[0];
        var d = embeddings.Shape[1];
        var result = new T[n * n];

        // ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        // First compute norms using engine
        var norms = new T[n];
        for (int i = 0; i < n; i++)
        {
            var row = new T[d];
            for (int j = 0; j < d; j++)
            {
                row[j] = embeddings[i, j];
            }
            var vec = new Vector<T>(row);
            norms[i] = Engine.DotProduct(vec, vec);
        }

        // Compute similarity matrix using engine
        var similarity = ComputeSimilarityMatrix(embeddings, embeddings, normalize: false);

        // Compute distances: ||a||^2 + ||b||^2 - 2 * a.b
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                var dist = NumOps.Subtract(
                    NumOps.Add(norms[i], norms[j]),
                    NumOps.Multiply(NumOps.FromDouble(2.0), similarity[i, j]));
                // Clamp negative values due to floating point errors
                result[i * n + j] = NumOps.GreaterThan(dist, NumOps.Zero) ? dist : NumOps.Zero;
            }
        }

        return new Tensor<T>(result, [n, n]);
    }
}
