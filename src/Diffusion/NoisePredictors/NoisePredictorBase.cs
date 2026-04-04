using AiDotNet.Autodiff;
using AiDotNet.Engines;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Base class for noise prediction networks used in diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class provides common functionality for all noise predictors,
/// including timestep embedding, parameter management, serialization, and gradient computation.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation that all noise prediction networks build upon.
/// Noise predictors are the neural networks at the heart of diffusion models that learn to
/// predict what noise was added to a sample. Different architectures (U-Net, DiT, etc.)
/// extend this base class.
/// </para>
/// </remarks>
public abstract class NoisePredictorBase<T> : INoisePredictor<T>, IModelShape
{
    /// <summary>
    /// Provides access to the hardware-accelerated tensor engine.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Random number generator for initialization and stochastic operations.
    /// </summary>
    protected Random RandomGenerator;

    /// <summary>
    /// The loss function used for training (typically MSE for noise prediction).
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Active feature indices used by the model.
    /// </summary>
    private HashSet<int> _activeFeatureIndices = new HashSet<int>();

    /// <inheritdoc />
    public abstract int InputChannels { get; }

    /// <inheritdoc />
    public abstract int OutputChannels { get; }

    /// <inheritdoc />
    public abstract int BaseChannels { get; }

    /// <inheritdoc />
    public abstract int TimeEmbeddingDim { get; }

    /// <inheritdoc />
    public abstract int ParameterCount { get; }

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;
    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;


    /// <inheritdoc />
    public abstract bool SupportsCFG { get; }

    /// <inheritdoc />
    public abstract bool SupportsCrossAttention { get; }

    /// <inheritdoc />
    public abstract int ContextDimension { get; }

    /// <inheritdoc />
    public ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <inheritdoc />
    public virtual bool SupportsJitCompilation => false;

    /// <summary>
    /// Initializes a new instance of the NoisePredictorBase class.
    /// </summary>
    /// <param name="lossFunction">Optional custom loss function. Defaults to MSE.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    protected NoisePredictorBase(ILossFunction<T>? lossFunction = null, int? seed = null)
    {
        LossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        RandomGenerator = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    #region INoisePredictor<T> Implementation

    /// <inheritdoc />
    public abstract Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null);

    /// <inheritdoc />
    public virtual Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        // Default implementation: use the first element of timeEmbedding to get timestep
        // Derived classes should override for proper batch handling
        var timestep = (int)NumOps.ToDouble(timeEmbedding[0, 0]);
        return PredictNoise(noisySample, timestep, conditioning);
    }

    /// <summary>
    /// Cache for timestep embeddings to avoid recomputing sinusoidal embeddings
    /// for the same timestep during the denoising loop.
    /// </summary>
    private readonly Dictionary<int, Tensor<T>> _timestepEmbeddingCache = new();

    /// <inheritdoc />
    public virtual Tensor<T> GetTimestepEmbedding(int timestep)
    {
        if (_timestepEmbeddingCache.TryGetValue(timestep, out var cached))
            return cached;

        // Sinusoidal timestep embedding (like in Transformers)
        var halfDim = TimeEmbeddingDim / 2;
        var embedding = new Tensor<T>(new[] { TimeEmbeddingDim });
        var embSpan = embedding.AsWritableSpan();

        var logScale = Math.Log(10000.0) / (halfDim - 1);

        for (int i = 0; i < halfDim; i++)
        {
            var freq = Math.Exp(-i * logScale);
            var angle = timestep * freq;

            embSpan[i] = NumOps.FromDouble(Math.Sin(angle));
            embSpan[i + halfDim] = NumOps.FromDouble(Math.Cos(angle));
        }

        _timestepEmbeddingCache[timestep] = embedding;
        return embedding;
    }

    #endregion

    #region IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> Implementation

    /// <inheritdoc />
    public virtual void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Compute gradients and apply them
        var gradients = ComputeGradients(input, expectedOutput, LossFunction);
        var learningRate = NumOps.FromDouble(1e-4);
        ApplyGradients(gradients, learningRate);
    }

    /// <inheritdoc />
    public virtual Tensor<T> Predict(Tensor<T> input)
    {
        // Suppress tape recording during inference
        using var _ = new NoGradScope<T>();
        return PredictNoise(input, 500, null);
    }

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = GetType().Name,
            FeatureCount = ParameterCount,
            Complexity = ParameterCount,
            Description = $"Noise predictor with {ParameterCount} parameters, {InputChannels} input channels, {BaseChannels} base channels."
        };
    }

    #endregion

    #region IParameterizable<T, Tensor<T>, Tensor<T>> Implementation

    /// <inheritdoc />
    public abstract Vector<T> GetParameters();

    /// <inheritdoc />
    public abstract void SetParameters(Vector<T> parameters);

    /// <inheritdoc />
    public virtual IFullModel<T, Tensor<T>, Tensor<T>> WithParameters(Vector<T> parameters)
    {
        var clone = (NoisePredictorBase<T>)Clone();
        clone.SetParameters(parameters);
        return clone;
    }

    #endregion

    #region IModelSerializer Implementation

    /// <inheritdoc />
    public virtual byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        using var stream = new MemoryStream();
        SaveState(stream);
        return stream.ToArray();
    }

    /// <inheritdoc />
    public virtual void Deserialize(byte[] data)
    {
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        using var stream = new MemoryStream(data);
        LoadState(stream);
    }

    /// <inheritdoc/>
    public virtual int[] GetInputShape()
    {
        return new[] { InputChannels };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        return new[] { OutputChannels };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <inheritdoc />
    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or whitespace.", nameof(filePath));

        var data = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            data, this, GetInputShape(), GetOutputShape(), SerializationFormat.Binary);
        File.WriteAllBytes(filePath, envelopedData);
    }

    /// <inheritdoc />
    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or whitespace.", nameof(filePath));

        var data = File.ReadAllBytes(filePath);

        // Extract payload from AIMF envelope
        data = ModelFileHeader.ExtractPayload(data);

        Deserialize(data);
    }

    #endregion

    #region ICheckpointableModel Implementation

    /// <inheritdoc />
    public virtual void SaveState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite)
            throw new ArgumentException("Stream must be writable.", nameof(stream));

        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // Save version for future compatibility
        writer.Write(1); // Version 1

        // Save architecture info
        writer.Write(InputChannels);
        writer.Write(OutputChannels);
        writer.Write(BaseChannels);
        writer.Write(TimeEmbeddingDim);

        // Save model parameters using SerializationHelper
        SerializationHelper<T>.SerializeVector(writer, GetParameters());

        stream.Flush();
    }

    /// <inheritdoc />
    public virtual void LoadState(Stream stream)
    {
        if (stream == null)
            throw new ArgumentNullException(nameof(stream));
        if (!stream.CanRead)
            throw new ArgumentException("Stream must be readable.", nameof(stream));

        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // Read version
        var version = reader.ReadInt32();
        if (version != 1)
            throw new InvalidOperationException($"Unsupported model version: {version}");

        // Read and validate architecture info
        var savedInputChannels = reader.ReadInt32();
        var savedOutputChannels = reader.ReadInt32();
        var savedBaseChannels = reader.ReadInt32();
        var savedTimeEmbeddingDim = reader.ReadInt32();

        if (savedInputChannels != InputChannels || savedOutputChannels != OutputChannels ||
            savedBaseChannels != BaseChannels || savedTimeEmbeddingDim != TimeEmbeddingDim)
        {
            throw new InvalidOperationException(
                $"Architecture mismatch: saved ({savedInputChannels}, {savedOutputChannels}, {savedBaseChannels}, {savedTimeEmbeddingDim}) " +
                $"vs current ({InputChannels}, {OutputChannels}, {BaseChannels}, {TimeEmbeddingDim}).");
        }

        // Load model parameters
        SetParameters(SerializationHelper<T>.DeserializeVector(reader));
    }

    #endregion

    #region IFeatureAware Implementation

    /// <summary>
    /// Ensures active feature indices are initialized with default values if empty.
    /// </summary>
    private void EnsureActiveFeatureIndicesInitialized()
    {
        if (_activeFeatureIndices.Count == 0 && ParameterCount > 0)
        {
            for (int i = 0; i < ParameterCount; i++)
            {
                _activeFeatureIndices.Add(i);
            }
        }
    }

    /// <inheritdoc />
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        EnsureActiveFeatureIndicesInitialized();
        return _activeFeatureIndices;
    }

    /// <inheritdoc />
    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        _activeFeatureIndices = new HashSet<int>(featureIndices);
    }

    /// <inheritdoc />
    public virtual bool IsFeatureUsed(int featureIndex)
    {
        EnsureActiveFeatureIndicesInitialized();
        return _activeFeatureIndices.Contains(featureIndex);
    }

    #endregion

    #region IFeatureImportance<T> Implementation

    /// <inheritdoc />
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        var uniformImportance = NumOps.FromDouble(1.0 / Math.Max(1, ParameterCount));

        for (int i = 0; i < ParameterCount; i++)
        {
            importance[$"param_{i}"] = uniformImportance;
        }

        return importance;
    }

    #endregion

    #region ICloneable<IFullModel<T, Tensor<T>, Tensor<T>>> Implementation

    /// <inheritdoc />
    public abstract IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy();

    /// <inheritdoc />
    IFullModel<T, Tensor<T>, Tensor<T>> ICloneable<IFullModel<T, Tensor<T>, Tensor<T>>>.Clone()
    {
        return Clone();
    }

    /// <summary>
    /// Creates a deep copy of the noise predictor.
    /// </summary>
    /// <returns>A new instance with the same parameters.</returns>
    public abstract INoisePredictor<T> Clone();

    #endregion

    #region IGradientComputable<T, Tensor<T>, Tensor<T>> Implementation

    /// <inheritdoc />
    public virtual Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (target == null)
            throw new ArgumentNullException(nameof(target));

        var effectiveLossFunction = lossFunction ?? LossFunction;

        // Layer-level backpropagation: forward through layers, compute loss gradient,
        // then backpropagate through the layer chain for exact gradients.
        // Forward pass
        var predicted = Forward(input);

        // Compute loss gradient: d(loss)/d(predicted)
        var lossGrad = effectiveLossFunction.CalculateDerivative(
            predicted.ToVector(), target.ToVector());
        var lossGradTensor = new Tensor<T>(predicted._shape, lossGrad);

        // Backpropagate through all layers

        // Extract parameter gradients from layers
        return GetParameterGradients();
    }

    /// <summary>
    /// Forward pass through the noise predictor's layers.
    /// Override to implement the actual forward computation.
    /// </summary>
    protected virtual Tensor<T> Forward(Tensor<T> input)
    {
        return PredictNoise(input, 0);
    }

    /// <summary>
    /// Computes gradients using the Tensors GradientTape for automatic differentiation.
    /// This is the preferred training path — gradients are computed by recording all
    /// engine ops during the forward pass and then running reverse-mode AD.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <param name="target">The target tensor for loss computation.</param>
    /// <param name="trainableParams">The trainable parameter tensors to compute gradients for.</param>
    /// <returns>Dictionary mapping each parameter tensor to its gradient.</returns>
    public Dictionary<Tensor<T>, Tensor<T>> ComputeGradientsWithTape(
        Tensor<T> input,
        Tensor<T> target,
        Tensor<T>[] trainableParams)
    {
        using var tape = new GradientTape<T>();

        // Forward pass (recorded by the engine)
        var predicted = Forward(input);

        // Compute MSE loss using tape-recorded engine ops
        var diff = Engine.TensorSubtract(predicted, target);
        var squared = Engine.TensorMultiply(diff, diff);
        // ReduceMean with all axes produces a scalar tensor that the tape can differentiate
        var allAxes = Enumerable.Range(0, squared.Shape.Length).ToArray();
        var loss = Engine.ReduceMean(squared, allAxes, keepDims: false);

        // Reverse-mode AD: compute gradients for all trainable parameters
        return tape.ComputeGradients(loss, trainableParams);
    }

    /// <summary>
    /// Extracts accumulated parameter gradients from all layers after backpropagation.
    /// </summary>
    protected virtual Vector<T> GetParameterGradients()
    {
        throw new NotSupportedException(
            $"{GetType().Name} does not implement GetParameterGradients. " +
            "Override this method to extract layer-level gradients.");
    }

    /// <inheritdoc />
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();

        // Vectorized SGD: params = params - lr * gradients
        var scaledGradients = Engine.Multiply(gradients, learningRate);
        var updated = Engine.Subtract(parameters, scaledGradients);

        SetParameters(updated);
    }

    #endregion

    #region IJitCompilable<T> Implementation

    /// <inheritdoc />
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("This noise predictor does not support JIT compilation. Override ExportComputationGraph in derived class if needed.");
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Samples random noise from a standard normal distribution.
    /// </summary>
    /// <param name="shape">The shape of the noise tensor.</param>
    /// <param name="rng">Optional random number generator.</param>
    /// <returns>A tensor of random noise values.</returns>
    protected virtual Tensor<T> SampleNoise(int[] shape, Random? rng = null)
    {
        rng = rng ?? RandomGenerator;
        long totalElements = 1;
        foreach (var dim in shape)
            totalElements = checked(totalElements * dim);

        var noise = new Tensor<T>(shape);
        var noiseSpan = noise.AsWritableSpan();

        for (int i = 0; i < noiseSpan.Length; i++)
        {
            noiseSpan[i] = NumOps.FromDouble(rng.NextGaussian());
        }

        return noise;
    }

    #endregion
}
