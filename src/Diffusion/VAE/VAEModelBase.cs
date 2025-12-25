using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

namespace AiDotNet.Diffusion.VAE;

/// <summary>
/// Base class for Variational Autoencoder (VAE) models used in latent diffusion.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class provides common functionality for all VAE implementations,
/// including encoding, decoding, sampling, and latent scaling operations.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the foundation for all VAE models in the library.
/// VAEs compress images to a small latent representation and decompress them back.
/// They are essential for efficient latent diffusion models like Stable Diffusion.
/// </para>
/// </remarks>
public abstract class VAEModelBase<T> : IVAEModel<T>
{
    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Random number generator for sampling operations.
    /// </summary>
    protected Random RandomGenerator;

    /// <summary>
    /// The loss function used for training.
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Active feature indices used by the model.
    /// </summary>
    private HashSet<int> _activeFeatureIndices = new HashSet<int>();

    /// <summary>
    /// Whether tiling mode is enabled for memory-efficient processing.
    /// </summary>
    protected bool TilingEnabled;

    /// <summary>
    /// Whether slicing mode is enabled for sequential processing.
    /// </summary>
    protected bool SlicingEnabled;

    /// <inheritdoc />
    public abstract int InputChannels { get; }

    /// <inheritdoc />
    public abstract int LatentChannels { get; }

    /// <inheritdoc />
    public abstract int DownsampleFactor { get; }

    /// <inheritdoc />
    public abstract double LatentScaleFactor { get; }

    /// <inheritdoc />
    public abstract int ParameterCount { get; }

    /// <inheritdoc />
    public virtual bool SupportsTiling => false;

    /// <inheritdoc />
    public virtual bool SupportsSlicing => false;

    /// <inheritdoc />
    public ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <inheritdoc />
    public virtual bool SupportsJitCompilation => false;

    /// <summary>
    /// Initializes a new instance of the VAEModelBase class.
    /// </summary>
    /// <param name="lossFunction">Optional custom loss function. Defaults to MSE.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    protected VAEModelBase(ILossFunction<T>? lossFunction = null, int? seed = null)
    {
        LossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        RandomGenerator = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    #region IVAEModel<T> Implementation

    /// <inheritdoc />
    public abstract Tensor<T> Encode(Tensor<T> image, bool sampleMode = true);

    /// <inheritdoc />
    public abstract (Tensor<T> Mean, Tensor<T> LogVariance) EncodeWithDistribution(Tensor<T> image);

    /// <inheritdoc />
    public abstract Tensor<T> Decode(Tensor<T> latent);

    /// <inheritdoc />
    public virtual Tensor<T> Sample(Tensor<T> mean, Tensor<T> logVariance, int? seed = null)
    {
        // Reparameterization trick: z = mean + std * epsilon
        // std = exp(0.5 * logVariance)

        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;
        var epsilon = SampleNoise(mean.Shape, rng);

        var result = new Tensor<T>(mean.Shape);
        var meanSpan = mean.AsSpan();
        var logVarSpan = logVariance.AsSpan();
        var epsilonSpan = epsilon.AsSpan();
        var resultSpan = result.AsWritableSpan();

        var halfOne = NumOps.FromDouble(0.5);

        for (int i = 0; i < resultSpan.Length; i++)
        {
            // std = exp(0.5 * logVar)
            var std = NumOps.Exp(NumOps.Multiply(halfOne, logVarSpan[i]));
            // z = mean + std * epsilon
            resultSpan[i] = NumOps.Add(meanSpan[i], NumOps.Multiply(std, epsilonSpan[i]));
        }

        return result;
    }

    /// <inheritdoc />
    public virtual Tensor<T> ScaleLatent(Tensor<T> latent)
    {
        var scaleFactor = NumOps.FromDouble(LatentScaleFactor);
        var result = new Tensor<T>(latent.Shape);
        var latentSpan = latent.AsSpan();
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = NumOps.Multiply(latentSpan[i], scaleFactor);
        }

        return result;
    }

    /// <inheritdoc />
    public virtual Tensor<T> UnscaleLatent(Tensor<T> latent)
    {
        var invScaleFactor = NumOps.FromDouble(1.0 / LatentScaleFactor);
        var result = new Tensor<T>(latent.Shape);
        var latentSpan = latent.AsSpan();
        var resultSpan = result.AsWritableSpan();

        for (int i = 0; i < resultSpan.Length; i++)
        {
            resultSpan[i] = NumOps.Multiply(latentSpan[i], invScaleFactor);
        }

        return result;
    }

    /// <inheritdoc />
    public virtual void SetTilingEnabled(bool enabled)
    {
        if (enabled && !SupportsTiling)
            throw new NotSupportedException("This VAE does not support tiling mode.");
        TilingEnabled = enabled;
    }

    /// <inheritdoc />
    public virtual void SetSlicingEnabled(bool enabled)
    {
        if (enabled && !SupportsSlicing)
            throw new NotSupportedException("This VAE does not support slicing mode.");
        SlicingEnabled = enabled;
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
        // For VAE, prediction is encode->decode (reconstruction)
        var latent = Encode(input, sampleMode: false);
        return Decode(latent);
    }

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = GetType().Name,
            ModelType = Enums.ModelType.NeuralNetwork,
            FeatureCount = ParameterCount,
            Complexity = ParameterCount,
            Description = $"VAE with {ParameterCount} parameters, {InputChannels} input channels, " +
                          $"{LatentChannels} latent channels, {DownsampleFactor}x downsampling."
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
        var clone = (VAEModelBase<T>)Clone();
        clone.SetParameters(parameters);
        return clone;
    }

    #endregion

    #region IModelSerializer Implementation

    /// <inheritdoc />
    public virtual byte[] Serialize()
    {
        using var stream = new MemoryStream();
        SaveState(stream);
        return stream.ToArray();
    }

    /// <inheritdoc />
    public virtual void Deserialize(byte[] data)
    {
        using var stream = new MemoryStream(data);
        LoadState(stream);
    }

    /// <inheritdoc />
    public virtual void SaveModel(string filePath)
    {
        var data = Serialize();
        File.WriteAllBytes(filePath, data);
    }

    /// <inheritdoc />
    public virtual void LoadModel(string filePath)
    {
        var data = File.ReadAllBytes(filePath);
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
        writer.Write(LatentChannels);
        writer.Write(DownsampleFactor);
        writer.Write(LatentScaleFactor);

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
        var savedLatentChannels = reader.ReadInt32();
        var savedDownsampleFactor = reader.ReadInt32();
        var savedLatentScaleFactor = reader.ReadDouble();

        if (savedInputChannels != InputChannels || savedLatentChannels != LatentChannels ||
            savedDownsampleFactor != DownsampleFactor)
        {
            throw new InvalidOperationException(
                $"Architecture mismatch: saved ({savedInputChannels}, {savedLatentChannels}, {savedDownsampleFactor}) " +
                $"vs current ({InputChannels}, {LatentChannels}, {DownsampleFactor}).");
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
    /// Creates a deep copy of the VAE model.
    /// </summary>
    /// <returns>A new instance with the same parameters.</returns>
    public abstract IVAEModel<T> Clone();

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
        var parameters = GetParameters();
        var gradients = new Vector<T>(parameters.Length);

        // Numerical gradient computation using finite differences
        var epsilon = NumOps.FromDouble(1e-5);
        var twoEpsilon = NumOps.Multiply(epsilon, NumOps.FromDouble(2.0));

        for (int i = 0; i < parameters.Length; i++)
        {
            // Compute f(x + epsilon)
            var paramsPlus = new Vector<T>(parameters.Length);
            for (int j = 0; j < parameters.Length; j++)
            {
                paramsPlus[j] = j == i ? NumOps.Add(parameters[j], epsilon) : parameters[j];
            }
            SetParameters(paramsPlus);
            var outputPlus = Predict(input);
            var lossPlus = effectiveLossFunction.CalculateLoss(outputPlus.ToVector(), target.ToVector());

            // Compute f(x - epsilon)
            var paramsMinus = new Vector<T>(parameters.Length);
            for (int j = 0; j < parameters.Length; j++)
            {
                paramsMinus[j] = j == i ? NumOps.Subtract(parameters[j], epsilon) : parameters[j];
            }
            SetParameters(paramsMinus);
            var outputMinus = Predict(input);
            var lossMinus = effectiveLossFunction.CalculateLoss(outputMinus.ToVector(), target.ToVector());

            // Gradient = (f(x+eps) - f(x-eps)) / (2*eps)
            gradients[i] = NumOps.Divide(NumOps.Subtract(lossPlus, lossMinus), twoEpsilon);
        }

        // Restore original parameters
        SetParameters(parameters);

        return gradients;
    }

    /// <inheritdoc />
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();

        for (int i = 0; i < parameters.Length && i < gradients.Length; i++)
        {
            var update = NumOps.Multiply(gradients[i], learningRate);
            parameters[i] = NumOps.Subtract(parameters[i], update);
        }

        SetParameters(parameters);
    }

    #endregion

    #region IJitCompilable<T> Implementation

    /// <inheritdoc />
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        throw new NotSupportedException("This VAE does not support JIT compilation. Override ExportComputationGraph in derived class if needed.");
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
        var noise = new Tensor<T>(shape);
        var noiseSpan = noise.AsWritableSpan();

        // Box-Muller transform for normal distribution
        for (int i = 0; i < noiseSpan.Length; i += 2)
        {
            var u1 = rng.NextDouble();
            var u2 = rng.NextDouble();

            while (u1 <= double.Epsilon)
                u1 = rng.NextDouble();

            var mag = Math.Sqrt(-2.0 * Math.Log(u1));
            var z0 = mag * Math.Cos(2.0 * Math.PI * u2);
            var z1 = mag * Math.Sin(2.0 * Math.PI * u2);

            noiseSpan[i] = NumOps.FromDouble(z0);
            if (i + 1 < noiseSpan.Length)
                noiseSpan[i + 1] = NumOps.FromDouble(z1);
        }

        return noise;
    }

    /// <summary>
    /// Computes the KL divergence loss for VAE training.
    /// </summary>
    /// <param name="mean">The mean of the latent distribution.</param>
    /// <param name="logVariance">The log variance of the latent distribution.</param>
    /// <returns>The KL divergence loss value.</returns>
    /// <remarks>
    /// KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    /// </remarks>
    protected virtual T ComputeKLDivergence(Tensor<T> mean, Tensor<T> logVariance)
    {
        var meanSpan = mean.AsSpan();
        var logVarSpan = logVariance.AsSpan();

        var sum = NumOps.Zero;
        var one = NumOps.One;
        var halfNeg = NumOps.FromDouble(-0.5);

        for (int i = 0; i < meanSpan.Length; i++)
        {
            var mu = meanSpan[i];
            var logVar = logVarSpan[i];

            // 1 + log(sigma^2) - mu^2 - sigma^2
            // = 1 + logVar - mu^2 - exp(logVar)
            var term = NumOps.Add(one, logVar);
            term = NumOps.Subtract(term, NumOps.Multiply(mu, mu));
            term = NumOps.Subtract(term, NumOps.Exp(logVar));

            sum = NumOps.Add(sum, term);
        }

        // -0.5 * sum
        return NumOps.Multiply(halfNeg, sum);
    }

    #endregion
}
