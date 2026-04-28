using AiDotNet.Autodiff;
using AiDotNet.Engines;
using AiDotNet.Extensions;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Tensors.Engines.Autodiff;
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
public abstract class VAEModelBase<T> : IVAEModel<T>, IModelShape
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

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;
    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;


    /// <inheritdoc />
    public virtual bool SupportsTiling => false;

    /// <inheritdoc />
    public virtual bool SupportsSlicing => false;

    /// <inheritdoc />
    public ILossFunction<T> DefaultLossFunction => LossFunction;

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
        var epsilon = SampleNoise(mean._shape, rng);

        var result = new Tensor<T>(mean._shape);
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
        return Engine.TensorMultiplyScalar(latent, scaleFactor);
    }

    /// <inheritdoc />
    public virtual Tensor<T> UnscaleLatent(Tensor<T> latent)
    {
        var invScaleFactor = NumOps.FromDouble(1.0 / LatentScaleFactor);
        return Engine.TensorMultiplyScalar(latent, invScaleFactor);
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
        // Suppress tape recording during inference
        using var _ = new NoGradScope<T>();
        var latent = Encode(input, sampleMode: false);
        return Decode(latent);
    }

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = GetType().Name,
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
        // VAE input is [Channels, Height, Width] where H/W are dynamic
        return new[] { InputChannels, -1, -1 };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        // Predict() returns Decode(Encode(input)), which is in input space [C, H, W]
        return new[] { InputChannels, -1, -1 };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        // Height and Width dimensions (indices 1,2) are dynamic
        return new DynamicShapeInfo
        {
            DynamicInputDimensions = new[] { 1, 2 },
            DynamicOutputDimensions = new[] { 1, 2 }
        };
    }


    /// <inheritdoc />
    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        var data = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            data, this, GetInputShape(), GetOutputShape(), SerializationFormat.Binary,
            GetDynamicShapeInfo());
        File.WriteAllBytes(fullPath, envelopedData);
    }

    /// <inheritdoc />
    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var data = File.ReadAllBytes(filePath);

        // Extract payload from AIMF envelope if present; use raw bytes for legacy files
        if (ModelFileHeader.HasHeader(data))
        {
            data = ModelFileHeader.ExtractPayload(data);
        }

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
        _ = reader.ReadDouble(); // Read latent scale factor (compatible across versions)

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

        // Primary path: tape-recorded forward + layer-level backprop. ForwardForTraining
        // runs encode+decode without suppressing tape recording so the per-layer caches
        // populate. We then push the loss derivative back through the layer chain via
        // BackpropagateLossGradient, and finally read accumulated gradients via the
        // concrete VAE's GetParameterGradients override.
        try
        {
            var predicted = ForwardForTraining(input);

            var lossGrad = effectiveLossFunction.CalculateDerivative(
                predicted.ToVector(), target.ToVector());
            var lossGradTensor = new Tensor<T>(predicted._shape, lossGrad);

            BackpropagateLossGradient(lossGradTensor);

            var gradients = GetParameterGradients();

            bool hasValidGradients = false;
            for (int i = 0; i < Math.Min(gradients.Length, 100); i++)
            {
                if (!NumOps.Equals(gradients[i], NumOps.Zero))
                {
                    hasValidGradients = true;
                    break;
                }
            }

            if (hasValidGradients)
                return gradients;
        }
        catch (NotSupportedException ex)
        {
            // Subclass deliberately doesn't implement layer-level gradients (e.g. it
            // hasn't overridden BackpropagateLossGradient yet) — fall back to SPSA.
            System.Diagnostics.Trace.TraceWarning(
                $"VAE layer backpropagation not implemented, falling back to SPSA: {ex.Message}");
        }
        catch (NotImplementedException ex)
        {
            // Same intent as NotSupportedException; some subclasses use this variant.
            System.Diagnostics.Trace.TraceWarning(
                $"VAE layer backpropagation not implemented, falling back to SPSA: {ex.Message}");
        }
        // Other exceptions (shape bugs, broken overrides, serialization corruption,
        // null derefs from incomplete state) are real implementation bugs — let them
        // bubble up so regressions are caught at the test boundary instead of being
        // silently masked by the SPSA fallback.

        // Fallback: SPSA (6 forward passes total vs 2N for finite differences).
        // Snapshot parameters BEFORE the perturbation loop and always restore them in a
        // finally block — without that, an exception inside SetParameters/Predict/
        // CalculateLoss would exit with perturbed weights still installed and silently
        // corrupt later training/inference.
        var parameters = GetParameters();
        try
        {
            var gradients_spsa = new Vector<T>(parameters.Length);
            var epsilon = NumOps.FromDouble(1e-3);
            var twoEpsilon = NumOps.Multiply(epsilon, NumOps.FromDouble(2.0));
            var rng = RandomGenerator;
            var delta = new Vector<T>(parameters.Length);

            for (int s = 0; s < 3; s++)
            {
                for (int i = 0; i < parameters.Length; i++)
                    delta[i] = rng.NextDouble() < 0.5 ? NumOps.FromDouble(-1.0) : NumOps.FromDouble(1.0);

                var eDelta = Engine.Multiply(delta, epsilon);
                SetParameters(Engine.Add(parameters, eDelta));
                var lossPlus = effectiveLossFunction.CalculateLoss(Predict(input).ToVector(), target.ToVector());

                SetParameters(Engine.Subtract(parameters, eDelta));
                var lossMinus = effectiveLossFunction.CalculateLoss(Predict(input).ToVector(), target.ToVector());

                var lossDiff = NumOps.Subtract(lossPlus, lossMinus);
                var scaledDelta = Engine.Multiply(delta, twoEpsilon);
                gradients_spsa = Engine.Add(gradients_spsa, Engine.Divide(
                    Engine.Fill(parameters.Length, lossDiff), scaledDelta));
            }

            gradients_spsa = Engine.Multiply(gradients_spsa, NumOps.FromDouble(1.0 / 3.0));
            return gradients_spsa;
        }
        finally
        {
            // Always restore the original weights, even on exception.
            SetParameters(parameters);
        }
    }

    /// <summary>
    /// Runs the VAE forward pass (encode + decode) without suppressing tape recording.
    /// Used for tape-based training where the forward ops must be recorded.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The reconstructed output.</returns>
    protected virtual Tensor<T> ForwardForTraining(Tensor<T> input)
    {
        var latent = Encode(input, sampleMode: false);
        return Decode(latent);
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
    /// <remarks>
    /// Internal training plumbing; library users should drive training through
    /// <c>PredictionModelBuilder</c> and read results from <c>PredictionModelResult</c>
    /// instead of calling this directly.
    /// </remarks>
    protected Dictionary<Tensor<T>, Tensor<T>> ComputeGradientsWithTape(
        Tensor<T> input,
        Tensor<T> target,
        Tensor<T>[] trainableParams)
    {
        using var tape = new GradientTape<T>();

        // Forward pass (recorded by the engine)
        var predicted = ForwardForTraining(input);

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
    /// Pushes a loss gradient tensor (shape matching the decoder's output) back through
    /// the decoder and encoder layer chain so each layer's parameter gradient cache is
    /// populated. Concrete VAEs that don't have a tape-level backward path should
    /// throw <see cref="NotSupportedException"/> from this override; the
    /// <c>ComputeGradients</c> exception handler will catch it and fall through to SPSA.
    /// </summary>
    /// <remarks>
    /// Made abstract instead of a no-op virtual: a silent default would let concrete
    /// VAEs forget the override and quietly degrade to stale/zero gradients before
    /// reaching the SPSA fallback. Forcing every subclass to make an explicit choice
    /// (implement, or throw NotSupportedException) ensures the fallback path is hit
    /// only when the model author has acknowledged it.
    /// </remarks>
    /// <param name="lossGradient">
    /// dL/dy for the decoder's output. Shape must match what <see cref="ForwardForTraining"/>
    /// returned.
    /// </param>
    protected abstract void BackpropagateLossGradient(Tensor<T> lossGradient);

    /// <summary>
    /// Extracts accumulated parameter gradients from all encoder/decoder/norm layers after
    /// <see cref="BackpropagateLossGradient"/> has populated them. Concrete VAEs must walk
    /// their owned layers and concatenate <see cref="LayerBase{T}.GetParameterGradients"/>
    /// in the same order they expose their flat parameter vector via <see cref="GetParameters"/>.
    /// </summary>
    protected abstract Vector<T> GetParameterGradients();

    /// <inheritdoc />
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();
        parameters = Engine.Subtract(parameters, Engine.Multiply(gradients, learningRate));
        SetParameters(parameters);
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

        for (int i = 0; i < noiseSpan.Length; i++)
        {
            noiseSpan[i] = NumOps.FromDouble(rng.NextGaussian());
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
