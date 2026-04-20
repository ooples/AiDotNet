using System.Linq;
using AiDotNet.Autodiff;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Extensions;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Diffusion;

/// <summary>
/// Base class for diffusion-based generative models providing common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This abstract base class implements the common behavior for all diffusion models,
/// including the generation loop, noise addition, loss computation, and state management.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all diffusion models build upon.
/// It handles the common tasks that every diffusion model needs:
/// <list type="bullet">
/// <item><description>The generation loop (iteratively denoising from noise)</description></item>
/// <item><description>Adding noise during training</description></item>
/// <item><description>Computing the training loss</description></item>
/// <item><description>Saving and loading the model</description></item>
/// </list>
/// Specific diffusion models (like DDPM, Latent Diffusion) extend this base to implement
/// their unique noise prediction architectures.</para>
/// </remarks>
public abstract class DiffusionModelBase<T> : IDiffusionModel<T>, IConfigurableModel<T>, IModelShape, IDisposable
{
    /// <summary>
    /// Concrete diffusion models can override this method to yield the components
    /// they own that hold disposable resources — typically the noise predictor
    /// (DiT, UNet, MMDiT) plus, for latent diffusion, the VAE and conditioner.
    /// </summary>
    /// <remarks>
    /// <para>
    /// We use a method-based opt-in rather than a <c>NoisePredictor</c> property
    /// to avoid name collision with <see cref="ILatentDiffusionModel{T}.NoisePredictor"/>
    /// (which returns the non-nullable interface type and is part of an existing
    /// contract). Subclasses of <c>LatentDiffusionModelBase</c> can override
    /// this method to surface the same predictor for Dispose cleanup without
    /// changing their interface obligations.
    /// </para>
    /// <para>
    /// <b>Default behavior</b>: when a subclass does NOT override this, the base
    /// performs a reflection walk over its own and the subclass's instance
    /// fields and yields anything that implements <see cref="IDisposable"/>.
    /// This catches the common case (a private predictor field) without forcing
    /// every existing concrete model to override — but for predictable cleanup
    /// in performance-sensitive code, an explicit override remains preferred.
    /// </para>
    /// <example>
    /// <code>
    /// public class DDPMModel&lt;T&gt; : DiffusionModelBase&lt;T&gt; {
    ///     private readonly UNetNoisePredictor&lt;T&gt; _unet;
    ///     protected override IEnumerable&lt;IDisposable&gt; EnumerateDisposableComponents() {
    ///         yield return _unet;
    ///     }
    /// }
    /// </code>
    /// </example>
    /// </remarks>
    protected virtual IEnumerable<IDisposable> EnumerateDisposableComponents() =>
        ReflectInstanceDisposables(this);

    /// <summary>
    /// Walks an object's instance fields and yields anything that implements
    /// <see cref="IDisposable"/>. Used as the default fallback for
    /// <see cref="EnumerateDisposableComponents"/> so concrete subclasses don't
    /// need to override just to get correct cleanup.
    /// </summary>
    /// <remarks>
    /// Uses a <see cref="HashSet{Object}"/> of visited references to avoid
    /// double-yielding when the same disposable is reachable from multiple
    /// fields (e.g., a predictor stored as both an interface and a concrete
    /// type alias). Skips primitives, value types, strings, and the model's
    /// own scheduler (handled separately in Dispose).
    /// </remarks>
    private static IEnumerable<IDisposable> ReflectInstanceDisposables(object root)
    {
        var visited = new HashSet<object>(Helpers.TensorReferenceComparer<object>.Instance);
        if (!visited.Add(root)) yield break;

        var type = root.GetType();
        const System.Reflection.BindingFlags fieldFlags =
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.Public |
            System.Reflection.BindingFlags.NonPublic;
        for (var t = type; t != null && t != typeof(object); t = t.BaseType)
        {
            foreach (var field in t.GetFields(fieldFlags | System.Reflection.BindingFlags.DeclaredOnly))
            {
                if (field.FieldType.IsValueType || field.FieldType == typeof(string)) continue;
                // Skip _scheduler — Dispose(bool) handles it explicitly. Yielding
                // it here would cause a double-dispose attempt on the cascade.
                if (field.Name == "_scheduler") continue;
                object? value;
                try { value = field.GetValue(root); }
                catch (Exception ex)
                {
                    // Trace the read failure rather than silently skip — without
                    // this a private field whose getter throws would leak its
                    // disposable resource without any diagnostic trail.
                    System.Diagnostics.Trace.TraceWarning(
                        $"DiffusionModelBase.Dispose: skipping field '{field.Name}' " +
                        $"on {t.Name} due to reflection read failure: {ex.GetType().Name}: {ex.Message}");
                    continue;
                }
                if (value is null) continue;
                if (!visited.Add(value)) continue;
                if (value is IDisposable disposable)
                {
                    yield return disposable;
                }
                else if (value is System.Collections.IDictionary dictionary)
                {
                    // Dictionary<K, V>.GetEnumerator yields KeyValuePair<K,V>,
                    // not the values — so the generic IEnumerable branch below
                    // would MISS disposables held in the values slot. Handle
                    // IDictionary explicitly by walking values through
                    // DictionaryEntry, which gives us the value directly.
                    foreach (System.Collections.DictionaryEntry entry in dictionary)
                    {
                        if (entry.Value is IDisposable nested && visited.Add(entry.Value))
                            yield return nested;
                    }
                }
                else if (value is System.Collections.IEnumerable enumerable && value is not string)
                {
                    // Walk collections that hold disposables (e.g.,
                    // List<IDisposable>). Dictionary is handled above.
                    foreach (var item in enumerable)
                    {
                        if (item is IDisposable nested && visited.Add(item))
                            yield return nested;
                    }
                }
            }
        }
    }

    private bool _disposed;

    /// <summary>
    /// Provides access to the hardware-accelerated tensor engine.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// Provides numeric operations for the specific type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Random number generator for noise sampling.
    /// </summary>
    protected Random RandomGenerator;

    /// <summary>
    /// The step scheduler controlling the diffusion process.
    /// </summary>
    private readonly INoiseScheduler<T> _scheduler;

    /// <summary>
    /// The loss function used for training (typically MSE for noise prediction).
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// The configuration options for this diffusion model.
    /// </summary>
    private readonly DiffusionModelOptions<T> _options;

    /// <summary>
    /// Gets the configuration options for this model.
    /// </summary>
    protected ModelOptions Options => _options;

    /// <inheritdoc/>
    public virtual ModelOptions GetOptions() => _options;

    /// <summary>
    /// The optional neural network architecture blueprint for custom layer configuration.
    /// </summary>
    private readonly NeuralNetworkArchitecture<T>? _architecture;

    /// <summary>
    /// Active feature indices used by the model.
    /// </summary>
    private HashSet<int> _activeFeatureIndices = new HashSet<int>();

    /// <summary>
    /// The learning rate converted to type T for training computations.
    /// </summary>
    protected T LearningRate;

    /// <summary>
    /// Cached result of the reflection walk that discovers trainable parameter tensors.
    /// The walk was called per Train step, consuming a non-trivial amount of time on
    /// large models. Tensor references are stable (DenseLayer.SetParameters modifies in
    /// place) so caching is safe. Subclasses that swap layer references at runtime can
    /// invalidate via InvalidateTrainableParametersCache.
    /// </summary>
    private Tensor<T>[]? _cachedTrainableParameters;

    /// <inheritdoc />
    public INoiseScheduler<T> Scheduler => _scheduler;

    /// <inheritdoc />
    public abstract int ParameterCount { get; }

    /// <inheritdoc/>
    public virtual Vector<T> SanitizeParameters(Vector<T> parameters) => parameters;

    /// <inheritdoc/>
    public virtual bool SupportsParameterInitialization => ParameterCount > 0;

    /// <inheritdoc />
    public ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <summary>
    /// Gets the optional neural network architecture used for custom layer configuration.
    /// </summary>
    /// <remarks>
    /// When provided, derived models should check <c>Architecture.Layers</c> first before
    /// creating default layers. This allows users to supply custom layer configurations
    /// via <see cref="NeuralNetworkArchitecture{T}"/>.
    /// </remarks>
    public NeuralNetworkArchitecture<T>? Architecture => _architecture;

    /// <summary>
    /// Initializes a new instance of the DiffusionModelBase class.
    /// </summary>
    /// <param name="options">Configuration options for the diffusion model. If null, uses default options.</param>
    /// <param name="scheduler">Optional custom scheduler. If null, creates one from options.</param>
    /// <param name="architecture">
    /// Optional neural network architecture for custom layer configuration.
    /// When provided, derived models should check <c>Architecture.Layers</c> first before
    /// creating default layers. If null, models use their own research-paper defaults.
    /// </param>
    protected DiffusionModelBase(
        DiffusionModelOptions<T>? options = null,
        INoiseScheduler<T>? scheduler = null,
        NeuralNetworkArchitecture<T>? architecture = null)
    {
        _architecture = architecture;
        _options = options ?? new DiffusionModelOptions<T>();

        // Create scheduler from options if not provided
        if (scheduler != null)
        {
            _scheduler = scheduler;
        }
        else
        {
            var schedulerConfig = new SchedulerConfig<T>(
                trainTimesteps: _options.TrainTimesteps,
                betaStart: NumOps.FromDouble(_options.BetaStart),
                betaEnd: NumOps.FromDouble(_options.BetaEnd),
                betaSchedule: _options.BetaSchedule,
                clipSample: _options.ClipSample,
                predictionType: _options.PredictionType);
            _scheduler = new DDIMScheduler<T>(schedulerConfig);
        }

        // Set loss function from options or default
        LossFunction = _options.LossFunction ?? new MeanSquaredErrorLoss<T>();

        // Convert learning rate from double to T
        LearningRate = NumOps.FromDouble(_options.LearningRate);

        // Set up random generator
        RandomGenerator = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    #region IDiffusionModel<T> Implementation

    /// <inheritdoc />
    public virtual Tensor<T> Generate(int[] shape, int numInferenceSteps = 50, int? seed = null)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape must be a non-empty array.", nameof(shape));
        if (numInferenceSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(numInferenceSteps), "Must be positive.");

        // Validate all dimensions are positive
        var invalidDims = shape.Where(d => d <= 0).ToArray();
        if (invalidDims.Length > 0)
            throw new ArgumentOutOfRangeException(nameof(shape), $"All dimensions must be positive, but found {invalidDims[0]}.");

        // Suppress tape recording during inference (like PyTorch torch.no_grad())
        using var _ = new NoGradScope<T>();

        // Set up random generator
        var rng = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomGenerator;

        // Initialize with random noise using checked arithmetic to detect overflow
        long totalElements = 1;
        foreach (var dim in shape)
        {
            totalElements = checked(totalElements * dim);
        }

        if (totalElements > int.MaxValue)
            throw new ArgumentException("Total tensor size exceeds maximum supported size.", nameof(shape));

        var sample = SampleNoise((int)totalElements, rng);

        // Set up the scheduler for inference
        _scheduler.SetTimesteps(numInferenceSteps);

        // Pre-allocate reusable tensor for the denoising loop to avoid
        // creating a new Tensor per step (50 allocations → 1)
        var sampleTensor = new Tensor<T>(shape, sample);

        // Pre-allocate reusable noise prediction vector to avoid per-step allocation
        var noisePredVec = new Vector<T>(sample.Length);

        // Iterative denoising loop
        foreach (var timestep in _scheduler.Timesteps)
        {
            // Update tensor data in-place from sample vector using Span copy
            var sampleSpan = sample.AsSpan();
            var tensorSpan = sampleTensor.AsWritableSpan();
            sampleSpan.CopyTo(tensorSpan);

            // Predict the noise
            var noisePrediction = PredictNoise(sampleTensor, timestep);

            // Copy prediction to pre-allocated vector (avoids ToVector() allocation)
            var predSpan = noisePrediction.AsSpan();
            for (int idx = 0; idx < predSpan.Length && idx < noisePredVec.Length; idx++)
                noisePredVec[idx] = predSpan[idx];

            // Perform one denoising step
            // eta=0 for deterministic generation
            sample = _scheduler.Step(
                noisePredVec,
                timestep,
                sample,
                NumOps.Zero);
        }

        return new Tensor<T>(shape, sample);
    }

    /// <inheritdoc />
    public abstract Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep);

    /// <inheritdoc />
    public virtual T ComputeLoss(Tensor<T> cleanSamples, Tensor<T> noise, int[] timesteps)
    {
        if (cleanSamples == null)
            throw new ArgumentNullException(nameof(cleanSamples));
        if (noise == null)
            throw new ArgumentNullException(nameof(noise));
        if (timesteps == null || timesteps.Length == 0)
            throw new ArgumentException("Timesteps must be a non-empty array.", nameof(timesteps));

        var cleanVector = cleanSamples.ToVector();
        var noiseVector = noise.ToVector();

        // Add noise to clean samples at the given timesteps
        // For simplicity, we use the first timestep (batch processing would use different timesteps per sample)
        var noisySample = _scheduler.AddNoise(cleanVector, noiseVector, timesteps[0]);

        // Create tensor for noise prediction
        var noisySampleTensor = new Tensor<T>(cleanSamples._shape, noisySample);

        // Predict the noise
        var predictedNoise = PredictNoise(noisySampleTensor, timesteps[0]);

        // Compute MSE between predicted and actual noise
        return LossFunction.CalculateLoss(predictedNoise.ToVector(), noiseVector);
    }

    #endregion

    #region IModel<Tensor<T>, Tensor<T>, ModelMetadata<T>> Implementation

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Performs one training step using the denoising score matching objective.
    /// Computes gradients and updates model parameters using the configured learning rate.
    /// </para>
    /// <para><b>For Beginners:</b> This method performs a single training iteration:
    /// <list type="number">
    /// <item><description>Computes how wrong the model's noise predictions are (gradients)</description></item>
    /// <item><description>Adjusts the model's parameters using the learning rate to make better predictions</description></item>
    /// </list>
    /// You can control the step size by setting the LearningRate in the options.</para>
    /// </remarks>
    public virtual void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Tape-based direct per-tensor SGD step. The forward pass records Engine
        // ops onto the thread-local gradient tape, backward returns per-tensor
        // gradients, and we apply them in place via param -= lr * grad. This
        // bypasses the legacy flat-vector round-trip (GetParameters →
        // FlattenGradients → ApplyGradients → SetParameters) entirely — that
        // path doesn't work once the reflection walker discovers more trainable
        // tensors than GetParameters knows about, which is now the norm after
        // migrating layers like FlashAttentionLayer from Matrix<T> to Tensor<T>.

        // Sample a random timestep and build the noisy training sample.
        var timestep = RandomGenerator.Next(_scheduler.Config.TrainTimesteps);
        var inputVector = input.ToVector();
        var noiseVector = SampleNoise(inputVector.Length, RandomGenerator);
        var noisySample = _scheduler.AddNoise(inputVector, noiseVector, timestep);
        var noisySampleTensor = new Tensor<T>(input._shape, noisySample);

        using var tape = new GradientTape<T>();

        // Forward pass — triggers lazy layer initialization, then we walk for
        // trainable parameters. Collection must happen AFTER the forward pass so
        // newly-initialized layers are visible to the walker.
        var predicted = PredictNoise(noisySampleTensor, timestep);
        var paramTensors = CollectTrainableParameters();
        if (paramTensors.Length == 0)
        {
            throw new InvalidOperationException(
                $"{GetType().Name} has no trainable parameters discoverable via " +
                "CollectTrainableParameters. Make sure layers register their weights via " +
                "LayerBase.RegisterTrainableParameter so the gradient tape can reach them.");
        }

        // MSE loss against the true noise — tape-tracked.
        var noiseTensor = new Tensor<T>(predicted._shape, noiseVector);
        var diff = Engine.TensorSubtract(predicted, noiseTensor);
        var sq = Engine.TensorMultiply(diff, diff);
        var loss = Engine.ReduceSum(sq, null);

        // Backward pass via graph-based autodiff.
        var grads = tape.ComputeGradients(loss, paramTensors);

        // Per-tensor SGD: param -= lr * grad, applied in place so registered
        // tensor references stay stable across training steps.
        foreach (var param in paramTensors)
        {
            if (!grads.TryGetValue(param, out var grad) || grad is null) continue;
            var update = Engine.TensorMultiplyScalar(grad, LearningRate);
            var paramSpan = param.Data.Span;
            var updateSpan = update.AsSpan();
            int n = Math.Min(paramSpan.Length, updateSpan.Length);
            for (int i = 0; i < n; i++)
            {
                paramSpan[i] = NumOps.Subtract(paramSpan[i], updateSpan[i]);
            }
        }
    }

    /// <inheritdoc />
    public virtual Tensor<T> Predict(Tensor<T> input)
    {
        // For diffusion models, prediction is generating samples.
        // Use a deterministic seed derived from the input so Predict is reproducible
        // across clones and repeated calls with the same input.
        int seed = 0;
        for (int i = 0; i < Math.Min(input.Length, 16); i++)
        {
            seed = unchecked(seed * 31 + NumOps.ToDouble(input[i]).GetHashCode());
        }
        return Generate(input._shape, _options.DefaultInferenceSteps, seed);
    }

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = GetType().Name,
            FeatureCount = ParameterCount,
            Complexity = ParameterCount,
            Description = $"Diffusion model with {ParameterCount} parameters using {_scheduler.GetType().Name} scheduler."
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
        var clone = (DiffusionModelBase<T>)Clone();
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
        return new[] { ParameterCount };
    }

    /// <inheritdoc/>
    public virtual int[] GetOutputShape()
    {
        return new[] { ParameterCount };
    }

    /// <inheritdoc/>
    public virtual DynamicShapeInfo GetDynamicShapeInfo()
    {
        return DynamicShapeInfo.None;
    }


    /// <inheritdoc />
    public virtual void SaveModel(string filePath)
    {
        var data = Serialize();
        byte[] envelopedData = ModelFileHeader.WrapWithHeader(
            data, this, GetInputShape(), GetOutputShape(), SerializationFormat.Binary);
        File.WriteAllBytes(filePath, envelopedData);
    }

    /// <inheritdoc />
    public virtual void LoadModel(string filePath)
    {
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

        // Save scheduler config
        writer.Write(_scheduler.Config.TrainTimesteps);
        writer.Write(NumOps.ToDouble(_scheduler.Config.BetaStart));
        writer.Write(NumOps.ToDouble(_scheduler.Config.BetaEnd));
        writer.Write((int)_scheduler.Config.BetaSchedule);
        writer.Write((int)_scheduler.Config.PredictionType);
        writer.Write(_scheduler.Config.ClipSample);

        // Save model parameters using SerializationHelper
        SerializationHelper<T>.SerializeVector(writer, GetParameters());

        stream.Flush();
    }

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Loads model state from a stream, including scheduler configuration validation
    /// and model parameters. Throws if the saved scheduler config doesn't match
    /// the current instance's scheduler configuration.
    /// </para>
    /// <para><b>For Beginners:</b> This restores a previously saved model. The scheduler
    /// settings must match between the saved model and this instance to ensure
    /// the loaded parameters work correctly with the noise schedule.</para>
    /// </remarks>
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

        // Read and validate scheduler config
        var savedTrainTimesteps = reader.ReadInt32();
        var savedBetaStart = reader.ReadDouble();
        var savedBetaEnd = reader.ReadDouble();
        var savedBetaSchedule = (BetaSchedule)reader.ReadInt32();
        var savedPredictionType = (DiffusionPredictionType)reader.ReadInt32();
        var savedClipSample = reader.ReadBoolean();

        // Validate critical scheduler parameters match
        if (savedTrainTimesteps != _scheduler.Config.TrainTimesteps)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved TrainTimesteps={savedTrainTimesteps}, " +
                $"current={_scheduler.Config.TrainTimesteps}. Create a model with matching scheduler config.");
        }

        if (Math.Abs(savedBetaStart - NumOps.ToDouble(_scheduler.Config.BetaStart)) > 1e-9)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved BetaStart={savedBetaStart}, " +
                $"current={NumOps.ToDouble(_scheduler.Config.BetaStart)}. Create a model with matching scheduler config.");
        }

        if (Math.Abs(savedBetaEnd - NumOps.ToDouble(_scheduler.Config.BetaEnd)) > 1e-9)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved BetaEnd={savedBetaEnd}, " +
                $"current={NumOps.ToDouble(_scheduler.Config.BetaEnd)}. Create a model with matching scheduler config.");
        }

        if (savedBetaSchedule != _scheduler.Config.BetaSchedule)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved BetaSchedule={savedBetaSchedule}, " +
                $"current={_scheduler.Config.BetaSchedule}. Create a model with matching scheduler config.");
        }

        if (savedPredictionType != _scheduler.Config.PredictionType)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved PredictionType={savedPredictionType}, " +
                $"current={_scheduler.Config.PredictionType}. Create a model with matching scheduler config.");
        }

        if (savedClipSample != _scheduler.Config.ClipSample)
        {
            throw new InvalidOperationException(
                $"Scheduler config mismatch: saved ClipSample={savedClipSample}, " +
                $"current={_scheduler.Config.ClipSample}. Create a model with matching scheduler config.");
        }

        // Load model parameters using SerializationHelper
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
    /// Creates a deep copy of the model.
    /// </summary>
    /// <returns>A new instance with the same parameters.</returns>
    public abstract IDiffusionModel<T> Clone();

    #endregion

    #region IGradientComputable<T, Tensor<T>, Tensor<T>> Implementation

    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Computes gradients for diffusion model training using the denoising score matching objective.
    /// This default implementation uses automatic differentiation via GradientTape when available,
    /// with a fallback to numerical gradients. Derived classes can override for custom gradient computation.
    /// </para>
    /// <para><b>For Beginners:</b> This is how the diffusion model learns:
    /// <list type="bullet">
    /// <item><description>Take a clean sample and add random noise</description></item>
    /// <item><description>Try to predict what noise was added</description></item>
    /// <item><description>Measure how wrong the prediction was (the loss)</description></item>
    /// <item><description>Figure out how to adjust parameters to be less wrong (the gradients)</description></item>
    /// </list></para>
    /// </remarks>
    public virtual Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));
        if (target == null)
            throw new ArgumentNullException(nameof(target));

        // Sample a random timestep
        var timestep = RandomGenerator.Next(_scheduler.Config.TrainTimesteps);

        // Sample noise and build the noisy sample
        var inputVector = input.ToVector();
        var noiseVector = SampleNoise(inputVector.Length, RandomGenerator);
        var noisySample = _scheduler.AddNoise(inputVector, noiseVector, timestep);
        var noisySampleTensor = new Tensor<T>(input._shape, noisySample);

        // Tape-based automatic differentiation. Forward pass runs first so lazy
        // layer initialization (DiTNoisePredictor.EnsureLayersInitialized etc.)
        // fires before we walk for trainable parameters. Every Engine op in the
        // forward records a GradFn entry, so tape.ComputeGradients returns exact
        // per-tensor gradients without requiring a manual backward.
        using var tape = new GradientTape<T>();
        var predicted = PredictNoise(noisySampleTensor, timestep);
        var paramTensors = CollectTrainableParameters();
        if (paramTensors.Length == 0)
        {
            throw new InvalidOperationException(
                $"{GetType().Name} has no trainable parameters discoverable via " +
                "CollectTrainableParameters. Make sure layers register their weights via " +
                "LayerBase.RegisterTrainableParameter so the gradient tape can reach them.");
        }

        var noiseTensor = new Tensor<T>(predicted._shape, noiseVector);
        var diff = Engine.TensorSubtract(predicted, noiseTensor);
        var sq = Engine.TensorMultiply(diff, diff);
        var loss = Engine.ReduceSum(sq, null);
        var grads = tape.ComputeGradients(loss, paramTensors);

        // Flatten gradients into a single vector matching the tape-collected
        // parameter order. External callers that go through IGradientComputable
        // still get a flat Vector<T>, though the internal Train path prefers the
        // per-tensor direct apply via TryTapeDirectTrainStep to avoid the flat
        // round-trip entirely.
        return FlattenGradients(paramTensors, grads);
    }

    /// <summary>
    /// Collects all trainable parameter tensors from the noise predictor's layers.
    /// Used by tape-based training to identify which tensors need gradients.
    /// </summary>
    protected virtual Tensor<T>[] CollectTrainableParameters()
    {
        // Cached reflection walk: the walker traverses the full object graph to
        // find every ITrainableLayer's parameter tensors. Layer structure and
        // tensor references are stable after construction (DenseLayer.SetParameters
        // modifies in place), so we only need to walk once per model instance.
        if (_cachedTrainableParameters is not null)
            return _cachedTrainableParameters;

        var allParams = new List<Tensor<T>>();
        CollectLayerParameters(this, allParams, new HashSet<object>(AiDotNet.Helpers.TensorReferenceComparer<object>.Instance));

        // Only cache non-empty results. An empty result usually means lazy
        // initialization hasn't run yet — don't pin that empty list.
        if (allParams.Count > 0)
            _cachedTrainableParameters = allParams.ToArray();

        return allParams.ToArray();
    }

    /// <summary>
    /// Invalidates the cached trainable-parameter walk. Call this from subclasses
    /// that swap layer references at runtime so the next training step re-discovers
    /// the updated structure.
    /// </summary>
    protected void InvalidateTrainableParametersCache()
    {
        _cachedTrainableParameters = null;
    }

    private void CollectLayerParameters(object? obj, List<Tensor<T>> allParams, HashSet<object> visited)
    {
        if (obj is null || !visited.Add(obj)) return;

        if (obj is Interfaces.ITrainableLayer<T> trainable)
        {
            var parameters = trainable.GetTrainableParameters();
            if (parameters is not null)
            {
                foreach (var p in parameters)
                    if (p is not null && p.Length > 0) allParams.Add(p);
            }
        }

        // Recurse into every reference-type instance field so nested composites
        // (e.g., DiffusionModel -> UNetNoisePredictor -> List<Layer>) are fully
        // walked even when the intermediate types don't implement ITrainableLayer.
        // The visited set handles cycles.
        var type = obj.GetType();
        if (type.IsPrimitive || type == typeof(string) || type.IsEnum) return;

        foreach (var field in type.GetFields(
            System.Reflection.BindingFlags.Instance |
            System.Reflection.BindingFlags.NonPublic |
            System.Reflection.BindingFlags.Public))
        {
            // Skip compiler-generated backing fields for non-ref properties and
            // fields whose declared type can't hold a trainable layer.
            if (field.FieldType.IsPrimitive || field.FieldType.IsEnum ||
                field.FieldType == typeof(string) || field.FieldType == typeof(Tensor<T>))
                continue;

            var val = field.GetValue(obj);
            if (val is null) continue;

            if (val is System.Collections.IEnumerable enumerable && val is not string)
            {
                foreach (var item in enumerable)
                    CollectLayerParameters(item, allParams, visited);
            }
            else
            {
                CollectLayerParameters(val, allParams, visited);
            }
        }
    }

    /// <summary>
    /// Flattens gradient tensors into a single vector matching GetParameters() layout.
    /// </summary>
    private Vector<T> FlattenGradients(Tensor<T>[] paramTensors, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        int totalSize = 0;
        foreach (var p in paramTensors) totalSize += p.Length;

        var flat = new Vector<T>(totalSize);
        int offset = 0;
        foreach (var p in paramTensors)
        {
            if (grads.TryGetValue(p, out var grad))
            {
                var gradSpan = grad.AsSpan();
                for (int i = 0; i < gradSpan.Length; i++)
                    flat[offset + i] = gradSpan[i];
            }
            offset += p.Length;
        }
        return flat;
    }

    /// <inheritdoc />
    public virtual void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        var parameters = GetParameters();

        // Vectorized SGD: params = params - lr * gradients
        var scaledGradients = Engine.Multiply(gradients, learningRate);
        parameters = Engine.Subtract(parameters, scaledGradients);

        SetParameters(parameters);
    }

    #endregion


    #region Helper Methods

    /// <summary>
    /// Samples random noise from a standard normal distribution.
    /// </summary>
    /// <param name="length">The number of elements to sample.</param>
    /// <param name="rng">The random number generator to use.</param>
    /// <returns>A vector of random noise values.</returns>
    protected virtual Vector<T> SampleNoise(int length, Random rng)
    {
        var noise = new Vector<T>(length);

        for (int i = 0; i < length; i++)
        {
            noise[i] = NumOps.FromDouble(rng.NextGaussian());
        }

        return noise;
    }

    #endregion

    #region IDisposable

    /// <inheritdoc />
    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Cascades Dispose to every disposable component the concrete model exposes
    /// via <see cref="EnumerateDisposableComponents"/> (default: reflection walk
    /// over instance fields), plus the owned <c>_scheduler</c>. Concrete diffusion
    /// models that want to constrain WHAT gets disposed (e.g., skip injected
    /// dependencies they don't own) override <see cref="EnumerateDisposableComponents"/>
    /// to return an explicit allow-list. Models that hold additional disposable
    /// composites beyond reflection-walk reach can also override this method and
    /// call <c>base.Dispose(disposing)</c>.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed || !disposing) return;
        _disposed = true;

        // Always dispose the scheduler we own — schedulers may hold buffers
        // (precomputed alpha/beta arrays, native handles for accelerated
        // sampling) that survive model disposal otherwise. Route through the
        // guard so the scheduler instance is disposed at most once even if
        // another owner also cascades into it.
        if (_scheduler is IDisposable disposableScheduler)
        {
            AiDotNet.Helpers.DisposeOnceGuard.TryDispose(disposableScheduler);
        }

        // Cascade to every disposable component the concrete model exposes via
        // EnumerateDisposableComponents (default: reflection walk over instance
        // fields). Shared components (a predictor reused across two diffusion
        // wrappers for ensembling, a VAE loaded once and injected into several
        // models) are common — the guard ensures each instance is disposed
        // exactly once regardless of how many cascades reach it. Many
        // components aren't idempotent on double-Dispose (they'd double-return
        // pooled buffers or crash on null derefs), which is why a plain
        // try/catch around ObjectDisposedException is insufficient.
        foreach (var component in EnumerateDisposableComponents())
        {
            if (component is null) continue;
            AiDotNet.Helpers.DisposeOnceGuard.TryDispose(component);
        }
    }

    #endregion
}
