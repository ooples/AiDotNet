using System.Linq;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Extensions;
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
public abstract class DiffusionModelBase<T> : IDiffusionModel<T>, IConfigurableModel<T>
{
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

    /// <inheritdoc />
    public INoiseScheduler<T> Scheduler => _scheduler;

    /// <inheritdoc />
    public abstract int ParameterCount { get; }

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

    /// <inheritdoc />
    public virtual bool SupportsJitCompilation => false;

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

        // Iterative denoising loop
        foreach (var timestep in _scheduler.Timesteps)
        {
            // Convert vector to tensor for noise prediction
            var sampleTensor = new Tensor<T>(shape, sample);

            // Predict the noise
            var noisePrediction = PredictNoise(sampleTensor, timestep);

            // Perform one denoising step
            // eta=0 for deterministic generation
            sample = _scheduler.Step(
                noisePrediction.ToVector(),
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
        var noisySampleTensor = new Tensor<T>(cleanSamples.Shape, noisySample);

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
        // For diffusion models, training involves:
        // 1. Sample random timesteps
        // 2. Add noise to input at those timesteps
        // 3. Predict the noise
        // 4. Update parameters to minimize prediction error

        // Compute gradients using the denoising score matching objective
        var gradients = ComputeGradients(input, expectedOutput, LossFunction);

        // Apply gradients using the configured learning rate
        ApplyGradients(gradients, LearningRate);
    }

    /// <inheritdoc />
    public virtual Tensor<T> Predict(Tensor<T> input)
    {
        // For diffusion models, prediction is generating samples
        // Use the input shape for generation
        return Generate(input.Shape, _options.DefaultInferenceSteps);
    }

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = GetType().Name,
            ModelType = ModelType.NeuralNetwork,
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

        var effectiveLossFunction = lossFunction ?? LossFunction;

        // Sample a random timestep
        var timestep = RandomGenerator.Next(_scheduler.Config.TrainTimesteps);

        // Sample noise
        var inputVector = input.ToVector();
        var noiseVector = SampleNoise(inputVector.Length, RandomGenerator);

        // Add noise to the clean sample using the scheduler
        var noisySample = _scheduler.AddNoise(inputVector, noiseVector, timestep);
        var noisySampleTensor = new Tensor<T>(input.Shape, noisySample);

        // Get current parameters
        var parameters = GetParameters();
        var gradients = new Vector<T>(parameters.Length);

        // Try to use autodiff if computation graph is available
        try
        {
            using var tape = new GradientTape<T>();

            // Create computation nodes for parameters
            var paramNodes = new List<ComputationNode<T>>();
            for (int i = 0; i < parameters.Length; i++)
            {
                var paramVector = new Vector<T>(1) { [0] = parameters[i] };
                var paramTensor = new Tensor<T>(new[] { 1 }, paramVector);
                var paramNode = TensorOperations<T>.Variable(paramTensor, $"param_{i}");
                tape.Watch(paramNode);
                paramNodes.Add(paramNode);
            }

            // Forward pass: predict noise
            var predictedNoise = PredictNoise(noisySampleTensor, timestep);

            // Compute loss using the effective loss function
            var loss = effectiveLossFunction.CalculateLoss(predictedNoise.ToVector(), noiseVector);

            // Create loss node for autodiff
            var lossVector = new Vector<T>(1) { [0] = loss };
            var lossTensor = new Tensor<T>(new[] { 1 }, lossVector);
            var lossNode = TensorOperations<T>.Variable(lossTensor, "loss", requiresGradient: false);

            // Compute gradients via autodiff
            var gradientDict = tape.Gradient(lossNode, paramNodes);

            // Extract gradients
            foreach (var kvp in gradientDict)
            {
                var idx = paramNodes.IndexOf(kvp.Key);
                if (idx >= 0 && idx < gradients.Length)
                {
                    gradients[idx] = kvp.Value[0];
                }
            }

            // Check if autodiff produced valid gradients
            bool hasValidGradients = false;
            for (int i = 0; i < gradients.Length; i++)
            {
                if (!NumOps.Equals(gradients[i], NumOps.Zero))
                {
                    hasValidGradients = true;
                    break;
                }
            }

            if (hasValidGradients)
            {
                return gradients;
            }
        }
        catch (InvalidOperationException)
        {
            // GradientTape may throw if computation graph is not properly built
            // Fall through to numerical gradients
        }
        catch (NotSupportedException)
        {
            // Some tensor operations may not support autodiff
            // Fall through to numerical gradients
        }

        // Fallback: Numerical gradient computation using finite differences
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
            var predictedPlus = PredictNoise(noisySampleTensor, timestep);
            var lossPlus = effectiveLossFunction.CalculateLoss(predictedPlus.ToVector(), noiseVector);

            // Compute f(x - epsilon)
            var paramsMinus = new Vector<T>(parameters.Length);
            for (int j = 0; j < parameters.Length; j++)
            {
                paramsMinus[j] = j == i ? NumOps.Subtract(parameters[j], epsilon) : parameters[j];
            }
            SetParameters(paramsMinus);
            var predictedMinus = PredictNoise(noisySampleTensor, timestep);
            var lossMinus = effectiveLossFunction.CalculateLoss(predictedMinus.ToVector(), noiseVector);

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
        throw new NotSupportedException("This diffusion model does not support JIT compilation. Override ExportComputationGraph in derived class if needed.");
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
}
