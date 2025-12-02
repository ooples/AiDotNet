using AiDotNet.Autodiff;
using AiDotNet.Diffusion.Schedulers;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;

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
/// <para>
/// <b>For Beginners:</b> This is the foundation that all diffusion models build upon.
/// It handles the common tasks that every diffusion model needs:
/// - The generation loop (iteratively denoising from noise)
/// - Adding noise during training
/// - Computing the training loss
/// - Saving and loading the model
///
/// Specific diffusion models (like DDPM, Latent Diffusion) extend this base to implement
/// their unique noise prediction architectures.
/// </para>
/// </remarks>
public abstract class DiffusionModelBase<T> : IDiffusionModel<T>
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
    private readonly IStepScheduler<T> _scheduler;

    /// <summary>
    /// The loss function used for training (typically MSE for noise prediction).
    /// </summary>
    protected readonly ILossFunction<T> LossFunction;

    /// <summary>
    /// Active feature indices used by the model.
    /// </summary>
    private HashSet<int> _activeFeatureIndices = [];

    /// <inheritdoc />
    public IStepScheduler<T> Scheduler => _scheduler;

    /// <inheritdoc />
    public abstract int ParameterCount { get; }

    /// <inheritdoc />
    public ILossFunction<T> DefaultLossFunction => LossFunction;

    /// <inheritdoc />
    public virtual bool SupportsJitCompilation => false;

    /// <summary>
    /// Initializes a new instance of the DiffusionModelBase class.
    /// </summary>
    /// <param name="scheduler">The step scheduler for the diffusion process. If null, uses DDIM scheduler with default config.</param>
    /// <param name="lossFunction">Optional loss function. Defaults to Mean Squared Error.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    protected DiffusionModelBase(IStepScheduler<T>? scheduler = null, ILossFunction<T>? lossFunction = null, int? seed = null)
    {
        _scheduler = scheduler ?? new DDIMScheduler<T>(SchedulerConfig<T>.CreateDefault());
        LossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        RandomGenerator = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    #region IDiffusionModel<T> Implementation

    /// <inheritdoc />
    public virtual Tensor<T> Generate(int[] shape, int numInferenceSteps = 50, int? seed = null)
    {
        if (shape == null || shape.Length == 0)
            throw new ArgumentException("Shape must be a non-empty array.", nameof(shape));
        if (numInferenceSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(numInferenceSteps), "Must be positive.");

        // Set up random generator
        var rng = seed.HasValue ? new Random(seed.Value) : RandomGenerator;

        // Initialize with random noise
        int totalElements = 1;
        foreach (var dim in shape)
            totalElements *= dim;

        var sample = SampleNoise(totalElements, rng);

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
    public virtual void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // For diffusion models, training involves:
        // 1. Sample random timesteps
        // 2. Add noise to input at those timesteps
        // 3. Predict the noise
        // 4. Update parameters to minimize prediction error
        // This is a simplified implementation - derived classes can override for specific training logic
        var timestep = RandomGenerator.Next(_scheduler.Config.TrainTimesteps);
        _ = ComputeLoss(input, expectedOutput, [timestep]);
    }

    /// <inheritdoc />
    public virtual Tensor<T> Predict(Tensor<T> input)
    {
        // For diffusion models, prediction is generating samples
        // Use the input shape for generation
        return Generate(input.Shape);
    }

    /// <inheritdoc />
    public virtual ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = GetType().Name,
            ModelType = ModelType.NeuralNetwork, // Diffusion models are generative neural network models
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

        // Save scheduler config (not mutable state - scheduler is recreated from config)
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

        // Read scheduler config (skip - we use the existing scheduler's config)
        _ = reader.ReadInt32();  // TrainTimesteps
        _ = reader.ReadDouble(); // BetaStart
        _ = reader.ReadDouble(); // BetaEnd
        _ = reader.ReadInt32();  // BetaSchedule
        _ = reader.ReadInt32();  // PredictionType
        _ = reader.ReadBoolean(); // ClipSample

        // Load model parameters using SerializationHelper
        SetParameters(SerializationHelper<T>.DeserializeVector(reader));
    }

    #endregion

    #region IFeatureAware Implementation

    /// <inheritdoc />
    public virtual IEnumerable<int> GetActiveFeatureIndices()
    {
        // For diffusion models, all input features are typically used
        if (_activeFeatureIndices.Count == 0)
        {
            // Default: assume all features up to ParameterCount are active
            for (int i = 0; i < ParameterCount; i++)
            {
                _activeFeatureIndices.Add(i);
            }
        }
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
        return _activeFeatureIndices.Contains(featureIndex);
    }

    #endregion

    #region IFeatureImportance<T> Implementation

    /// <inheritdoc />
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        // For diffusion models, feature importance is typically uniform
        // Derived classes can override for more sophisticated importance measures
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
    public virtual Vector<T> ComputeGradients(Tensor<T> input, Tensor<T> target, ILossFunction<T>? lossFunction = null)
    {
        // For diffusion models, gradients are computed based on noise prediction error
        // This is a simplified implementation - real implementations would use autodiff
        var loss = lossFunction ?? LossFunction;

        // Sample a random timestep
        var timestep = RandomGenerator.Next(_scheduler.Config.TrainTimesteps);

        // Add noise to input
        var noise = SampleNoise(input.ToVector().Length, RandomGenerator);
        var noisyInput = _scheduler.AddNoise(input.ToVector(), noise, timestep);

        // Predict noise
        var noisySampleTensor = new Tensor<T>(input.Shape, noisyInput);
        var predictedNoise = PredictNoise(noisySampleTensor, timestep);

        // Compute gradient (simplified: difference between predicted and actual noise)
        var gradients = new Vector<T>(ParameterCount);
        var diff = predictedNoise.ToVector();

        for (int i = 0; i < Math.Min(gradients.Length, diff.Length); i++)
        {
            gradients[i] = NumOps.Subtract(diff[i], noise[i]);
        }

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
        // Diffusion models typically don't support JIT compilation due to their iterative nature
        // Derived classes can override if they support it
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

        // Box-Muller transform for normal distribution
        for (int i = 0; i < length; i += 2)
        {
            var u1 = rng.NextDouble();
            var u2 = rng.NextDouble();

            // Avoid log(0)
            while (u1 <= double.Epsilon)
                u1 = rng.NextDouble();

            var mag = Math.Sqrt(-2.0 * Math.Log(u1));
            var z0 = mag * Math.Cos(2.0 * Math.PI * u2);
            var z1 = mag * Math.Sin(2.0 * Math.PI * u2);

            noise[i] = NumOps.FromDouble(z0);
            if (i + 1 < length)
                noise[i + 1] = NumOps.FromDouble(z1);
        }

        return noise;
    }

    #endregion
}
