using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;
using System.Text;

namespace AiDotNet.FineTuning;

/// <summary>
/// Base class for all fine-tuning methods.
/// </summary>
/// <remarks>
/// <para>
/// This abstract class provides common functionality shared by all fine-tuning implementations,
/// including serialization, option management, and utility methods.
/// </para>
/// <para><b>For Beginners:</b> This is the foundation that all specific fine-tuning methods
/// (like DPO, RLHF, SimPO) build upon. It provides common functionality so each method
/// only needs to implement its unique training logic.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public abstract class FineTuningBase<T, TInput, TOutput> : IFineTuning<T, TInput, TOutput>
{
    /// <summary>
    /// The numeric operations helper for type T.
    /// </summary>
    protected static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// The configuration options for this fine-tuning method.
    /// </summary>
    protected FineTuningOptions<T> Options;

    /// <summary>
    /// Random number generator for training.
    /// </summary>
    protected Random Random;

    /// <summary>
    /// Metrics collected during training.
    /// </summary>
    protected FineTuningMetrics<T> CurrentMetrics;

    /// <summary>
    /// Initializes a new instance of the fine-tuning base class.
    /// </summary>
    /// <param name="options">The fine-tuning configuration options.</param>
    protected FineTuningBase(FineTuningOptions<T> options)
    {
        Options = options ?? throw new ArgumentNullException(nameof(options));
        Random = options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(options.RandomSeed.Value)
            : RandomHelper.CreateSeededRandom(Environment.TickCount);
        CurrentMetrics = new FineTuningMetrics<T>();
    }

    /// <inheritdoc/>
    public abstract string MethodName { get; }

    /// <inheritdoc/>
    public abstract FineTuningCategory Category { get; }

    /// <inheritdoc/>
    public abstract bool RequiresRewardModel { get; }

    /// <inheritdoc/>
    public abstract bool RequiresReferenceModel { get; }

    /// <inheritdoc/>
    public virtual bool SupportsPEFT => true;

    /// <inheritdoc/>
    public abstract Task<IFullModel<T, TInput, TOutput>> FineTuneAsync(
        IFullModel<T, TInput, TOutput> baseModel,
        FineTuningData<T, TInput, TOutput> trainingData,
        CancellationToken cancellationToken = default);

    /// <inheritdoc/>
    public abstract Task<FineTuningMetrics<T>> EvaluateAsync(
        IFullModel<T, TInput, TOutput> model,
        FineTuningData<T, TInput, TOutput> evaluationData,
        CancellationToken cancellationToken = default);

    /// <inheritdoc/>
    public FineTuningOptions<T> GetOptions() => Options;

    /// <inheritdoc/>
    public virtual void Reset()
    {
        CurrentMetrics = new FineTuningMetrics<T>();
        Random = Options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(Options.RandomSeed.Value)
            : RandomHelper.CreateSeededRandom(Environment.TickCount);
    }

    /// <inheritdoc/>
    public virtual byte[] Serialize()
    {
        var json = JsonConvert.SerializeObject(Options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public virtual void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var json = Encoding.UTF8.GetString(data);
        Options = JsonConvert.DeserializeObject<FineTuningOptions<T>>(json) ?? new FineTuningOptions<T>();
    }

    /// <inheritdoc/>
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

        File.WriteAllBytes(fullPath, Serialize());
    }

    /// <inheritdoc/>
    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Fine-tuning configuration file not found.", fullPath);
        }

        Deserialize(File.ReadAllBytes(fullPath));
    }

    /// <summary>
    /// Validates that the training data is appropriate for this fine-tuning method.
    /// </summary>
    /// <param name="data">The training data to validate.</param>
    /// <exception cref="ArgumentException">Thrown if the data is invalid for this method.</exception>
    protected virtual void ValidateTrainingData(FineTuningData<T, TInput, TOutput> data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        if (data.Count == 0)
        {
            throw new ArgumentException("Training data cannot be empty.", nameof(data));
        }
    }

    /// <summary>
    /// Creates batches from the training data.
    /// </summary>
    /// <param name="data">The full training data.</param>
    /// <param name="batchSize">The batch size.</param>
    /// <param name="shuffle">Whether to shuffle the data before batching.</param>
    /// <returns>An enumerable of data batches.</returns>
    protected virtual IEnumerable<FineTuningData<T, TInput, TOutput>> CreateBatches(
        FineTuningData<T, TInput, TOutput> data,
        int batchSize,
        bool shuffle = true)
    {
        var indices = Enumerable.Range(0, data.Count).ToArray();

        if (shuffle)
        {
            // Fisher-Yates shuffle
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = Random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        for (int i = 0; i < indices.Length; i += batchSize)
        {
            var batchIndices = indices.Skip(i).Take(batchSize).ToArray();
            yield return data.Subset(batchIndices);
        }
    }

    /// <summary>
    /// Computes the log probability of an output given an input.
    /// </summary>
    /// <param name="model">The model to use for computation.</param>
    /// <param name="input">The input.</param>
    /// <param name="output">The output to compute log probability for.</param>
    /// <returns>The log probability.</returns>
    protected virtual double ComputeLogProbability(
        IFullModel<T, TInput, TOutput> model,
        TInput input,
        TOutput output)
    {
        // This is a placeholder - actual implementation depends on model type
        // and output representation (tokens, vectors, etc.)
        var prediction = model.Predict(input);
        return ComputeLogProbabilityFromPrediction(prediction, output);
    }

    /// <summary>
    /// Computes log probability from a prediction and target output.
    /// </summary>
    /// <param name="prediction">The model's prediction.</param>
    /// <param name="target">The target output.</param>
    /// <returns>The log probability.</returns>
    protected virtual double ComputeLogProbabilityFromPrediction(TOutput prediction, TOutput target)
    {
        // Default implementation for vector-based outputs
        // Subclasses should override for specific output types
        return 0.0;
    }

    /// <summary>
    /// Computes the KL divergence between two probability distributions.
    /// </summary>
    /// <param name="policyProbs">The policy probabilities.</param>
    /// <param name="referenceProbs">The reference probabilities.</param>
    /// <returns>The KL divergence.</returns>
    protected virtual double ComputeKLDivergence(double[] policyProbs, double[] referenceProbs)
    {
        if (policyProbs.Length != referenceProbs.Length)
        {
            throw new ArgumentException("Probability arrays must have the same length.");
        }

        double kl = 0.0;
        for (int i = 0; i < policyProbs.Length; i++)
        {
            if (policyProbs[i] > 1e-10 && referenceProbs[i] > 1e-10)
            {
                kl += policyProbs[i] * Math.Log(policyProbs[i] / referenceProbs[i]);
            }
        }

        return kl;
    }

    /// <summary>
    /// Applies the sigmoid function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The sigmoid output.</returns>
    protected static double Sigmoid(double x)
    {
        if (x >= 0)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
        else
        {
            var exp = Math.Exp(x);
            return exp / (1.0 + exp);
        }
    }

    /// <summary>
    /// Applies the log sigmoid function.
    /// </summary>
    /// <param name="x">The input value.</param>
    /// <returns>The log sigmoid output.</returns>
    protected static double LogSigmoid(double x)
    {
        return x >= 0
            ? -Math.Log(1.0 + Math.Exp(-x))
            : x - Math.Log(1.0 + Math.Exp(x));
    }

    /// <summary>
    /// Updates training metrics with batch results.
    /// </summary>
    /// <param name="batchLoss">The loss for the current batch.</param>
    /// <param name="step">The current training step.</param>
    protected virtual void UpdateMetrics(double batchLoss, int step)
    {
        CurrentMetrics.LossHistory.Add(batchLoss);
        CurrentMetrics.TrainingLoss = CurrentMetrics.LossHistory.Average();
        CurrentMetrics.TrainingSteps = step;
    }

    /// <summary>
    /// Logs training progress.
    /// </summary>
    /// <param name="step">The current step.</param>
    /// <param name="totalSteps">The total number of steps.</param>
    /// <param name="loss">The current loss.</param>
    /// <param name="additionalInfo">Optional additional information to log.</param>
    protected virtual void LogProgress(int step, int totalSteps, double loss, string? additionalInfo = null)
    {
        if (step % Options.LoggingSteps == 0)
        {
            var message = $"[{MethodName}] Step {step}/{totalSteps} - Loss: {loss:F4}";
            if (!string.IsNullOrEmpty(additionalInfo))
            {
                message += $" - {additionalInfo}";
            }

            // In a real implementation, this would use a proper logging framework
            Console.WriteLine(message);
        }
    }
}
