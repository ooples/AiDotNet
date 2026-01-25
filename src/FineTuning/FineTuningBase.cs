using System.Text;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

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
    /// <remarks>
    /// <para>
    /// This method computes a log probability score that measures how well the model's
    /// prediction matches the target. Higher values indicate better match.
    /// </para>
    /// <para>
    /// The implementation handles different output types:
    /// - Arrays/IEnumerable: Computes cross-entropy-like score for probability distributions
    /// - Numeric types: Computes negative squared error converted to log probability
    /// - Other types: Uses equality-based scoring
    /// </para>
    /// <para><b>For Beginners:</b> This is like a "similarity score" between what the model
    /// predicted and what we wanted it to predict. A score of 0 means perfect match,
    /// negative values indicate worse matches. The more negative, the worse the match.
    /// </para>
    /// </remarks>
    protected virtual double ComputeLogProbabilityFromPrediction(TOutput prediction, TOutput target)
    {
        if (prediction == null || target == null)
        {
            return double.NegativeInfinity;
        }

        // Handle string/text outputs FIRST (before IEnumerable, since strings implement IEnumerable<char>)
        if (prediction is string predStr && target is string targetStr)
        {
            return ComputeStringLogProbability(predStr, targetStr);
        }

        // Handle array types (probability distributions, embeddings)
        if (prediction is Array predArray && target is Array targetArray)
        {
            return ComputeArrayLogProbability(predArray, targetArray);
        }

        // Handle IEnumerable types (strings already handled above)
        if (prediction is System.Collections.IEnumerable predEnum && target is System.Collections.IEnumerable targetEnum)
        {
            var predList = predEnum.Cast<object>().ToArray();
            var targetList = targetEnum.Cast<object>().ToArray();
            if (predList.Length > 0 && targetList.Length > 0)
            {
                return ComputeEnumerableLogProbability(predList, targetList);
            }
        }

        // Handle numeric types directly
        if (IsNumericType(prediction.GetType()))
        {
            double predVal = Convert.ToDouble(prediction);
            double targetVal = Convert.ToDouble(target);
            return ComputeScalarLogProbability(predVal, targetVal);
        }

        // Default: exact match gives 0 (log(1)), mismatch gives large negative
        return prediction.Equals(target) ? 0.0 : -10.0;
    }

    /// <summary>
    /// Computes log probability for array outputs (probability distributions or embeddings).
    /// </summary>
    private double ComputeArrayLogProbability(Array predArray, Array targetArray)
    {
        if (predArray.Length == 0 || targetArray.Length == 0)
        {
            return double.NegativeInfinity;
        }

        // Convert to double arrays
        var pred = new double[predArray.Length];
        var target = new double[targetArray.Length];

        try
        {
            for (int i = 0; i < predArray.Length; i++)
            {
                pred[i] = Convert.ToDouble(predArray.GetValue(i));
            }
            for (int i = 0; i < targetArray.Length; i++)
            {
                target[i] = Convert.ToDouble(targetArray.GetValue(i));
            }
        }
        catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException)
        {
            // If conversion fails, fall back to equality check using SequenceEqual
            return predArray.Cast<object>().SequenceEqual(targetArray.Cast<object>()) ? 0.0 : -10.0;
        }

        // If lengths differ, use the minimum length and penalize
        int minLen = Math.Min(pred.Length, target.Length);
        double lengthPenalty = Math.Abs(pred.Length - target.Length) * -0.1;

        // Special case: single-element arrays should be treated as scalars
        // because cosine similarity is always 1.0 for same-sign single values
        if (minLen == 1)
        {
            return ComputeScalarLogProbability(pred[0], target[0]) + lengthPenalty;
        }

        // Check if prediction looks like a probability distribution (sums close to 1, all non-negative)
        double predSum = pred.Take(minLen).Sum();
        bool isProbDist = pred.Take(minLen).All(p => p >= 0) && Math.Abs(predSum - 1.0) < 0.1;

        if (isProbDist)
        {
            // Cross-entropy style: -sum(target * log(pred))
            // This is the standard log probability for probability distributions
            double logProb = 0.0;
            for (int i = 0; i < minLen; i++)
            {
                double p = Math.Max(pred[i], 1e-10); // Avoid log(0)
                double t = target[i];
                if (t > 0)
                {
                    logProb += t * Math.Log(p);
                }
            }
            return logProb + lengthPenalty;
        }
        else
        {
            // Cosine similarity converted to log probability for embeddings
            double dotProduct = 0.0;
            double predNorm = 0.0;
            double targetNorm = 0.0;

            for (int i = 0; i < minLen; i++)
            {
                dotProduct += pred[i] * target[i];
                predNorm += pred[i] * pred[i];
                targetNorm += target[i] * target[i];
            }

            predNorm = Math.Sqrt(predNorm);
            targetNorm = Math.Sqrt(targetNorm);

            if (predNorm < 1e-10 || targetNorm < 1e-10)
            {
                return -10.0 + lengthPenalty;
            }

            double cosineSim = dotProduct / (predNorm * targetNorm);
            // Convert cosine similarity [-1, 1] to log probability
            // cosineSim = 1 -> log(1) = 0, cosineSim = -1 -> log(~0) = -inf
            double prob = (cosineSim + 1.0) / 2.0; // Map to [0, 1]
            return Math.Log(Math.Max(prob, 1e-10)) + lengthPenalty;
        }
    }

    /// <summary>
    /// Computes log probability for IEnumerable outputs.
    /// </summary>
    private double ComputeEnumerableLogProbability(object[] pred, object[] target)
    {
        // Try to convert to numeric arrays
        try
        {
            var predDouble = pred.Select(p => Convert.ToDouble(p)).ToArray();
            var targetDouble = target.Select(t => Convert.ToDouble(t)).ToArray();

            // Use the array method
            return ComputeArrayLogProbability(predDouble, targetDouble);
        }
        catch (Exception ex) when (ex is InvalidCastException or FormatException or OverflowException)
        {
            // Fall back to sequence matching
            int matches = 0;
            int minLen = Math.Min(pred.Length, target.Length);
            for (int i = 0; i < minLen; i++)
            {
                if (pred[i]?.Equals(target[i]) == true)
                {
                    matches++;
                }
            }

            double matchRatio = minLen > 0 ? (double)matches / minLen : 0.0;
            double lengthPenalty = Math.Abs(pred.Length - target.Length) * -0.1;
            return Math.Log(Math.Max(matchRatio, 1e-10)) + lengthPenalty;
        }
    }

    /// <summary>
    /// Computes log probability for scalar numeric outputs.
    /// </summary>
    private static double ComputeScalarLogProbability(double pred, double target)
    {
        // Use negative squared error converted to log probability
        // Perfect match -> 0, worse match -> more negative
        double squaredError = (pred - target) * (pred - target);

        // Convert to probability using Gaussian-like function
        // P = exp(-squaredError / (2 * sigma^2))
        // log(P) = -squaredError / (2 * sigma^2)
        // Using sigma = 1.0 as default scale
        const double sigma = 1.0;
        return -squaredError / (2.0 * sigma * sigma);
    }

    /// <summary>
    /// Computes log probability for string outputs using character-level matching.
    /// </summary>
    private static double ComputeStringLogProbability(string pred, string target)
    {
        if (string.IsNullOrEmpty(pred) || string.IsNullOrEmpty(target))
        {
            return pred == target ? 0.0 : double.NegativeInfinity;
        }

        // Compute character-level matching score
        int minLen = Math.Min(pred.Length, target.Length);
        int maxLen = Math.Max(pred.Length, target.Length);
        int matches = 0;

        for (int i = 0; i < minLen; i++)
        {
            if (pred[i] == target[i])
            {
                matches++;
            }
        }

        // Match ratio penalized by length difference
        double matchRatio = (double)matches / maxLen;
        return Math.Log(Math.Max(matchRatio, 1e-10));
    }

    /// <summary>
    /// Checks if a type is a numeric type.
    /// </summary>
    private static bool IsNumericType(Type type)
    {
        return type == typeof(byte) || type == typeof(sbyte) ||
               type == typeof(short) || type == typeof(ushort) ||
               type == typeof(int) || type == typeof(uint) ||
               type == typeof(long) || type == typeof(ulong) ||
               type == typeof(float) || type == typeof(double) ||
               type == typeof(decimal);
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
    protected static double Sigmoid(double x) =>
        x >= 0
            ? 1.0 / (1.0 + Math.Exp(-x))
            : Math.Exp(x) / (1.0 + Math.Exp(x));

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
