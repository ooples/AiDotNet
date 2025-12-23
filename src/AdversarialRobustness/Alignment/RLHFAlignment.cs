using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Newtonsoft.Json;

namespace AiDotNet.AdversarialRobustness.Alignment;

/// <summary>
/// Implements Reinforcement Learning from Human Feedback (RLHF) for AI alignment.
/// </summary>
/// <remarks>
/// <para>
/// RLHF trains models to align with human preferences by learning a reward model
/// from human feedback and using it to fine-tune the model via reinforcement learning.
/// </para>
/// <para><b>For Beginners:</b> RLHF is like having a human teacher grade the AI's responses
/// and using those grades to improve the AI. The AI learns what humans prefer and adjusts
/// its behavior accordingly. This is how models like ChatGPT learn to be helpful and follow
/// instructions.</para>
/// <para>
/// Original approaches: "Learning to summarize from human feedback" (OpenAI, 2020),
/// "Training language models to follow instructions with human feedback" (InstructGPT, 2022)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class RLHFAlignment<T> : IAlignmentMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private AlignmentMethodOptions<T> options;
    private Func<Vector<T>, Vector<T>, double>? rewardModel;

    /// <summary>
    /// Initializes a new instance of RLHF alignment.
    /// </summary>
    /// <param name="options">The alignment configuration options.</param>
    public RLHFAlignment(AlignmentMethodOptions<T> options)
    {
        this.options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc/>
    public IPredictiveModel<T, Vector<T>, Vector<T>> AlignModel(IPredictiveModel<T, Vector<T>, Vector<T>> baseModel, AlignmentFeedbackData<T> feedbackData)
    {
        // Step 1: Train a reward model from human preferences
        rewardModel = TrainRewardModel(feedbackData);

        // Step 2: Fine-tune the policy model using the reward model
        var alignedModel = FinetuneWithRL(baseModel, feedbackData, rewardModel);

        return alignedModel;
    }

    /// <inheritdoc/>
    public AlignmentMetrics<T> EvaluateAlignment(IPredictiveModel<T, Vector<T>, Vector<T>> model, AlignmentEvaluationData<T> evaluationData)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (evaluationData == null)
        {
            throw new ArgumentNullException(nameof(evaluationData));
        }

        var metrics = new AlignmentMetrics<T>();

        int helpfulCount = 0;
        int harmlessCount = 0;
        int honestCount = 0;
        double totalPreferenceMatch = 0.0;

        for (int i = 0; i < evaluationData.TestInputs.Rows; i++)
        {
            var input = evaluationData.TestInputs.GetRow(i);
            var expected = evaluationData.ExpectedOutputs.GetRow(i);
            var output = model.Predict(input);

            // Evaluate helpfulness (simplified)
            if (IsHelpful(output, expected))
            {
                helpfulCount++;
            }

            // Evaluate harmlessness (simplified)
            if (IsHarmless(output))
            {
                harmlessCount++;
            }

            // Evaluate honesty (simplified)
            if (IsHonest(output, input))
            {
                honestCount++;
            }

            // Preference matching
            if (i < evaluationData.ReferenceScores.Length)
            {
                var predictedScore = rewardModel?.Invoke(input, output) ?? 0.5;
                var referenceScore = evaluationData.ReferenceScores[i];
                totalPreferenceMatch += 1.0 - Math.Abs(predictedScore - referenceScore);
            }
        }

        int total = evaluationData.TestInputs.Rows;
        metrics.HelpfulnessScore = (double)helpfulCount / total;
        metrics.HarmlessnessScore = (double)harmlessCount / total;
        metrics.HonestyScore = (double)honestCount / total;
        metrics.PreferenceMatchRate = totalPreferenceMatch / total;
        metrics.OverallAlignmentScore = (metrics.HelpfulnessScore + metrics.HarmlessnessScore + metrics.HonestyScore) / 3.0;

        return metrics;
    }

    /// <inheritdoc/>
    public IPredictiveModel<T, Vector<T>, Vector<T>> ApplyConstitutionalPrinciples(IPredictiveModel<T, Vector<T>, Vector<T>> model, string[] principles)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (principles == null)
        {
            throw new ArgumentNullException(nameof(principles));
        }

        // Wrap the model with constitutional AI principles
        return new ConstitutionalPredictiveModel(model, this, principles);
    }

    /// <inheritdoc/>
    public RedTeamingResults<T> PerformRedTeaming(IPredictiveModel<T, Vector<T>, Vector<T>> model, Matrix<T> adversarialPrompts)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (adversarialPrompts == null)
        {
            throw new ArgumentNullException(nameof(adversarialPrompts));
        }

        if (adversarialPrompts.Rows == 0)
        {
            return new RedTeamingResults<T>
            {
                AdversarialPrompts = adversarialPrompts,
                ModelResponses = Matrix<T>.Empty(),
                SuccessfulAttacks = Array.Empty<bool>(),
                SeverityScores = Array.Empty<double>(),
                VulnerabilityTypes = Array.Empty<string>(),
                Vulnerabilities = new List<VulnerabilityReport>(),
                SuccessRate = 0.0,
                AverageSeverity = 0.0
            };
        }

        var firstResponse = model.Predict(adversarialPrompts.GetRow(0));
        var responseLength = firstResponse.Length;
        var modelResponses = new Matrix<T>(adversarialPrompts.Rows, responseLength);
        modelResponses.SetRow(0, firstResponse);

        var results = new RedTeamingResults<T>
        {
            AdversarialPrompts = adversarialPrompts,
            ModelResponses = modelResponses,
            SuccessfulAttacks = new bool[adversarialPrompts.Rows],
            SeverityScores = new double[adversarialPrompts.Rows],
            VulnerabilityTypes = new string[adversarialPrompts.Rows],
            Vulnerabilities = new List<VulnerabilityReport>()
        };

        int successfulAttacks = 0;
        double totalSeverity = 0.0;

        for (int i = 0; i < adversarialPrompts.Rows; i++)
        {
            Vector<T> response;
            if (i == 0)
            {
                response = firstResponse;
            }
            else
            {
                response = model.Predict(adversarialPrompts.GetRow(i));
                if (response.Length != responseLength)
                {
                    throw new InvalidOperationException("Model output length must be consistent across prompts.");
                }

                modelResponses.SetRow(i, response);
            }

            // Analyze response for misalignment
            var (isVulnerable, severity, vulnType) = AnalyzeResponseForVulnerability(response);

            results.SuccessfulAttacks[i] = isVulnerable;
            results.SeverityScores[i] = severity;
            results.VulnerabilityTypes[i] = vulnType;

            if (isVulnerable)
            {
                successfulAttacks++;
                totalSeverity += severity;

                results.Vulnerabilities.Add(new VulnerabilityReport
                {
                    Type = vulnType,
                    Severity = severity,
                    Description = $"Model showed misaligned behavior of type: {vulnType}",
                    ExamplePrompt = ConvertToString(adversarialPrompts.GetRow(i)),
                    ProblematicResponse = ConvertToString(response),
                    Recommendations = new[]
                    {
                        "Add safety filters",
                        "Improve RLHF training data",
                        "Strengthen constitutional principles"
                    }
                });
            }
        }

        results.SuccessRate = (double)successfulAttacks / adversarialPrompts.Rows;
        results.AverageSeverity = successfulAttacks > 0 ? totalSeverity / successfulAttacks : 0.0;

        return results;
    }

    /// <inheritdoc/>
    public AlignmentMethodOptions<T> GetOptions() => options;

    /// <inheritdoc/>
    public void Reset() { }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        var json = JsonConvert.SerializeObject(options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var json = Encoding.UTF8.GetString(data);
        options = JsonConvert.DeserializeObject<AlignmentMethodOptions<T>>(json) ?? new AlignmentMethodOptions<T>();

        // Reset reward model - it cannot be serialized and must be retrained
        // by calling AlignModel with new feedback data
        rewardModel = null;
    }

    /// <summary>
    /// Gets whether the reward model has been trained.
    /// </summary>
    /// <remarks>
    /// <para>
    /// After deserialization, the reward model will be null because functions cannot
    /// be serialized. Call <see cref="AlignModel"/> with feedback data to retrain it.
    /// </para>
    /// </remarks>
    public bool IsRewardModelTrained => rewardModel != null;

    /// <inheritdoc/>
    public void SaveModel(string filePath)
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
    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        var fullPath = Path.GetFullPath(filePath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Model file not found.", fullPath);
        }

        Deserialize(File.ReadAllBytes(fullPath));
    }

    private Func<Vector<T>, Vector<T>, double> TrainRewardModel(AlignmentFeedbackData<T> feedbackData)
    {
        // Train a reward model from human preference comparisons
        // This is a simplified placeholder - real implementation would use neural networks

        return (input, output) =>
        {
            // Simple reward heuristic based on output characteristics
            double sum = 0.0;
            for (int i = 0; i < output.Length; i++)
            {
                sum += NumOps.ToDouble(output[i]);
            }
            var outputMean = output.Length > 0 ? (sum / output.Length) : 0.0;

            // Higher reward for moderate values (heuristic for "reasonable" outputs)
            var reward = 1.0 - Math.Abs(outputMean - 0.5);
            return MathHelper.Clamp(reward, 0.0, 1.0);
        };
    }

    private IPredictiveModel<T, Vector<T>, Vector<T>> FinetuneWithRL(IPredictiveModel<T, Vector<T>, Vector<T>> baseModel, AlignmentFeedbackData<T> feedbackData, Func<Vector<T>, Vector<T>, double> rewardModelFunc)
    {
        // Simplified PPO-like fine-tuning.
        // Real implementation would integrate with a RL framework.
        _ = feedbackData;
        return new RlhfFineTunedPredictiveModel(baseModel, rewardModelFunc, options.KLCoefficient);
    }

    private static string GenerateCritique(Vector<T> response, string[] principles)
    {
        // Generate critique based on constitutional principles
        // Simplified version - real implementation would use language models
        _ = response;
        return $"Response evaluated against {principles.Length} constitutional principles";
    }

    private static Vector<T> ReviseBasedOnCritique(IPredictiveModel<T, Vector<T>, Vector<T>> model, Vector<T> input, Vector<T> response, string critique)
    {
        // Revise response based on critique
        // Simplified version - real implementation would regenerate with critique context
        _ = model;
        _ = input;
        _ = critique;
        return response;
    }

    private bool IsHelpful(Vector<T> output, Vector<T> expected)
    {
        // Simplified helpfulness check
        var similarity = ComputeCosineSimilarity(output, expected);
        return similarity > 0.5;
    }

    private bool IsHarmless(Vector<T> output)
    {
        // Check for harmful patterns (simplified)
        double maxValue = double.NegativeInfinity;
        double minValue = double.PositiveInfinity;
        for (int i = 0; i < output.Length; i++)
        {
            var v = NumOps.ToDouble(output[i]);
            if (v > maxValue) maxValue = v;
            if (v < minValue) minValue = v;
        }
        return (maxValue - minValue) < 0.9; // Heuristic for extremity
    }

    private bool IsHonest(Vector<T> output, Vector<T> input)
    {
        // Simplified honesty check
        _ = output;
        _ = input;
        return true; // Placeholder
    }

    private (bool isVulnerable, double severity, string type) AnalyzeResponseForVulnerability(Vector<T> response)
    {
        // Analyze response for potential misalignment
        double sum = 0.0;
        for (int i = 0; i < response.Length; i++)
        {
            sum += NumOps.ToDouble(response[i]);
        }
        var mean = response.Length > 0 ? (sum / response.Length) : 0.0;

        double varianceSum = 0.0;
        for (int i = 0; i < response.Length; i++)
        {
            var delta = NumOps.ToDouble(response[i]) - mean;
            varianceSum += delta * delta;
        }
        var variance = response.Length > 0 ? (varianceSum / response.Length) : 0.0;

        if (variance > 0.3)
        {
            return (true, 0.7, "HighVariance");
        }

        if (mean < 0.2 || mean > 0.8)
        {
            return (true, 0.6, "ExtremeBias");
        }

        return (false, 0.0, "None");
    }

    private double ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        if (a.Length != b.Length) return 0.0;

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < a.Length; i++)
        {
            var aVal = NumOps.ToDouble(a[i]);
            var bVal = NumOps.ToDouble(b[i]);
            dotProduct += aVal * bVal;
            normA += aVal * aVal;
            normB += bVal * bVal;
        }

        return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB) + 1e-10);
    }

    private static string ConvertToString(Vector<T> data)
    {
        if (data == null || data.Length == 0)
            return string.Empty;

        var builder = new StringBuilder();
        for (int i = 0; i < data.Length; i++)
        {
            if (i > 0)
            {
                builder.Append(',');
            }

            object? value = data[i];
            builder.Append(value?.ToString() ?? string.Empty);
        }

        return builder.ToString();
    }

    private static T Clip01(T value)
    {
        return MathHelper.Clamp(value, NumOps.Zero, NumOps.One);
    }

    private sealed class RlhfFineTunedPredictiveModel : IPredictiveModel<T, Vector<T>, Vector<T>>
    {
        private readonly IPredictiveModel<T, Vector<T>, Vector<T>> _baseModel;
        private readonly Func<Vector<T>, Vector<T>, double> _rewardModel;
        private readonly double _klCoefficient;

        public RlhfFineTunedPredictiveModel(IPredictiveModel<T, Vector<T>, Vector<T>> baseModel, Func<Vector<T>, Vector<T>, double> rewardModel, double klCoefficient)
        {
            _baseModel = baseModel ?? throw new ArgumentNullException(nameof(baseModel));
            _rewardModel = rewardModel ?? throw new ArgumentNullException(nameof(rewardModel));
            _klCoefficient = klCoefficient;
        }

        public Vector<T> Predict(Vector<T> input)
        {
            var output = _baseModel.Predict(input);

            // Apply KL penalty to stay close to base model (placeholder, kept for future integration).
            _ = _klCoefficient;

            var reward = _rewardModel(input, output);
            var adjusted = new Vector<T>(output.Length);
            var adjustment = NumOps.FromDouble(reward * 0.1);

            for (int i = 0; i < output.Length; i++)
            {
                adjusted[i] = Clip01(NumOps.Add(output[i], adjustment));
            }

            return adjusted;
        }

        public ModelMetadata<T> GetModelMetadata()
        {
            return _baseModel.GetModelMetadata();
        }

        public byte[] Serialize()
        {
            return _baseModel.Serialize();
        }

        public void Deserialize(byte[] data)
        {
            _baseModel.Deserialize(data);
        }

        public void SaveModel(string filePath)
        {
            _baseModel.SaveModel(filePath);
        }

        public void LoadModel(string filePath)
        {
            _baseModel.LoadModel(filePath);
        }
    }

    private sealed class ConstitutionalPredictiveModel : IPredictiveModel<T, Vector<T>, Vector<T>>
    {
        private readonly IPredictiveModel<T, Vector<T>, Vector<T>> _inner;
        private readonly RLHFAlignment<T> _alignment;
        private readonly string[] _principles;

        public ConstitutionalPredictiveModel(IPredictiveModel<T, Vector<T>, Vector<T>> inner, RLHFAlignment<T> alignment, string[] principles)
        {
            _inner = inner ?? throw new ArgumentNullException(nameof(inner));
            _alignment = alignment ?? throw new ArgumentNullException(nameof(alignment));
            _principles = principles ?? throw new ArgumentNullException(nameof(principles));
        }

        public Vector<T> Predict(Vector<T> input)
        {
            var response = _inner.Predict(input);
            for (int i = 0; i < _alignment.options.CritiqueIterations; i++)
            {
                var critique = GenerateCritique(response, _principles);
                response = ReviseBasedOnCritique(_inner, input, response, critique);
            }

            return response;
        }

        public ModelMetadata<T> GetModelMetadata()
        {
            return _inner.GetModelMetadata();
        }

        public byte[] Serialize()
        {
            return _inner.Serialize();
        }

        public void Deserialize(byte[] data)
        {
            _inner.Deserialize(data);
        }

        public void SaveModel(string filePath)
        {
            _inner.SaveModel(filePath);
        }

        public void LoadModel(string filePath)
        {
            _inner.LoadModel(filePath);
        }
    }

}
