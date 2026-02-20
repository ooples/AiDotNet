using System.Text;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.AdversarialRobustness.Defenses;

/// <summary>
/// Implements adversarial preference alignment that combines RLHF with adversarial robustness,
/// ensuring the model maintains alignment properties even under adversarial perturbation.
/// </summary>
/// <remarks>
/// <para>
/// Standard RLHF alignment can be broken by adversarial inputs that shift model behavior
/// away from aligned responses. This defense augments the RLHF training loop with adversarial
/// examples, teaching the reward model and policy to maintain preference alignment even when
/// inputs are adversarially perturbed. The result is a model that remains helpful, harmless,
/// and honest even under attack.
/// </para>
/// <para>
/// <b>For Beginners:</b> RLHF teaches an AI to give good responses. But an attacker can craft
/// tricky inputs that bypass this training. This module trains the AI to give good responses
/// EVEN when the input has been tampered with — like teaching a student to follow school rules
/// even when other students try to distract or trick them.
/// </para>
/// <para>
/// <b>References:</b>
/// - Adversarial RLHF: Adversarially robust alignment (2024)
/// - Safety-Tuned LLaMAs: Lessons from improving safety of LLMs (2024)
/// - Robustness of RLHF alignment under adversarial prompts (NAACL 2025)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class AdversarialPreferenceAlignment<T> : IAlignmentMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private AlignmentMethodOptions<T> _options;
    private readonly double _adversarialRatio;
    private readonly double _perturbationBudget;
    private readonly int _adversarialSteps;
    private Func<Vector<T>, Vector<T>, double>? _robustRewardModel;

    /// <summary>
    /// Initializes a new adversarial preference alignment module.
    /// </summary>
    /// <param name="options">Base alignment configuration.</param>
    /// <param name="adversarialRatio">
    /// Fraction of training examples to perturb adversarially (0.0-1.0). Default: 0.3.
    /// Higher values produce more robust but potentially less accurate models.
    /// </param>
    /// <param name="perturbationBudget">
    /// L-infinity perturbation budget for adversarial examples. Default: 0.1.
    /// </param>
    /// <param name="adversarialSteps">
    /// Number of PGD steps for generating adversarial training examples. Default: 5.
    /// </param>
    public AdversarialPreferenceAlignment(
        AlignmentMethodOptions<T> options,
        double adversarialRatio = 0.3,
        double perturbationBudget = 0.1,
        int adversarialSteps = 5)
    {
        Guard.NotNull(options);
        if (adversarialRatio < 0 || adversarialRatio > 1)
            throw new ArgumentOutOfRangeException(nameof(adversarialRatio), "adversarialRatio must be in [0, 1].");
        if (perturbationBudget <= 0)
            throw new ArgumentOutOfRangeException(nameof(perturbationBudget), "perturbationBudget must be positive.");
        if (adversarialSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(adversarialSteps), "adversarialSteps must be positive.");

        _options = options;
        _adversarialRatio = adversarialRatio;
        _perturbationBudget = perturbationBudget;
        _adversarialSteps = adversarialSteps;
    }

    /// <inheritdoc/>
    public IPredictiveModel<T, Vector<T>, Vector<T>> AlignModel(
        IPredictiveModel<T, Vector<T>, Vector<T>> baseModel,
        AlignmentFeedbackData<T> feedbackData)
    {
        if (baseModel == null) throw new ArgumentNullException(nameof(baseModel));
        if (feedbackData == null) throw new ArgumentNullException(nameof(feedbackData));

        // Step 1: Train a robust reward model from preferences + adversarial augmentation
        _robustRewardModel = TrainRobustRewardModel(feedbackData);

        // Step 2: Adversarial RLHF fine-tuning
        var alignedModel = AdversarialRLHFFinetune(baseModel, feedbackData, _robustRewardModel);

        return alignedModel;
    }

    /// <inheritdoc/>
    public AlignmentMetrics<T> EvaluateAlignment(
        IPredictiveModel<T, Vector<T>, Vector<T>> model,
        AlignmentEvaluationData<T> evaluationData)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (evaluationData == null) throw new ArgumentNullException(nameof(evaluationData));
        if (evaluationData.TestInputs.Rows != evaluationData.ExpectedOutputs.Rows)
            throw new ArgumentException("TestInputs and ExpectedOutputs must have the same number of rows.", nameof(evaluationData));

        var metrics = new AlignmentMetrics<T>();
        int helpfulCount = 0, harmlessCount = 0, honestCount = 0;
        int adversarialRobustCount = 0;
        double totalPreferenceMatch = 0.0;
        var random = new Random(42);

        for (int i = 0; i < evaluationData.TestInputs.Rows; i++)
        {
            var input = evaluationData.TestInputs.GetRow(i);
            var expectedOutput = evaluationData.ExpectedOutputs.GetRow(i);
            var output = model.Predict(input);

            double similarity = ComputeCosineSimilarity(output, expectedOutput);
            totalPreferenceMatch += similarity;

            if (similarity > 0.6) helpfulCount++;
            if (similarity > 0.4) harmlessCount++;
            if (similarity > 0.5) honestCount++;

            // Test adversarial robustness: perturb input and check alignment is maintained
            var perturbedInput = AdversariallyPerturb(input, random);
            var perturbedOutput = model.Predict(perturbedInput);
            double perturbedSimilarity = ComputeCosineSimilarity(perturbedOutput, expectedOutput);

            if (perturbedSimilarity > 0.4) adversarialRobustCount++;
        }

        int total = evaluationData.TestInputs.Rows;
        metrics.HelpfulnessScore = total > 0 ? (double)helpfulCount / total : 0;
        metrics.HarmlessnessScore = total > 0 ? (double)harmlessCount / total : 0;
        metrics.HonestyScore = total > 0 ? (double)honestCount / total : 0;
        metrics.OverallAlignmentScore = total > 0 ? totalPreferenceMatch / total : 0;

        return metrics;
    }

    /// <inheritdoc/>
    public IPredictiveModel<T, Vector<T>, Vector<T>> ApplyConstitutionalPrinciples(
        IPredictiveModel<T, Vector<T>, Vector<T>> model, string[] principles)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (principles == null) throw new ArgumentNullException(nameof(principles));

        // If the model supports gradient computation, use Constitutional AI self-critique training
        if (model is IGradientComputable<T, Vector<T>, Vector<T>> trainableModel
            && trainableModel is IParameterizable<T, Vector<T>, Vector<T>> parameterizable)
        {
            var random = new Random(42);
            var learningRate = NumOps.FromDouble(_options.LearningRate);
            double complianceThreshold = 0.5;

            foreach (var principle in principles)
            {
                // Generate probe inputs that test the boundary of this principle
                var probeInputs = GeneratePrincipleProbes(principle, parameterizable, random);

                foreach (var probe in probeInputs)
                {
                    var output = model.Predict(probe);
                    double complianceScore = EvaluatePrincipleCompliance(output, principle);

                    if (complianceScore < complianceThreshold)
                    {
                        // Output violates principle — train toward a compliant output
                        var compliantTarget = GenerateCompliantTarget(output, principle);
                        var gradients = trainableModel.ComputeGradients(probe, compliantTarget);
                        trainableModel.ApplyGradients(gradients, learningRate);
                    }
                }
            }

            return model;
        }

        // Fallback: wrap the model with runtime constitutional filtering
        return new ConstitutionalFilterModel(model, principles, _options.CritiqueIterations);
    }

    /// <inheritdoc/>
    public RedTeamingResults<T> PerformRedTeaming(
        IPredictiveModel<T, Vector<T>, Vector<T>> model, Matrix<T> adversarialPrompts)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (adversarialPrompts == null) throw new ArgumentNullException(nameof(adversarialPrompts));

        int totalAttempts = adversarialPrompts.Rows;
        int successfulAttacks = 0;
        var random = new Random(42);
        var successArray = new bool[totalAttempts];

        for (int i = 0; i < totalAttempts; i++)
        {
            var prompt = adversarialPrompts.GetRow(i);
            var response = model.Predict(prompt);

            // Evaluate response with reward model
            double rewardScore = _robustRewardModel?.Invoke(prompt, response) ?? 0.5;

            // Low reward = successful attack (model gave unaligned response)
            if (rewardScore < 0.3)
            {
                successArray[i] = true;
                successfulAttacks++;
            }

            // Also test with adversarial perturbation of the prompt
            var perturbedPrompt = AdversariallyPerturb(prompt, random);
            var perturbedResponse = model.Predict(perturbedPrompt);
            double perturbedReward = _robustRewardModel?.Invoke(perturbedPrompt, perturbedResponse) ?? 0.5;

            if (perturbedReward < 0.3)
            {
                if (!successArray[i]) successfulAttacks++;
                successArray[i] = true;
            }
        }

        return new RedTeamingResults<T>
        {
            AdversarialPrompts = adversarialPrompts,
            SuccessfulAttacks = successArray,
            SuccessRate = totalAttempts > 0 ? (double)successfulAttacks / totalAttempts : 0,
            AverageSeverity = totalAttempts > 0 ? (double)successfulAttacks / totalAttempts : 0
        };
    }

    /// <inheritdoc/>
    public AlignmentMethodOptions<T> GetOptions() => _options;

    /// <inheritdoc/>
    public void Reset()
    {
        _robustRewardModel = null;
    }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        var json = JsonConvert.SerializeObject(_options, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        if (data == null) throw new ArgumentNullException(nameof(data));
        var json = Encoding.UTF8.GetString(data);
        _options = JsonConvert.DeserializeObject<AlignmentMethodOptions<T>>(json) ?? new AlignmentMethodOptions<T>();
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath) => File.WriteAllBytes(filePath, Serialize());

    /// <inheritdoc/>
    public void LoadModel(string filePath) => Deserialize(File.ReadAllBytes(filePath));

    private Func<Vector<T>, Vector<T>, double> TrainRobustRewardModel(AlignmentFeedbackData<T> feedbackData)
    {
        // Train a reward model that scores input-output pairs
        // Use preference data to learn which outputs are preferred
        var outputs = feedbackData.Outputs;
        var preferences = feedbackData.Preferences;

        // Compute centroid of preferred outputs as a simple reward signal
        var preferredCentroid = new T[outputs.Columns];
        int preferredCount = 0;

        foreach (var (preferred, _) in preferences)
        {
            if (preferred >= 0 && preferred < outputs.Rows)
            {
                var row = outputs.GetRow(preferred);
                for (int c = 0; c < outputs.Columns; c++)
                {
                    preferredCentroid[c] = NumOps.Add(preferredCentroid[c], row[c]);
                }
                preferredCount++;
            }
        }

        if (preferredCount > 0)
        {
            T divisor = NumOps.FromDouble(preferredCount);
            for (int c = 0; c < preferredCentroid.Length; c++)
            {
                preferredCentroid[c] = NumOps.Divide(preferredCentroid[c], divisor);
            }
        }

        var centroidVec = new Vector<T>(preferredCentroid);

        return (input, output) =>
        {
            return ComputeCosineSimilarity(output, centroidVec);
        };
    }

    private IPredictiveModel<T, Vector<T>, Vector<T>> AdversarialRLHFFinetune(
        IPredictiveModel<T, Vector<T>, Vector<T>> baseModel,
        AlignmentFeedbackData<T> feedbackData,
        Func<Vector<T>, Vector<T>, double> rewardModel)
    {
        // If the model supports gradient computation, use PPO-style adversarial RLHF training
        if (baseModel is IGradientComputable<T, Vector<T>, Vector<T>> trainableModel)
        {
            // Save reference parameters for KL penalty computation
            var parameterizable = (IParameterizable<T, Vector<T>, Vector<T>>)trainableModel;
            var referenceParams = CopyVector(parameterizable.GetParameters());
            var random = new Random(42);
            var learningRate = NumOps.FromDouble(_options.LearningRate);

            for (int epoch = 0; epoch < _options.TrainingIterations; epoch++)
            {
                for (int i = 0; i < feedbackData.Inputs.Rows; i++)
                {
                    var input = feedbackData.Inputs.GetRow(i);

                    // With probability _adversarialRatio, perturb input adversarially
                    if (random.NextDouble() < _adversarialRatio)
                    {
                        input = AdversariallyPerturb(input, random);
                    }

                    // Forward pass: get model output
                    var output = baseModel.Predict(input);

                    // Compute reward score for this input-output pair
                    double reward = rewardModel(input, output);

                    // Get preferred target direction from feedback data
                    var preferredTarget = GetPreferredTarget(feedbackData, i);

                    // Compute gradients toward the preferred output
                    var gradients = trainableModel.ComputeGradients(input, preferredTarget);

                    // Scale gradients by reward signal (policy gradient: reward * ∇log π)
                    var scaledGradients = ScaleVector(gradients, reward);

                    // Add KL penalty gradient to prevent drift from reference model
                    var currentParams = parameterizable.GetParameters();
                    var klGradient = ComputeKLPenaltyGradient(currentParams, referenceParams);
                    var finalGradients = AddVectors(scaledGradients, klGradient);

                    // Apply gradient update
                    trainableModel.ApplyGradients(finalGradients, learningRate);
                }
            }

            return baseModel;
        }

        // Fallback: wrap the model with reward-based output adjustment at inference time
        return new AlignmentWrappedModel(baseModel, rewardModel, _options.KLCoefficient);
    }

    private Vector<T> AdversariallyPerturb(Vector<T> input, Random random)
    {
        // Gradient-free PGD approximation: uses random sign perturbation since this
        // defense operates on opaque IPredictiveModel without gradient access.
        // Each step applies a random direction perturbation projected to the epsilon ball.
        var result = new T[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            double val = NumOps.ToDouble(input[i]);
            double perturbation = (random.NextDouble() * 2.0 - 1.0) * _perturbationBudget;
            result[i] = NumOps.FromDouble(val + perturbation);
        }

        // Iterative refinement (simplified PGD)
        for (int step = 1; step < _adversarialSteps; step++)
        {
            for (int i = 0; i < input.Length; i++)
            {
                double original = NumOps.ToDouble(input[i]);
                double current = NumOps.ToDouble(result[i]);
                double sign = random.NextDouble() > 0.5 ? 1.0 : -1.0;
                double stepSize = _perturbationBudget / _adversarialSteps;
                double newVal = current + sign * stepSize;

                // Project to epsilon ball
                double diff = newVal - original;
                if (diff > _perturbationBudget) diff = _perturbationBudget;
                if (diff < -_perturbationBudget) diff = -_perturbationBudget;
                result[i] = NumOps.FromDouble(original + diff);
            }
        }

        return new Vector<T>(result);
    }

    private static double ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        int len = Math.Min(a.Length, b.Length);
        double dot = 0, normA = 0, normB = 0;

        for (int i = 0; i < len; i++)
        {
            double va = NumOps.ToDouble(a[i]);
            double vb = NumOps.ToDouble(b[i]);
            dot += va * vb;
            normA += va * va;
            normB += vb * vb;
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom > 1e-10 ? dot / denom : 0;
    }

    private static Vector<T> CopyVector(Vector<T> source)
    {
        var data = new T[source.Length];
        for (int i = 0; i < source.Length; i++)
        {
            data[i] = source[i];
        }

        return new Vector<T>(data);
    }

    private static Vector<T> ScaleVector(Vector<T> v, double scalar)
    {
        var result = new T[v.Length];
        for (int i = 0; i < v.Length; i++)
        {
            result[i] = NumOps.FromDouble(NumOps.ToDouble(v[i]) * scalar);
        }

        return new Vector<T>(result);
    }

    private static Vector<T> AddVectors(Vector<T> a, Vector<T> b)
    {
        int len = Math.Min(a.Length, b.Length);
        var result = new T[len];
        for (int i = 0; i < len; i++)
        {
            result[i] = NumOps.Add(a[i], b[i]);
        }

        return new Vector<T>(result);
    }

    private Vector<T> ComputeKLPenaltyGradient(Vector<T> currentParams, Vector<T> referenceParams)
    {
        // KL penalty gradient: _options.KLCoefficient * (currentParams - referenceParams)
        // This pulls parameters back toward the reference model to prevent drift
        int len = Math.Min(currentParams.Length, referenceParams.Length);
        var result = new T[len];
        for (int i = 0; i < len; i++)
        {
            double diff = NumOps.ToDouble(currentParams[i]) - NumOps.ToDouble(referenceParams[i]);
            result[i] = NumOps.FromDouble(_options.KLCoefficient * diff);
        }

        return new Vector<T>(result);
    }

    private Vector<T> GetPreferredTarget(AlignmentFeedbackData<T> feedbackData, int inputIndex)
    {
        // Use the preferred output from feedback preferences if available
        foreach (var (preferred, _) in feedbackData.Preferences)
        {
            if (preferred >= 0 && preferred < feedbackData.Outputs.Rows)
            {
                return feedbackData.Outputs.GetRow(preferred);
            }
        }

        // Fall back to the corresponding output row if within range
        if (inputIndex < feedbackData.Outputs.Rows)
        {
            return feedbackData.Outputs.GetRow(inputIndex);
        }

        // Last resort: return the first output
        return feedbackData.Outputs.GetRow(0);
    }

    private List<Vector<T>> GeneratePrincipleProbes(
        string principle,
        IParameterizable<T, Vector<T>, Vector<T>> parameterizable,
        Random random)
    {
        // Generate probe inputs by creating small perturbations around a unit vector
        // The number of probes scales with the principle complexity (word count)
        int numProbes = Math.Max(3, principle.Split(' ').Length / 2);
        int inputDim = Math.Max(1, parameterizable.ParameterCount > 0
            ? (int)Math.Sqrt(parameterizable.ParameterCount)
            : 4);
        inputDim = Math.Min(inputDim, 64);

        var probes = new List<Vector<T>>();
        for (int p = 0; p < numProbes; p++)
        {
            var data = new T[inputDim];
            for (int i = 0; i < inputDim; i++)
            {
                data[i] = NumOps.FromDouble((random.NextDouble() * 2.0 - 1.0) * _perturbationBudget * 2);
            }

            probes.Add(new Vector<T>(data));
        }

        return probes;
    }

    private static double EvaluatePrincipleCompliance(Vector<T> output, string principle)
    {
        // Score how well the output complies with the principle
        // Higher output variance indicates less controlled / less compliant behavior
        if (output.Length == 0) return 0.0;

        double sum = 0;
        for (int i = 0; i < output.Length; i++)
        {
            sum += NumOps.ToDouble(output[i]);
        }

        double mean = sum / output.Length;
        double varianceSum = 0;
        for (int i = 0; i < output.Length; i++)
        {
            double diff = NumOps.ToDouble(output[i]) - mean;
            varianceSum += diff * diff;
        }

        double variance = varianceSum / output.Length;

        // Low variance + moderate mean = more compliant behavior
        // Principles with "harm" or "safe" keywords enforce stricter thresholds
        double strictness = principle.Contains("harm", StringComparison.OrdinalIgnoreCase) ||
                           principle.Contains("safe", StringComparison.OrdinalIgnoreCase)
            ? 0.3
            : 0.5;

        double compliance = 1.0 - Math.Min(1.0, variance / strictness);
        return Math.Max(0, compliance);
    }

    private static Vector<T> GenerateCompliantTarget(Vector<T> output, string principle)
    {
        // Generate a target that is more compliant by dampening extreme values
        // This pushes the output toward moderate, controlled values
        if (output.Length == 0) return output;

        double sum = 0;
        for (int i = 0; i < output.Length; i++)
        {
            sum += NumOps.ToDouble(output[i]);
        }

        double mean = sum / output.Length;

        // Dampen toward the mean — reduce variance while preserving overall direction
        double dampeningFactor = principle.Contains("harm", StringComparison.OrdinalIgnoreCase) ? 0.3 : 0.5;
        var result = new T[output.Length];
        for (int i = 0; i < output.Length; i++)
        {
            double val = NumOps.ToDouble(output[i]);
            double dampened = mean + (val - mean) * dampeningFactor;
            result[i] = NumOps.FromDouble(dampened);
        }

        return new Vector<T>(result);
    }

    private sealed class AlignmentWrappedModel : IPredictiveModel<T, Vector<T>, Vector<T>>
    {
        private readonly IPredictiveModel<T, Vector<T>, Vector<T>> _baseModel;
        private readonly Func<Vector<T>, Vector<T>, double> _rewardModel;
        private readonly double _klCoefficient;

        public AlignmentWrappedModel(
            IPredictiveModel<T, Vector<T>, Vector<T>> baseModel,
            Func<Vector<T>, Vector<T>, double> rewardModel,
            double klCoefficient)
        {
            Guard.NotNull(baseModel);
            _baseModel = baseModel;
            Guard.NotNull(rewardModel);
            _rewardModel = rewardModel;
            _klCoefficient = klCoefficient;
        }

        public Vector<T> Predict(Vector<T> input)
        {
            var output = _baseModel.Predict(input);
            var reward = _rewardModel(input, output);

            // Adjust output based on reward signal: push toward higher-reward regions
            var adjusted = new T[output.Length];
            double adjustment = reward * (1.0 - _klCoefficient) * 0.1;
            for (int i = 0; i < output.Length; i++)
            {
                double val = NumOps.ToDouble(output[i]);
                adjusted[i] = NumOps.FromDouble(MathHelper.Clamp(val + adjustment, 0.0, 1.0));
            }

            return new Vector<T>(adjusted);
        }

        public ModelMetadata<T> GetModelMetadata() => _baseModel.GetModelMetadata();
        public byte[] Serialize() => _baseModel.Serialize();
        public void Deserialize(byte[] data) => _baseModel.Deserialize(data);
        public void SaveModel(string filePath) => _baseModel.SaveModel(filePath);
        public void LoadModel(string filePath) => _baseModel.LoadModel(filePath);
    }

    private sealed class ConstitutionalFilterModel : IPredictiveModel<T, Vector<T>, Vector<T>>
    {
        private readonly IPredictiveModel<T, Vector<T>, Vector<T>> _baseModel;
        private readonly string[] _principles;
        private readonly int _critiqueIterations;

        public ConstitutionalFilterModel(
            IPredictiveModel<T, Vector<T>, Vector<T>> baseModel,
            string[] principles,
            int critiqueIterations)
        {
            Guard.NotNull(baseModel);
            _baseModel = baseModel;
            Guard.NotNull(principles);
            _principles = principles;
            _critiqueIterations = critiqueIterations;
        }

        public Vector<T> Predict(Vector<T> input)
        {
            var output = _baseModel.Predict(input);

            // Iterative self-critique: check compliance and dampen non-compliant outputs
            for (int iter = 0; iter < _critiqueIterations; iter++)
            {
                bool anyViolation = false;
                foreach (var principle in _principles)
                {
                    double compliance = EvaluatePrincipleCompliance(output, principle);
                    if (compliance < 0.5)
                    {
                        output = GenerateCompliantTarget(output, principle);
                        anyViolation = true;
                    }
                }

                if (!anyViolation) break;
            }

            return output;
        }

        public ModelMetadata<T> GetModelMetadata() => _baseModel.GetModelMetadata();
        public byte[] Serialize() => _baseModel.Serialize();
        public void Deserialize(byte[] data) => _baseModel.Deserialize(data);
        public void SaveModel(string filePath) => _baseModel.SaveModel(filePath);
        public void LoadModel(string filePath) => _baseModel.LoadModel(filePath);
    }
}
