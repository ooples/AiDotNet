using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Safety;
using AiDotNet.Safety.Benchmarking;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
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
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.ReinforcementLearningAgent)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Training Language Models to Follow Instructions with Human Feedback", "https://arxiv.org/abs/2203.02155", Year = 2022, Authors = "Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe")]
public class RLHFAlignment<T> : IAlignmentMethod<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private static IEngine Engine => AiDotNetEngine.Current;

    private AlignmentMethodOptions<T> options;
    private Func<Vector<T>, Vector<T>, double>? rewardModel;

    /// <summary>
    /// When false (default — industry convention: honesty-on-non-comparable),
    /// <see cref="IsHonest"/> returns <c>true</c> for inputs the heuristic
    /// cannot evaluate (null / empty / length-mismatch input vs output) and
    /// emits a Trace warning so the unscored pair is observable in
    /// diagnostics. When true, the same pairs return <c>false</c> so they
    /// can't inflate <c>HonestyScore</c> for batches that never actually
    /// flow input through the honesty heuristic. Turn this on when you'd
    /// rather under-count honesty than over-count it (e.g. comparing
    /// alignment runs across tokenizers that produce mismatched-length
    /// input/output token vectors).
    /// </summary>
    public bool StrictHonestyMode { get; set; }

    /// <summary>
    /// Initializes a new instance of RLHF alignment.
    /// </summary>
    /// <param name="options">The alignment configuration options.</param>
    public RLHFAlignment(AlignmentMethodOptions<T> options)
    {
        Guard.NotNull(options);
        this.options = options;
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
        // Reset the per-batch warn-once flag so this evaluation run gets a
        // fresh chance to surface the diagnostic, but subsequent IsHonest
        // calls within the run still warn at most once.
        _hasLoggedNonComparableHonestyWarning = false;

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

            // Preference matching: map unbounded Bradley-Terry log-odds r(x, y) to [0, 1] via logistic sigmoid (r = 0 -> 0.5).
            if (i < evaluationData.ReferenceScores.Length)
            {
                double predictedScore = rewardModel is not null
                    ? 1.0 / (1.0 + Math.Exp(-rewardModel(input, output)))
                    : 0.5;
                double referenceScore = MathHelper.Clamp(evaluationData.ReferenceScores[i], 0.0, 1.0);
                totalPreferenceMatch += MathHelper.Clamp(1.0 - Math.Abs(predictedScore - referenceScore), 0.0, 1.0);
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
    public RedTeamingResults PerformRedTeaming(Func<string, string> model, SafetyPipeline<T> pipeline)
    {
        if (model == null)
        {
            throw new ArgumentNullException(nameof(model));
        }

        if (pipeline == null)
        {
            throw new ArgumentNullException(nameof(pipeline));
        }

        return SafetyRedTeamer.RedTeamModel(model, pipeline, options);
    }

    /// <inheritdoc/>
    public AlignmentMethodOptions<T> GetOptions() => options;

    /// <inheritdoc/>
    public void Reset() { }

    /// <summary>
    /// Persisted state for serialise/deserialise round-trip — bundles
    /// options with the behaviour-changing <see cref="StrictHonestyMode"/>
    /// flag so a saved + reloaded alignment object keeps the same scoring
    /// semantics. Wrapped in a private class so adding new flags later is
    /// a backward-compatible JSON extension.
    /// </summary>
    private sealed class RlhfAlignmentState
    {
        public AlignmentMethodOptions<T>? Options { get; set; }
        public bool StrictHonestyMode { get; set; }
    }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        var state = new RlhfAlignmentState
        {
            Options = options,
            StrictHonestyMode = StrictHonestyMode,
        };
        var json = JsonConvert.SerializeObject(state, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        if (data == null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        var json = Encoding.UTF8.GetString(data);
        // Backward-compat: try the new wrapped state first; fall back to the
        // raw-options shape so checkpoints written before the StrictHonestyMode
        // flag was added still load. JsonConvert returns null when the
        // payload doesn't match the wrapper's properties — in that case
        // Options stays null and we re-parse as bare AlignmentMethodOptions.
        var state = JsonConvert.DeserializeObject<RlhfAlignmentState>(json);
        if (state?.Options is not null)
        {
            options = state.Options;
            StrictHonestyMode = state.StrictHonestyMode;
        }
        else
        {
            options = JsonConvert.DeserializeObject<AlignmentMethodOptions<T>>(json) ?? new AlignmentMethodOptions<T>();
        }

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

        Helpers.ModelPersistenceGuard.EnforceBeforeSave();

        var fullPath = Path.GetFullPath(filePath);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            File.WriteAllBytes(fullPath, Serialize());
        }
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        }

        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();

        var fullPath = Path.GetFullPath(filePath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException("Model file not found.", fullPath);
        }

        using (Helpers.ModelPersistenceGuard.InternalOperation())
        {
            Deserialize(File.ReadAllBytes(fullPath));
        }
    }

    /// <summary>
    /// Fits a Bradley-Terry reward model to the preference pairs (Christiano et al. 2017; Ouyang et al. 2022, eq. 1).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The reward is linear in the response, r(x, y) = w . y, and w maximizes the preference log-likelihood
    /// sum log sigmoid(r(y_preferred) - r(y_rejected)) by <see cref="AlignmentMethodOptions{T}.TrainingIterations"/>
    /// full-batch gradient-ascent steps of size <see cref="AlignmentMethodOptions{T}.RewardModelLearningRate"/>. Without
    /// preference pairs the reward is zero, so alignment leaves the model's distribution unchanged. The reward scores
    /// responses of the width of <c>AlignmentFeedbackData.Outputs</c> only: a response of another width is an error,
    /// not truncated or zero-padded.
    /// </para>
    /// </remarks>
    private Func<Vector<T>, Vector<T>, double> TrainRewardModel(AlignmentFeedbackData<T> feedbackData)
    {
        Guard.NotNull(feedbackData);
        var pairs = feedbackData.Preferences ?? Array.Empty<(int preferred, int notPreferred)>();
        int dimension = feedbackData.Outputs.Columns;
        if (pairs.Length == 0 || dimension == 0)
        {
            return (input, output) => 0.0;
        }

        if (double.IsNaN(options.RewardModelLearningRate) || double.IsInfinity(options.RewardModelLearningRate)
            || options.RewardModelLearningRate <= 0)
        {
            throw new InvalidOperationException(
                $"AlignmentMethodOptions.RewardModelLearningRate must be positive and finite (was {options.RewardModelLearningRate}).");
        }

        if (!feedbackData.ValidatePreferences())
        {
            throw new ArgumentException("A preference pair indexes a row outside AlignmentFeedbackData.Outputs.", nameof(feedbackData));
        }

        // D = Y_preferred - Y_rejected, one row per pair.
        var outputs = Tensor<T>.FromMatrix(feedbackData.Outputs);
        var preferred = Engine.TensorIndexSelect(outputs, new Tensor<int>(pairs.Select(p => p.preferred).ToArray(), new[] { pairs.Length }), 0);
        var rejected = Engine.TensorIndexSelect(outputs, new Tensor<int>(pairs.Select(p => p.notPreferred).ToArray(), new[] { pairs.Length }), 0);
        var differences = Engine.TensorSubtract(preferred, rejected);
        var differencesT = Engine.TensorTranspose(differences);
        var weights = new Tensor<T>(new[] { dimension, 1 });
        T step = NumOps.FromDouble(options.RewardModelLearningRate / pairs.Length);

        for (int iteration = 0; iteration < options.TrainingIterations; iteration++)
        {
            // d/dw sum log sigmoid(D w) = D^T (1 - sigmoid(D w)).
            var margins = Engine.TensorMatMul(differences, weights);
            var residual = Engine.TensorAddScalar(Engine.TensorNegate(Engine.TensorSigmoid(margins)), NumOps.One);
            var gradient = Engine.TensorMatMul(differencesT, residual);
            weights = Engine.TensorAdd(weights, Engine.TensorMultiplyScalar(gradient, step));
        }

        var w = Engine.Reshape(weights, new[] { dimension });
        return (input, output) =>
        {
            if (output.Length != dimension)
            {
                throw new InvalidOperationException(
                    $"The reward model was fit on responses of width {dimension} (AlignmentFeedbackData.Outputs columns) " +
                    $"but was asked to score one of width {output.Length}: the model's outputs and the feedback responses " +
                    "must have the same width.");
            }

            T mean = Engine.TensorMean(Engine.TensorMultiply(output.AsTensor(), w));
            return NumOps.ToDouble(mean) * dimension;
        };
    }

    /// <summary>
    /// The aligned policy: the optimum of the KL-regularized RLHF objective, softmax(z + r / KLCoefficient), over the
    /// reference model's output classes (see <see cref="KlRegularizedPolicy{T}"/>).
    /// </summary>
    private IPredictiveModel<T, Vector<T>, Vector<T>> FinetuneWithRL(IPredictiveModel<T, Vector<T>, Vector<T>> baseModel, AlignmentFeedbackData<T> feedbackData, Func<Vector<T>, Vector<T>, double> rewardModelFunc)
    {
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
        var similarity = VectorHelper.CosineSimilarity(output, expected);
        return similarity > 0.5;
    }

    private bool IsHarmless(Vector<T> output)
    {
        // Check for harmful patterns (simplified): the output range is a heuristic for extremity.
        if (output.Length == 0)
        {
            return true;
        }

        var values = output.AsTensor();
        T range = NumOps.Subtract(Engine.TensorMaxValue(values), Engine.TensorMinValue(values));
        return NumOps.LessThan(range, NumOps.FromDouble(0.9));
    }

    private bool IsHonest(Vector<T> output, Vector<T> input)
    {
        // "Honesty" in the 3H-RLHF sense (Bai et al. 2022 "Training a Helpful and
        // Harmless Assistant with RLHF" §2.2) requires comparing a claim against
        // ground truth — not derivable from (output, input) alone without a fact
        // checker or knowledge base. The structural proxy here flags outputs that
        // are *clearly* not honest engagement with the input: degenerate outputs
        // (all-NaN, all-Inf, all-zero) and saturated outputs that ignore the input
        // entirely. Users who need a true honesty signal should plug a trained
        // checker into the alignment pipeline; the in-line heuristic here is
        // strictly a structural sanity gate, not an honesty metric.
        if (output is null || output.Length == 0) return false;

        var outputTensor = output.AsTensor();
        var finite = Engine.TensorWhere(Engine.TensorIsFinite(outputTensor), Engine.TensorOnesLike(outputTensor), Engine.TensorZerosLike(outputTensor));
        if (NumOps.LessThan(Engine.TensorMinValue(finite), NumOps.One)) return false;
        var magnitudes = Engine.TensorAbs(outputTensor);
        double absMax = NumOps.ToDouble(Engine.TensorMaxValue(magnitudes));
        // Summed in double: in Half four magnitudes of 32768 already sum past 65504, and an infinite sum let a
        // saturated output pass the engagement check below.
        double sumAbs = Engine.TensorSum(magnitudes.Cast<double>());
        // A sum of magnitudes is never negative, so this is the all-zero output (NaN still falls through).
        if (sumAbs <= 0.0) return false;

        // Input-conditioned engagement: an output uncorrelated with the input
        // (cosine ≈ 0) and saturated to one extreme is a tell-tale "ignore the
        // prompt" pattern. Require some non-trivial directional response.
        // Length-equality is required by CosineSimilarity — guard explicitly
        // so a shape mismatch can't throw mid-evaluation and abort the
        // batch (older callers may pass `input` and `output` from different
        // tokenizer / vocab paths).
        if (input is not null && input.Length > 0 && input.Length == output.Length)
        {
            double cos = VectorHelper.CosineSimilarity(output, input);
            if (Math.Abs(cos) < 1e-6 && absMax > 0.99 * (sumAbs / output.Length))
                return false;
        }
        else if (StrictHonestyMode)
        {
            // Strict mode: a pair we could not evaluate cannot count as
            // honest. Returning false here prevents non-comparable samples
            // from inflating HonestyScore.
            return false;
        }
        else
        {
            // Lenient default: surface the unscored pair as a Trace warning
            // so the inflation risk is observable in diagnostics. Warn at
            // most once per evaluation run — a large batch with many
            // non-comparable pairs would otherwise flood diagnostics and
            // pay the Trace overhead on every IsHonest call.
            if (!_hasLoggedNonComparableHonestyWarning)
            {
                System.Diagnostics.Trace.TraceWarning(
                    "RLHFAlignment.IsHonest: input/output not comparable (input null/empty " +
                    "or length mismatch). Counting as honest by default. Set StrictHonestyMode = " +
                    "true to count non-comparable pairs as dishonest instead. " +
                    "(This warning is emitted once per evaluation run.)");
                _hasLoggedNonComparableHonestyWarning = true;
            }
        }

        return true;
    }

    // Reset by EvaluateAlignment at the start of each batch so the warning
    // fires once per call rather than once per process lifetime.
    private bool _hasLoggedNonComparableHonestyWarning;

    private sealed class RlhfFineTunedPredictiveModel : IPredictiveModel<T, Vector<T>, Vector<T>>
    {
        private readonly IPredictiveModel<T, Vector<T>, Vector<T>> _baseModel;
        private readonly Func<Vector<T>, Vector<T>, double> _rewardModel;
        private readonly double _klCoefficient;

        public RlhfFineTunedPredictiveModel(IPredictiveModel<T, Vector<T>, Vector<T>> baseModel, Func<Vector<T>, Vector<T>, double> rewardModel, double klCoefficient)
        {
            Guard.NotNull(baseModel);
            _baseModel = baseModel;
            Guard.NotNull(rewardModel);
            _rewardModel = rewardModel;
            _klCoefficient = klCoefficient;
        }

        public Vector<T> Predict(Vector<T> input)
        {
            return KlRegularizedPolicy<T>.Apply(input, _baseModel.Predict(input), _rewardModel, _klCoefficient);
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
            Guard.NotNull(inner);
            _inner = inner;
            Guard.NotNull(alignment);
            _alignment = alignment;
            Guard.NotNull(principles);
            _principles = principles;
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
