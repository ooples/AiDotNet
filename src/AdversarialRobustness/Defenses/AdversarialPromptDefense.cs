using System.Text;
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Validation;
using Newtonsoft.Json;

namespace AiDotNet.AdversarialRobustness.Defenses;

/// <summary>
/// Implements adaptive visual prompt-based defense that prepends learned perturbation-resistant
/// tokens/patches to inputs, improving adversarial robustness without model retraining.
/// </summary>
/// <remarks>
/// <para>
/// Visual prompt defense adds a set of learned "defense tokens" (for ViTs) or border patches
/// (for CNNs) to each input before inference. These prompts are optimized to absorb adversarial
/// perturbations, effectively neutralizing attacks. The key advantage is that the base model
/// weights remain frozen — only the prompt vectors are trained, making this defense extremely
/// lightweight and model-agnostic.
/// </para>
/// <para>
/// <b>For Beginners:</b> Imagine putting a special protective "frame" around every image before
/// showing it to the AI. This frame is carefully designed to cancel out any malicious changes an
/// attacker might have made. The AI itself doesn't need to change — the frame does all the
/// protective work. This is much faster to train than retraining the whole model.
/// </para>
/// <para>
/// <b>References:</b>
/// - RobustPrompt: Adaptive visual prompts, 61.1% improvement vs PGD (2025)
/// - Visual Prompting for Adversarial Robustness (Chen et al., ICASSP 2024)
/// - Adversarial Visual Prompt Tuning (Fu et al., 2023)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
[ModelDomain(ModelDomain.Vision)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Regularization)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Visual Prompting for Adversarial Robustness", "https://arxiv.org/abs/2210.06284", Year = 2023, Authors = "Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu")]
public class AdversarialPromptDefense<T, TInput, TOutput> : IAdversarialDefense<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the global execution engine for vectorized operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private AdversarialDefenseOptions<T> _options;
    [AiDotNet.Attributes.Scratch]
    private Vector<T>? _defensePrompt;
    private readonly int _promptLength;
    private readonly double _promptLearningRate;
    private readonly int _promptOptimizationSteps;
    private readonly Random _random;

    /// <summary>
    /// Initializes a new visual prompt-based adversarial defense.
    /// </summary>
    /// <param name="options">Base defense configuration.</param>
    /// <param name="promptLength">Number of defense prompt elements to prepend/add. Default: 32.</param>
    /// <param name="promptLearningRate">Learning rate for prompt optimization. Default: 0.01.</param>
    /// <param name="promptOptimizationSteps">Number of optimization steps for prompt training. Default: 100.</param>
    public AdversarialPromptDefense(
        AdversarialDefenseOptions<T> options,
        int promptLength = 32,
        double promptLearningRate = 0.01,
        int promptOptimizationSteps = 100)
    {
        Guard.NotNull(options);
        if (promptLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(promptLength), "promptLength must be positive.");
        if (promptLearningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(promptLearningRate), "promptLearningRate must be positive.");
        if (promptOptimizationSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(promptOptimizationSteps), "promptOptimizationSteps must be positive.");

        _options = options;
        _promptLength = promptLength;
        _promptLearningRate = promptLearningRate;
        _promptOptimizationSteps = promptOptimizationSteps;
        // The options' seed makes the prompt initialization and mini-batches reproducible; without one they are
        // drawn from a secure generator.
        _random = options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> ApplyDefense(
        TInput[] trainingData, TOutput[] labels, IFullModel<T, TInput, TOutput> model)
    {
        if (trainingData == null) throw new ArgumentNullException(nameof(trainingData));
        if (labels == null) throw new ArgumentNullException(nameof(labels));
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (trainingData.Length == 0)
            throw new ArgumentException("Training data must not be empty.", nameof(trainingData));
        if (trainingData.Length != labels.Length)
            throw new ArgumentException("Number of labels must match number of training samples.", nameof(labels));

        // Initialize defense prompt with small random values
        var promptData = new double[_promptLength];
        for (int i = 0; i < _promptLength; i++)
        {
            promptData[i] = (_random.NextDouble() - 0.5) * 0.1;
        }
        _defensePrompt = new Tensor<double>(promptData, new[] { _promptLength }).Cast<T>().ToVector();

        // Chen et al. (2023) train the prompt adversarially: min_delta E max_{||x_adv - x|| <= eps} L(f(x_adv + delta), y).
        // The inner maximization is FGSM at the prompted input; since the prompt is additive, attacking f at
        // x + delta and subtracting delta gives the adversarial input. The first AdversarialRatio share of each
        // mini-batch is adversarial, the rest clean.
        var attack = new FGSMAttack<T, TInput, TOutput>(new AdversarialAttackOptions<T>
        {
            Epsilon = _options.Epsilon,
            NormType = AdversarialNormType.LInfinity
        });
        T delta = NumOps.FromDouble(FiniteDifferenceStep);
        T learningRate = NumOps.FromDouble(_promptLearningRate);
        T upperLimit = NumOps.One;
        T lowerLimit = NumOps.Negate(upperLimit);
        int batchSize = Math.Min(32, trainingData.Length);
        T batchDivisor = NumOps.FromDouble(batchSize);

        // Optimize prompt to maximize accuracy on adversarial inputs
        for (int step = 0; step < _promptOptimizationSteps; step++)
        {
            // Sample a mini-batch
            int adversarialCount = (int)Math.Round(_options.AdversarialRatio * batchSize);
            var gradientAccumulator = new Vector<T>(_promptLength);

            for (int b = 0; b < batchSize; b++)
            {
                int idx = _random.Next(trainingData.Length);
                var input = b < adversarialCount
                    ? AdversarialInput(trainingData[idx], labels[idx], model, attack)
                    : trainingData[idx];
                var labelVec = ConversionsHelper.ConvertToVector<T, TOutput>(labels[idx]);
                gradientAccumulator = Engine.Add<T>(gradientAccumulator, PromptGradient(input, labelVec, model, delta));
            }

            // Gradient descent on the averaged gradient, then clamp prompt values to [-1, 1]
            var averaged = Engine.Divide<T>(gradientAccumulator, batchDivisor);
            var current = _defensePrompt ?? new Vector<T>(_promptLength);
            var updated = Engine.Subtract<T>(current, Engine.Multiply<T>(averaged, learningRate));
            _defensePrompt = Engine.Clamp<T>(updated, lowerLimit, upperLimit);
        }

        // The model itself is unchanged — defense is applied via PreprocessInput
        return model;
    }

    /// <summary>Central-difference step for the prompt gradient.</summary>
    private const double FiniteDifferenceStep = 0.001;

    /// <summary>
    /// The central-difference gradient of the MSE loss with respect to every prompt entry.
    /// </summary>
    private Vector<T> PromptGradient(TInput input, Vector<T> labelVec, IFullModel<T, TInput, TOutput> model, T delta)
    {
        var prompt = _defensePrompt ?? new Vector<T>(_promptLength);
        var gradient = new T[_promptLength];
        T twoDelta = NumOps.Add(delta, delta);
        for (int p = 0; p < _promptLength; p++)
        {
            var direction = Engine.Multiply<T>(Vector<T>.CreateStandardBasis(_promptLength, p), delta);
            T forward = LossWithPrompt(input, labelVec, model, Engine.Add<T>(prompt, direction));
            T backward = LossWithPrompt(input, labelVec, model, Engine.Subtract<T>(prompt, direction));
            gradient[p] = Difference(forward, backward, twoDelta);
        }

        _defensePrompt = prompt;
        return new Vector<T>(gradient);
    }

    private static T Difference(T forward, T backward, T width) => NumOps.Divide(NumOps.Subtract(forward, backward), width);

    private T LossWithPrompt(TInput input, Vector<T> labelVec, IFullModel<T, TInput, TOutput> model, Vector<T> prompt)
    {
        _defensePrompt = prompt;
        var output = ConversionsHelper.ConvertToVector<T, TOutput>(model.Predict(ApplyPromptToInput(input)));
        return ComputeLoss(output, labelVec);
    }

    /// <summary>
    /// FGSM(f, x + delta) - delta: the worst case for the prompted model within the epsilon ball around x.
    /// </summary>
    private TInput AdversarialInput(TInput input, TOutput label, IFullModel<T, TInput, TOutput> model, FGSMAttack<T, TInput, TOutput> attack)
    {
        var prompted = ApplyPromptToInput(input);
        var attacked = ConversionsHelper.ConvertToVector<T, TInput>(attack.GenerateAdversarialExample(prompted, label, model));
        var offset = Engine.Subtract<T>(attacked, ConversionsHelper.ConvertToVector<T, TInput>(prompted));
        var adversarial = Engine.Add<T>(ConversionsHelper.ConvertToVector<T, TInput>(input), offset);
        return ConversionsHelper.ConvertVectorToInput<T, TInput>(adversarial, input);
    }

    /// <inheritdoc/>
    public TInput PreprocessInput(TInput input)
    {
        if (_defensePrompt == null) return input;
        return ApplyPromptToInput(input);
    }

    /// <inheritdoc/>
    public RobustnessMetrics<T> EvaluateRobustness(
        IFullModel<T, TInput, TOutput> model,
        TInput[] testData,
        TOutput[] labels,
        IAdversarialAttack<T, TInput, TOutput> attack)
    {
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (testData == null) throw new ArgumentNullException(nameof(testData));
        if (labels == null) throw new ArgumentNullException(nameof(labels));
        if (attack == null) throw new ArgumentNullException(nameof(attack));
        if (testData.Length != labels.Length)
            throw new ArgumentException("Number of labels must match number of test samples.", nameof(labels));
        if (testData.Length == 0)
            throw new ArgumentException("Test data must not be empty.", nameof(testData));

        int cleanCorrect = 0, adversarialCorrect = 0;

        for (int i = 0; i < testData.Length; i++)
        {
            var labelVec = ConversionsHelper.ConvertToVector<T, TOutput>(labels[i]);
            int trueClass = ArgMax(labelVec);

            // Clean accuracy with prompt defense
            var defended = PreprocessInput(testData[i]);
            var cleanOutput = model.Predict(defended);
            var cleanVec = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
            if (ArgMax(cleanVec) == trueClass) cleanCorrect++;

            // Adversarial accuracy with prompt defense
            var adversarial = attack.GenerateAdversarialExample(testData[i], labels[i], model);
            var defendedAdv = PreprocessInput(adversarial);
            var advOutput = model.Predict(defendedAdv);
            var advVec = ConversionsHelper.ConvertToVector<T, TOutput>(advOutput);
            if (ArgMax(advVec) == trueClass) adversarialCorrect++;
        }

        return new RobustnessMetrics<T>
        {
            CleanAccuracy = (double)cleanCorrect / testData.Length,
            AdversarialAccuracy = (double)adversarialCorrect / testData.Length,
            RobustnessScore = ((double)cleanCorrect + adversarialCorrect) / (2.0 * testData.Length),
            AttackSuccessRate = 1.0 - (double)adversarialCorrect / testData.Length
        };
    }

    /// <inheritdoc/>
    public AdversarialDefenseOptions<T> GetOptions() => _options;

    /// <inheritdoc/>
    public void Reset()
    {
        _defensePrompt = null;
    }

    /// <inheritdoc/>
    public byte[] Serialize()
    {
        ModelPersistenceGuard.EnforceBeforeSerialize();
        var state = new PromptDefenseState
        {
            Options = _options,
            PromptData = _defensePrompt != null ? PromptToDoubleArray(_defensePrompt) : null
        };
        var json = JsonConvert.SerializeObject(state, Formatting.None);
        return Encoding.UTF8.GetBytes(json);
    }

    /// <inheritdoc/>
    public void Deserialize(byte[] data)
    {
        ModelPersistenceGuard.EnforceBeforeDeserialize();
        if (data == null) throw new ArgumentNullException(nameof(data));
        var json = Encoding.UTF8.GetString(data);
        var state = JsonConvert.DeserializeObject<PromptDefenseState>(json);
        _options = state?.Options ?? new AdversarialDefenseOptions<T>();
        if (state?.PromptData != null)
        {
            if (state.PromptData.Length != _promptLength)
                throw new InvalidOperationException(
                    $"Serialized prompt length ({state.PromptData.Length}) does not match expected prompt length ({_promptLength}).");
            _defensePrompt = new Tensor<double>(state.PromptData, new[] { state.PromptData.Length }).Cast<T>().ToVector();
        }
        else
        {
            _defensePrompt = null;
        }
    }

    private static double[] PromptToDoubleArray(Vector<T> prompt)
    {
        return prompt.AsTensor().Cast<double>().ToArray();
    }

    private sealed class PromptDefenseState
    {
        public AdversarialDefenseOptions<T>? Options { get; set; }
        public double[]? PromptData { get; set; }
    }

    /// <inheritdoc/>
    public void SaveModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeSave();
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        File.WriteAllBytes(filePath, Serialize());
    }

    /// <inheritdoc/>
    public void LoadModel(string filePath)
    {
        Helpers.ModelPersistenceGuard.EnforceBeforeLoad();
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found: {filePath}", filePath);
        Deserialize(File.ReadAllBytes(filePath));
    }

    /// <summary>
    /// Adds the prompt as a border frame (Chen et al. 2023, an additive padding prompt): the first half of the
    /// prompt goes on the first entries of the flattened input and the second half on the last ones, so a
    /// row-major image gets a band at its top and bottom. The input keeps its size, so the frozen model sees
    /// the shape it was trained on.
    /// </summary>
    private TInput ApplyPromptToInput(TInput input)
    {
        if (_defensePrompt == null) return input;

        var inputVec = ConversionsHelper.ConvertToVector<T, TInput>(input);
        int n = inputVec.Length;
        int length = Math.Min(_promptLength, n);
        int head = (length + 1) / 2;
        int tail = length - head;
        var prompt = _defensePrompt.AsTensor();
        var frame = new Tensor<T>(new[] { n });
        frame = Engine.TensorSetSlice(frame, Engine.TensorSlice(prompt, new[] { 0 }, new[] { head }), new[] { 0 });
        if (tail > 0)
        {
            frame = Engine.TensorSetSlice(frame, Engine.TensorSlice(prompt, new[] { head }, new[] { tail }), new[] { n - tail });
        }

        var combinedVec = Engine.TensorAdd(inputVec.AsTensor(), frame).ToVector();
        return ConversionsHelper.ConvertVectorToInput<T, TInput>(combinedVec, input);
    }

    private static T ComputeLoss(Vector<T> predicted, Vector<T> target)
    {
        // Simple MSE loss over the common length
        int len = Math.Min(predicted.Length, target.Length);
        if (len == 0)
        {
            return NumOps.Zero;
        }

        var engine = AiDotNetEngine.Current;
        var difference = engine.TensorSubtract(
            engine.TensorSlice(predicted.AsTensor(), new[] { 0 }, new[] { len }),
            engine.TensorSlice(target.AsTensor(), new[] { 0 }, new[] { len }));
        return engine.TensorMean(engine.TensorSquare(difference));
    }

    private static int ArgMax(Vector<T> vector)
    {
        return AiDotNetEngine.Current.TensorArgMax(vector.AsTensor(), 0).GetFlat(0);
    }
}
