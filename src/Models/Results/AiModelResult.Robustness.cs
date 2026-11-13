using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using Newtonsoft.Json;

namespace AiDotNet.Models.Results;

public partial class AiModelResult<T, TInput, TOutput>
{
    private readonly INumericOperations<T> _robustnessNumOps = MathHelper.GetNumericOperations<T>();
    private static IEngine RobustnessEngine => AiDotNetEngine.Current;

    [JsonProperty]
    internal AdversarialRobustnessOptions<T>? AdversarialRobustnessOptions { get; private set; }

    /// <summary>
    /// Gets whether this result has any adversarial robustness behaviour to apply at inference time.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Derived rather than stored. A configured <see cref="AdversarialDefense"/> IS adversarial
    /// robustness: a caller who supplies a defense has asked for it to run, and must not additionally
    /// have to discover an unrelated-sounding option flag
    /// (<c>EnableAdversarialTraining</c>/<c>UseInputPreprocessing</c>, both of which default to false)
    /// before their defense is honoured. Previously this was written in exactly one place — from those
    /// two flags — so a supplied defense was stored on the result and then silently ignored at
    /// <see cref="PredictWithDefense"/>.
    /// </para>
    /// <para>
    /// Deriving also removes an ordering hazard: while this was a stored flag,
    /// <see cref="SetAdversarialRobustnessOptions"/> unconditionally overwrote it, so a value set by
    /// <see cref="SetAdversarialDefense"/> would be clobbered depending on call order. The
    /// options-flag clause below is the previous expression verbatim, so flag-driven behaviour is
    /// unchanged.
    /// </para>
    /// <para>
    /// Not serialized: it is computed from <see cref="AdversarialRobustnessOptions"/> and
    /// <see cref="AdversarialDefense"/>, which are themselves persisted, and Newtonsoft cannot write
    /// a value back into a getter-only member.
    /// </para>
    /// </remarks>
    [JsonIgnore]
    internal bool HasAdversarialRobustness =>
        AdversarialDefense != null ||
        (AdversarialRobustnessOptions != null &&
         (AdversarialRobustnessOptions.EnableAdversarialTraining || AdversarialRobustnessOptions.UseInputPreprocessing));

    [JsonProperty]
    internal IAdversarialDefense<T, TInput, TOutput>? AdversarialDefense { get; private set; }

    internal void SetAdversarialRobustnessOptions(AdversarialRobustnessOptions<T>? options)
    {
        AdversarialRobustnessOptions = options;
    }

    internal void SetAdversarialDefense(IAdversarialDefense<T, TInput, TOutput>? defense)
    {
        AdversarialDefense = defense;
    }

    /// <summary>
    /// Makes a prediction with adversarial preprocessing applied if configured.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method applies defensive preprocessing to the input
    /// before making a prediction. Preprocessing can include techniques like:
    /// - JPEG compression to remove high-frequency adversarial perturbations
    /// - Bit depth reduction to quantize away small perturbations
    /// - Denoising to smooth out adversarial noise
    /// </para>
    /// </remarks>
    /// <param name="input">The input to make a prediction on.</param>
    /// <returns>The model's prediction after defensive preprocessing.</returns>
    public TOutput PredictWithDefense(TInput input)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        if (!HasAdversarialRobustness || AdversarialDefense == null)
        {
            return Predict(input);
        }

        var preprocessedInput = AdversarialDefense.PreprocessInput(input);
        return Predict(preprocessedInput);
    }

    /// <summary>
    /// Evaluates the model's robustness against a specific adversarial attack.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method tests how well your model resists a specific
    /// type of adversarial attack. It:
    /// 1. Makes predictions on clean (unmodified) inputs
    /// 2. Generates adversarial examples using the specified attack
    /// 3. Makes predictions on the adversarial examples
    /// 4. Computes metrics comparing clean vs adversarial performance
    /// </para>
    /// </remarks>
    /// <param name="testInputs">The test inputs to evaluate on.</param>
    /// <param name="testLabels">The true labels for the test inputs.</param>
    /// <param name="attack">The adversarial attack to evaluate against.</param>
    /// <returns>Robustness metrics including clean accuracy, adversarial accuracy, and attack success rate.</returns>
    public RobustnessStats<T> EvaluateRobustness(
        TInput[] testInputs,
        TOutput[] testLabels,
        IAdversarialAttack<T, TInput, TOutput> attack)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        if (testInputs == null)
        {
            throw new ArgumentNullException(nameof(testInputs));
        }

        if (testLabels == null)
        {
            throw new ArgumentNullException(nameof(testLabels));
        }

        if (attack == null)
        {
            throw new ArgumentNullException(nameof(attack));
        }

        if (testInputs.Length != testLabels.Length)
        {
            throw new ArgumentException("Number of test inputs must match number of labels.", nameof(testLabels));
        }

        var stats = new RobustnessStats<T>
        {
            AttackType = attack.GetType().Name,
            NormType = attack.GetOptions().NormType,
            EvaluationEpsilon = attack.GetOptions().Epsilon
        };

        // RobustBench / AutoAttack accounting: an example is robust only when the clean prediction is correct
        // and the prediction on its adversarial example is still correct. Inputs the model already gets wrong
        // count as non-robust (they need no attack), and the attack success rate is the fraction of
        // correctly classified clean inputs that the attack flips. A failing attack propagates its exception
        // instead of being counted as a successful defense. The clean and the adversarial predictions both
        // go through the configured defense, so both accuracies describe the same defended classifier.
        int cleanCorrect = 0;
        int robustCorrect = 0;
        var successfulPerturbationNorms = new List<T>();

        for (int i = 0; i < testInputs.Length; i++)
        {
            var input = testInputs[i];
            var label = testLabels[i];

            var labelVector = ConversionsHelper.ConvertToVector<T, TOutput>(label);
            var trueClass = ArgMaxVector(labelVector);

            var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(PredictWithDefense(input));
            if (ArgMaxVector(cleanOutputVector) != trueClass)
            {
                continue;
            }

            cleanCorrect++;

            var adversarial = attack.GenerateAdversarialExample(input, label, Model);

            var advOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(PredictWithDefense(adversarial));
            if (ArgMaxVector(advOutputVector) == trueClass)
            {
                robustCorrect++;
                continue;
            }

            // The attack succeeded: record the L2 size of the perturbation it needed.
            var inputVector = ConversionsHelper.ConvertToVector<T, TInput>(input);
            var adversarialVector = ConversionsHelper.ConvertToVector<T, TInput>(adversarial);
            // The engine's tensor norm accumulates in double, so a Half perturbation norm above 256 stays finite.
            var perturbation = SubtractVectors(adversarialVector, inputVector);
            successfulPerturbationNorms.Add(perturbation.Length == 0
                ? _robustnessNumOps.Zero
                : RobustnessEngine.TensorNorm(perturbation.AsTensor()).Data.Span[0]);
        }

        int fooled = cleanCorrect - robustCorrect;
        stats.CleanAccuracy = testInputs.Length > 0 ? (double)cleanCorrect / testInputs.Length : 0.0;
        stats.AdversarialAccuracy = testInputs.Length > 0 ? (double)robustCorrect / testInputs.Length : 0.0;
        stats.AttackSuccessRate = cleanCorrect > 0 ? (double)fooled / cleanCorrect : 0.0;
        stats.AveragePerturbationSize = successfulPerturbationNorms.Count > 0
            ? OverflowSafeReductions.Mean(new Vector<T>(successfulPerturbationNorms).AsTensor())
            : _robustnessNumOps.Zero;
        stats.RobustnessScore = (stats.CleanAccuracy + stats.AdversarialAccuracy) / 2.0;
        stats.IsEvaluated = true;

        return stats;
    }

    /// <summary>
    /// Evaluates the model's robustness using the default attack configuration.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a convenience method that uses PGD (Projected Gradient Descent)
    /// attack with default settings. PGD is one of the strongest gradient-based attacks and is
    /// commonly used as a benchmark for adversarial robustness.</para>
    /// </remarks>
    /// <param name="testInputs">The test inputs to evaluate on.</param>
    /// <param name="testLabels">The true labels for the test inputs.</param>
    /// <param name="epsilon">The maximum perturbation size (default: 0.03).</param>
    /// <returns>Robustness metrics.</returns>
    public RobustnessStats<T> EvaluateRobustness(
        TInput[] testInputs,
        TOutput[] testLabels,
        double epsilon = 0.03)
    {
        var attackOptions = new AdversarialAttackOptions<T>
        {
            Epsilon = epsilon,
            StepSize = epsilon / 4.0,
            Iterations = 10,
            NormType = AdversarialNormType.LInfinity,
            UseRandomStart = true
        };

        var attack = new PGDAttack<T, TInput, TOutput>(attackOptions);
        return EvaluateRobustness(testInputs, testLabels, attack);
    }

    /// <summary>
    /// Evaluates the model's robustness using AutoAttack (ensemble of diverse attacks).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> AutoAttack runs multiple different attack methods and picks
    /// the best result. This provides a more reliable robustness evaluation because if your
    /// model can resist AutoAttack, it's genuinely robust against a wide variety of attacks.</para>
    /// </remarks>
    /// <param name="testInputs">The test inputs to evaluate on.</param>
    /// <param name="testLabels">The true labels for the test inputs.</param>
    /// <param name="epsilon">The maximum perturbation size (default: 0.03).</param>
    /// <returns>Robustness metrics.</returns>
    public RobustnessStats<T> EvaluateRobustnessWithAutoAttack(
        TInput[] testInputs,
        TOutput[] testLabels,
        double epsilon = 0.03)
    {
        var attackOptions = new AdversarialAttackOptions<T>
        {
            Epsilon = epsilon,
            StepSize = epsilon / 4.0,
            Iterations = 10,
            NormType = AdversarialNormType.LInfinity,
            UseRandomStart = true
        };

        var attack = new AutoAttack<T, TInput, TOutput>(attackOptions);
        return EvaluateRobustness(testInputs, testLabels, attack);
    }

    /// <summary>
    /// Generates an adversarial example for a given input.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a slightly modified version of the input
    /// that is designed to fool the model. The modification is small enough to be imperceptible
    /// to humans but can cause the model to make incorrect predictions.</para>
    /// </remarks>
    /// <param name="input">The clean input to perturb.</param>
    /// <param name="trueLabel">The correct label for the input.</param>
    /// <param name="epsilon">The maximum perturbation size (default: 0.03).</param>
    /// <returns>The adversarial example.</returns>
    public TInput GenerateAdversarialExample(
        TInput input,
        TOutput trueLabel,
        double epsilon = 0.03)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        var attackOptions = new AdversarialAttackOptions<T>
        {
            Epsilon = epsilon,
            StepSize = epsilon / 4.0,
            Iterations = 10,
            NormType = AdversarialNormType.LInfinity,
            UseRandomStart = true
        };

        var attack = new PGDAttack<T, TInput, TOutput>(attackOptions);
        return attack.GenerateAdversarialExample(input, trueLabel, Model);
    }

    private static int ArgMaxVector(Vector<T> vector)
    {
        if (vector == null || vector.Length == 0)
        {
            return 0;
        }

        // First maximal index, the same tie rule as a strict ">" scan from index 0.
        return RobustnessEngine.TensorArgMax(vector.AsTensor(), 0).Data.Span[0];
    }

    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        return RobustnessEngine.Subtract(a, b);
    }

}
