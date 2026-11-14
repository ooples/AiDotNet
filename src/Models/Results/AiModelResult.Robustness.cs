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
    /// <para>
    /// For a targeted attack (<see cref="AdversarialAttackOptions{T}.IsTargeted"/>) the attack receives the one-hot
    /// label of <see cref="AdversarialAttackOptions{T}.TargetClass"/> instead of the true label, and the attack
    /// success rate counts the clean-correct inputs whose adversarial prediction becomes the target class.
    /// </para>
    /// </remarks>
    /// <param name="testInputs">The test inputs to evaluate on.</param>
    /// <param name="testLabels">The true labels for the test inputs.</param>
    /// <param name="attack">The adversarial attack to evaluate against.</param>
    /// <returns>Robustness metrics including clean accuracy, adversarial accuracy, and attack success rate.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A targeted attack names a class the labels do not have.</exception>
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

        var attackOptions = attack.GetOptions();
        bool targeted = attackOptions.IsTargeted;
        int targetClass = attackOptions.TargetClass;
        var stats = new RobustnessStats<T>
        {
            AttackType = attack.GetType().Name,
            NormType = attackOptions.NormType,
            EvaluationEpsilon = attackOptions.Epsilon
        };

        // RobustBench / AutoAttack accounting: an example is robust only when the clean prediction is correct
        // and the prediction on its adversarial example is still correct. Inputs the model already gets wrong
        // count as non-robust (they need no attack), and the attack success rate is the fraction of
        // correctly classified clean inputs that the attack flips (targeted: moves to the target class). A
        // failing attack propagates its exception instead of being counted as a successful defense. The clean
        // and the adversarial predictions both go through the configured defense, so both accuracies describe
        // the same defended classifier.
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

            var attackLabel = targeted ? TargetLabel(labelVector, label, targetClass) : label;
            var adversarial = attack.GenerateAdversarialExample(input, attackLabel, Model);

            int adversarialClass = ArgMaxVector(ConversionsHelper.ConvertToVector<T, TOutput>(PredictWithDefense(adversarial)));
            if (adversarialClass == trueClass)
            {
                robustCorrect++;
            }

            bool succeeded = targeted
                ? adversarialClass == targetClass && targetClass != trueClass
                : adversarialClass != trueClass;
            if (!succeeded)
            {
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

        stats.CleanAccuracy = testInputs.Length > 0 ? (double)cleanCorrect / testInputs.Length : 0.0;
        stats.AdversarialAccuracy = testInputs.Length > 0 ? (double)robustCorrect / testInputs.Length : 0.0;
        stats.AttackSuccessRate = cleanCorrect > 0 ? (double)successfulPerturbationNorms.Count / cleanCorrect : 0.0;
        stats.AveragePerturbationSize = successfulPerturbationNorms.Count > 0
            ? OverflowSafeReductions.Mean(new Vector<T>(successfulPerturbationNorms).AsTensor())
            : _robustnessNumOps.Zero;
        stats.RobustnessScore = (stats.CleanAccuracy + stats.AdversarialAccuracy) / 2.0;
        stats.IsEvaluated = true;

        return stats;
    }

    /// <summary>
    /// Evaluates the model's robustness with a PGD attack built from the configured attack settings.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a convenience method that uses PGD (Projected Gradient Descent).
    /// PGD is one of the strongest gradient-based attacks and is commonly used as a benchmark for
    /// adversarial robustness.</para>
    /// <para>
    /// With adversarial robustness options configured, the attack uses their epsilon, step size, iteration count,
    /// norm, targeting and random start (see <see cref="AdversarialRobustnessOptions{T}"/>). Without them it uses
    /// epsilon 0.03, step epsilon / 4, 10 iterations, the L-infinity norm and a random start.
    /// </para>
    /// </remarks>
    /// <param name="testInputs">The test inputs to evaluate on.</param>
    /// <param name="testLabels">The true labels for the test inputs.</param>
    /// <returns>Robustness metrics.</returns>
    public RobustnessStats<T> EvaluateRobustness(
        TInput[] testInputs,
        TOutput[] testLabels)
    {
        var attack = new PGDAttack<T, TInput, TOutput>(ConfiguredAttackOptions(null));
        return EvaluateRobustness(testInputs, testLabels, attack);
    }

    /// <summary>
    /// Evaluates the model's robustness with a PGD attack of the given budget and the configured attack settings.
    /// </summary>
    /// <remarks>
    /// The same as <see cref="EvaluateRobustness(TInput[], TOutput[])"/>, except that <paramref name="epsilon"/>
    /// overrides the configured <c>AttackEpsilon</c>.
    /// </remarks>
    /// <param name="testInputs">The test inputs to evaluate on.</param>
    /// <param name="testLabels">The true labels for the test inputs.</param>
    /// <param name="epsilon">The perturbation budget; must be positive and finite.</param>
    /// <returns>Robustness metrics.</returns>
    public RobustnessStats<T> EvaluateRobustness(
        TInput[] testInputs,
        TOutput[] testLabels,
        double epsilon)
    {
        var attack = new PGDAttack<T, TInput, TOutput>(ConfiguredAttackOptions(epsilon));
        return EvaluateRobustness(testInputs, testLabels, attack);
    }

    /// <summary>
    /// Evaluates the model's robustness using AutoAttack (ensemble of diverse attacks).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> AutoAttack runs multiple different attack methods and picks
    /// the best result. This provides a more reliable robustness evaluation because if your
    /// model can resist AutoAttack, it's genuinely robust against a wide variety of attacks.</para>
    /// <para>The attack settings come from the configured options, as for
    /// <see cref="EvaluateRobustness(TInput[], TOutput[])"/>.</para>
    /// </remarks>
    /// <param name="testInputs">The test inputs to evaluate on.</param>
    /// <param name="testLabels">The true labels for the test inputs.</param>
    /// <param name="epsilon">The perturbation budget; null uses the configured <c>AttackEpsilon</c> (or 0.03).</param>
    /// <returns>Robustness metrics.</returns>
    public RobustnessStats<T> EvaluateRobustnessWithAutoAttack(
        TInput[] testInputs,
        TOutput[] testLabels,
        double? epsilon = null)
    {
        var attack = new AutoAttack<T, TInput, TOutput>(ConfiguredAttackOptions(epsilon));
        return EvaluateRobustness(testInputs, testLabels, attack);
    }

    /// <summary>
    /// Runs every attack in the configured <c>AttackMethods</c> and returns the robustness metrics of each.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the robustness test that <c>EnableRobustnessTesting</c> switches on.
    /// Each attack uses the configured epsilon, step size, iterations, norm, targeting and random start.</para>
    /// </remarks>
    /// <param name="testInputs">The test inputs to evaluate on.</param>
    /// <param name="testLabels">The true labels for the test inputs.</param>
    /// <returns>The metrics of each configured attack, keyed by attack method.</returns>
    /// <exception cref="InvalidOperationException">
    /// No options are configured, robustness testing is disabled, or the attack settings are invalid.
    /// </exception>
    public IReadOnlyDictionary<AdversarialAttackMethod, RobustnessStats<T>> EvaluateConfiguredRobustness(
        TInput[] testInputs,
        TOutput[] testLabels)
    {
        var configured = AdversarialRobustnessOptions
            ?? throw new InvalidOperationException(
                "No adversarial robustness options are configured. Call ConfigureAdversarialRobustness on the builder " +
                "with EnableRobustnessTesting set to true.");
        if (!configured.EnableRobustnessTesting)
        {
            throw new InvalidOperationException(
                "Robustness testing is disabled. Set AdversarialRobustnessOptions.EnableRobustnessTesting to true to run " +
                "the configured attacks.");
        }

        var methods = configured.AttackMethods;
        if (methods is null || methods.Length == 0)
        {
            throw new InvalidOperationException(
                "AdversarialRobustnessOptions.AttackMethods must name at least one attack when robustness testing is enabled.");
        }

        var results = new Dictionary<AdversarialAttackMethod, RobustnessStats<T>>();
        foreach (var method in methods)
        {
            if (results.ContainsKey(method))
            {
                throw new InvalidOperationException(
                    $"AdversarialRobustnessOptions.AttackMethods lists {method} more than once.");
            }

            results[method] = EvaluateRobustness(testInputs, testLabels, CreateAttack(method, ConfiguredAttackOptions(null)));
        }

        return results;
    }

    /// <summary>
    /// Generates an adversarial example for a given input.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method creates a slightly modified version of the input
    /// that is designed to fool the model. The modification is small enough to be imperceptible
    /// to humans but can cause the model to make incorrect predictions.</para>
    /// <para>The PGD attack uses the configured attack settings, as for
    /// <see cref="EvaluateRobustness(TInput[], TOutput[])"/>; a targeted configuration pushes the input
    /// toward <c>TargetClass</c>.</para>
    /// </remarks>
    /// <param name="input">The clean input to perturb.</param>
    /// <param name="trueLabel">The correct label for the input.</param>
    /// <param name="epsilon">The perturbation budget; null uses the configured <c>AttackEpsilon</c> (or 0.03).</param>
    /// <returns>The adversarial example.</returns>
    public TInput GenerateAdversarialExample(
        TInput input,
        TOutput trueLabel,
        double? epsilon = null)
    {
        if (Model == null)
        {
            throw new InvalidOperationException("Model is not initialized.");
        }

        var attackOptions = ConfiguredAttackOptions(epsilon);
        var attack = new PGDAttack<T, TInput, TOutput>(attackOptions);
        var attackLabel = attackOptions.IsTargeted
            ? TargetLabel(ConversionsHelper.ConvertToVector<T, TOutput>(trueLabel), trueLabel, attackOptions.TargetClass)
            : trueLabel;
        return attack.GenerateAdversarialExample(input, attackLabel, Model);
    }

    /// <summary>
    /// The attack settings for the convenience robustness methods: the configured options, or the historical
    /// defaults (epsilon 0.03, step epsilon / 4, 10 iterations, L-infinity, random start) when none are configured.
    /// </summary>
    private AdversarialAttackOptions<T> ConfiguredAttackOptions(double? epsilon)
    {
        if (epsilon.HasValue && !(epsilon.Value > 0 && !double.IsInfinity(epsilon.Value)))
        {
            throw new ArgumentOutOfRangeException(nameof(epsilon), epsilon.Value, "Epsilon must be a positive finite number.");
        }

        var configured = AdversarialRobustnessOptions;
        if (configured is null)
        {
            double budget = epsilon ?? 0.03;
            return new AdversarialAttackOptions<T>
            {
                Epsilon = budget,
                StepSize = budget / 4.0,
                Iterations = 10,
                NormType = AdversarialNormType.LInfinity,
                UseRandomStart = true
            };
        }

        double attackEpsilon = epsilon ?? configured.AttackEpsilon;
        if (!(attackEpsilon > 0 && !double.IsInfinity(attackEpsilon)))
        {
            throw new InvalidOperationException(
                $"AdversarialRobustnessOptions.AttackEpsilon must be a positive finite number (was {configured.AttackEpsilon}).");
        }

        if (!(configured.AttackStepSize > 0 && !double.IsInfinity(configured.AttackStepSize)))
        {
            throw new InvalidOperationException(
                $"AdversarialRobustnessOptions.AttackStepSize must be a positive finite number (was {configured.AttackStepSize}).");
        }

        if (configured.AttackIterations < 1)
        {
            throw new InvalidOperationException(
                $"AdversarialRobustnessOptions.AttackIterations must be at least 1 (was {configured.AttackIterations}).");
        }

        if (!Enum.IsDefined(typeof(AdversarialNormType), configured.AttackNormType))
        {
            throw new InvalidOperationException(
                $"AdversarialRobustnessOptions.AttackNormType has the undefined value {(int)configured.AttackNormType}.");
        }

        if (configured.UseTargetedAttacks && configured.TargetClass < 0)
        {
            throw new InvalidOperationException(
                $"AdversarialRobustnessOptions.TargetClass must be a class index of at least 0 for targeted attacks (was {configured.TargetClass}).");
        }

        return new AdversarialAttackOptions<T>
        {
            Epsilon = attackEpsilon,
            StepSize = configured.AttackStepSize,
            Iterations = configured.AttackIterations,
            NormType = configured.AttackNormType,
            IsTargeted = configured.UseTargetedAttacks,
            TargetClass = configured.TargetClass,
            UseRandomStart = configured.UseRandomStartForAttacks
        };
    }

    private static IAdversarialAttack<T, TInput, TOutput> CreateAttack(
        AdversarialAttackMethod method,
        AdversarialAttackOptions<T> options)
    {
        return method switch
        {
            AdversarialAttackMethod.Fgsm => new FGSMAttack<T, TInput, TOutput>(options),
            AdversarialAttackMethod.Pgd => new PGDAttack<T, TInput, TOutput>(options),
            AdversarialAttackMethod.CarliniWagner => new CWAttack<T, TInput, TOutput>(options),
            AdversarialAttackMethod.AutoAttack => new AutoAttack<T, TInput, TOutput>(options),
            _ => throw new InvalidOperationException(
                $"AdversarialRobustnessOptions.AttackMethods contains the undefined value {(int)method}.")
        };
    }

    /// <summary>
    /// The one-hot label of <paramref name="targetClass"/>, shaped like <paramref name="label"/>.
    /// </summary>
    private static TOutput TargetLabel(Vector<T> labelVector, TOutput label, int targetClass)
    {
        if (targetClass < 0 || targetClass >= labelVector.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(targetClass),
                targetClass,
                $"The target class must be a class index below the label length {labelVector.Length}.");
        }

        var target = Vector<T>.CreateStandardBasis(labelVector.Length, targetClass);
        return ConversionsHelper.ConvertVectorToInput<T, TOutput>(target, label);
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
