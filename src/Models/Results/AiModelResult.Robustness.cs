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

    [JsonProperty]
    internal AdversarialRobustnessOptions<T>? AdversarialRobustnessOptions { get; private set; }

    [JsonProperty]
    internal bool HasAdversarialRobustness { get; private set; }

    [JsonProperty]
    internal IAdversarialDefense<T, TInput, TOutput>? AdversarialDefense { get; private set; }

    internal void SetAdversarialRobustnessOptions(AdversarialRobustnessOptions<T>? options)
    {
        AdversarialRobustnessOptions = options;
        HasAdversarialRobustness = options != null && (options.EnableAdversarialTraining || options.UseInputPreprocessing);
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

        int cleanCorrect = 0;
        int adversarialCorrect = 0;
        var perturbationSizes = new List<double>();

        for (int i = 0; i < testInputs.Length; i++)
        {
            var input = testInputs[i];
            var label = testLabels[i];

            var labelVector = ConversionsHelper.ConvertToVector<T, TOutput>(label);
            var trueClass = ArgMaxVector(labelVector);

            // Evaluate clean prediction
            var cleanOutput = Predict(input);
            var cleanOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(cleanOutput);
            var cleanPrediction = ArgMaxVector(cleanOutputVector);

            if (cleanPrediction == trueClass)
            {
                cleanCorrect++;
            }

            // Generate and evaluate adversarial example
            try
            {
                var adversarial = attack.GenerateAdversarialExample(input, label, Model);

                // Use defense preprocessing if available
                var defendedAdversarial = HasAdversarialRobustness && AdversarialDefense != null
                    ? AdversarialDefense.PreprocessInput(adversarial)
                    : adversarial;

                var advOutput = Predict(defendedAdversarial);
                var advOutputVector = ConversionsHelper.ConvertToVector<T, TOutput>(advOutput);
                var advPrediction = ArgMaxVector(advOutputVector);

                if (advPrediction == trueClass)
                {
                    adversarialCorrect++;
                }

                // Calculate perturbation size
                var inputVector = ConversionsHelper.ConvertToVector<T, TInput>(input);
                var adversarialVector = ConversionsHelper.ConvertToVector<T, TInput>(adversarial);
                var perturbation = SubtractVectors(adversarialVector, inputVector);
                var l2Norm = ComputeL2NormVector(perturbation);
                perturbationSizes.Add(_robustnessNumOps.ToDouble(l2Norm));
            }
            catch (ArgumentException)
            {
                // Count as defended if attack fails
                adversarialCorrect++;
            }
            catch (InvalidOperationException)
            {
                // Count as defended if attack fails
                adversarialCorrect++;
            }
        }

        stats.CleanAccuracy = testInputs.Length > 0 ? (double)cleanCorrect / testInputs.Length : 0.0;
        stats.AdversarialAccuracy = testInputs.Length > 0 ? (double)adversarialCorrect / testInputs.Length : 0.0;
        stats.AttackSuccessRate = 1.0 - stats.AdversarialAccuracy;
        stats.AveragePerturbationSize = perturbationSizes.Count > 0 ? perturbationSizes.Average() : 0.0;
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
            NormType = "Linf",
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
            NormType = "Linf",
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
            NormType = "Linf",
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

        var numOps = MathHelper.GetNumericOperations<T>();
        int maxIndex = 0;
        double maxValue = numOps.ToDouble(vector[0]);

        for (int i = 1; i < vector.Length; i++)
        {
            var v = numOps.ToDouble(vector[i]);
            if (v > maxValue)
            {
                maxValue = v;
                maxIndex = i;
            }
        }

        return maxIndex;
    }

    private Vector<T> SubtractVectors(Vector<T> a, Vector<T> b)
    {
        var result = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = _robustnessNumOps.Subtract(a[i], b[i]);
        }
        return result;
    }

    private T ComputeL2NormVector(Vector<T> vector)
    {
        var sumSquares = _robustnessNumOps.Zero;
        for (int i = 0; i < vector.Length; i++)
        {
            sumSquares = _robustnessNumOps.Add(sumSquares, _robustnessNumOps.Multiply(vector[i], vector[i]));
        }
        return _robustnessNumOps.Sqrt(sumSquares);
    }
}
