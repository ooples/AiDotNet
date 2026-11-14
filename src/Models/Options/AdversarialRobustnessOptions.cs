using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Unified configuration options for adversarial robustness and AI safety.
/// </summary>
/// <remarks>
/// <para>
/// This comprehensive options class consolidates all adversarial robustness settings including:
/// - Safety filtering (input validation, output filtering, harmful content detection)
/// - Adversarial attacks (for robustness testing and evaluation)
/// - Adversarial defenses (training and preprocessing)
/// - Certified robustness (provable guarantees)
/// - Content moderation (for LLM applications)
/// </para>
/// <para><b>For Beginners:</b> This is your one-stop shop for all AI safety and robustness settings.
/// You can configure how strictly inputs are validated, how the model is protected against attacks,
/// and what guarantees you want about model predictions.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
public class AdversarialRobustnessOptions<T> : ModelOptions
{
    // ========================================================================
    // SAFETY FILTERING OPTIONS
    // ========================================================================

    /// <summary>
    /// Gets or sets whether safety filtering is enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the master switch for safety filtering.
    /// When enabled, inputs and outputs are validated for safety.</para>
    /// </remarks>
    public bool EnableSafetyFiltering { get; set; } = true;

    /// <summary>
    /// Gets or sets the safety threshold for content filtering.
    /// </summary>
    /// <value>The threshold (0-1), defaulting to 0.8.</value>
    public double SafetyThreshold { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets the jailbreak detection sensitivity.
    /// </summary>
    /// <value>The sensitivity (0-1), defaulting to 0.7.</value>
    public double JailbreakSensitivity { get; set; } = 0.7;

    /// <summary>
    /// Gets or sets whether to enable input validation.
    /// </summary>
    public bool EnableInputValidation { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to enable output filtering.
    /// </summary>
    public bool EnableOutputFiltering { get; set; } = true;

    /// <summary>
    /// Gets or sets the harmful content categories to check for.
    /// </summary>
    public string[] HarmfulContentCategories { get; set; } = new[]
    {
        "Violence",
        "HateSpeech",
        "AdultContent",
        "PrivateInformation",
        "Misinformation",
        "SelfHarm",
        "IllegalActivities"
    };

    /// <summary>
    /// Gets or sets whether to use a classifier for harmful content detection.
    /// </summary>
    public bool UseContentClassifier { get; set; } = true;

    /// <summary>
    /// Gets or sets the maximum input length to process.
    /// </summary>
    public int MaxInputLength { get; set; } = 10000;

    /// <summary>
    /// Gets or sets whether to log filtered content for review.
    /// </summary>
    public bool LogFilteredContent { get; set; } = true;

    /// <summary>
    /// Gets or sets the file path for filtered content logging.
    /// </summary>
    public string? SafetyLogFilePath { get; set; }

    // ========================================================================
    // ADVERSARIAL ATTACK OPTIONS (for robustness testing)
    // ========================================================================
    //
    // These settings are read by the robustness methods of the trained model's result:
    // AiModelResult.EvaluateRobustness(inputs, labels), EvaluateRobustnessWithAutoAttack,
    // GenerateAdversarialExample(input, label) and EvaluateConfiguredRobustness. Invalid values are
    // rejected there with an ArgumentOutOfRangeException or InvalidOperationException naming the setting.

    /// <summary>
    /// Gets or sets whether to enable adversarial robustness testing.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, <c>AiModelResult.EvaluateConfiguredRobustness</c> tests the
    /// model against every attack in <see cref="AttackMethods"/>; when disabled that method refuses to run.</para>
    /// </remarks>
    public bool EnableRobustnessTesting { get; set; } = false;

    /// <summary>
    /// Gets or sets the maximum perturbation budget (epsilon) for attacks.
    /// </summary>
    /// <value>Defaults to 0.1; must be positive. An epsilon passed to a robustness method overrides it.</value>
    public double AttackEpsilon { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the step size for iterative attacks.
    /// </summary>
    /// <value>Defaults to 0.01; must be positive. PGD steps by this much per iteration.</value>
    public double AttackStepSize { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the number of iterations for iterative attacks.
    /// </summary>
    /// <value>Defaults to 40; must be at least 1. PGD takes this many steps and Carlini-Wagner this many
    /// optimizer steps.</value>
    public int AttackIterations { get; set; } = 40;

    /// <summary>
    /// Gets or sets the norm type for perturbation constraints.
    /// </summary>
    /// <value>Defaults to <see cref="AdversarialNormType.LInfinity"/>. Selects the PGD step (gradient sign for
    /// L-infinity, normalized gradient for L2), its projection ball and its random start.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how the attack's budget is measured. L-infinity lets every feature
    /// move by up to epsilon; L2 limits the overall (Euclidean) size of the change to epsilon. It only describes
    /// what attacks may do; <see cref="CertificationNormType"/> separately sets the norm certificates cover.</para>
    /// </remarks>
    public AdversarialNormType AttackNormType { get; set; } = AdversarialNormType.LInfinity;

    /// <summary>
    /// Gets or sets whether to use targeted attacks.
    /// </summary>
    /// <remarks>
    /// When enabled the attacks push every input toward <see cref="TargetClass"/>, and the attack success rate
    /// counts the inputs whose adversarial prediction becomes that class.
    /// </remarks>
    public bool UseTargetedAttacks { get; set; } = false;

    /// <summary>
    /// Gets or sets the target class for targeted attacks.
    /// </summary>
    /// <value>Defaults to 0; with targeted attacks it must be a valid class index of the model output.</value>
    public int TargetClass { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to use random initialization for attacks.
    /// </summary>
    /// <value>Defaults to true: PGD starts from a uniformly random point of the epsilon ball.</value>
    public bool UseRandomStartForAttacks { get; set; } = true;

    /// <summary>
    /// Gets or sets which attack methods to use for testing.
    /// </summary>
    /// <value>Defaults to FGSM, PGD and Carlini-Wagner; must be non-empty and free of duplicates.</value>
    public AdversarialAttackMethod[] AttackMethods { get; set; } =
        new[] { AdversarialAttackMethod.Fgsm, AdversarialAttackMethod.Pgd, AdversarialAttackMethod.CarliniWagner };

    // ========================================================================
    // ADVERSARIAL DEFENSE OPTIONS
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to enable adversarial training.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Adversarial training makes the model more robust by training
    /// on both clean and adversarial examples.</para>
    /// </remarks>
    public bool EnableAdversarialTraining { get; set; } = false;

    /// <summary>
    /// Gets or sets the ratio of adversarial examples to include in training.
    /// </summary>
    public double AdversarialTrainingRatio { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the perturbation budget for adversarial training.
    /// </summary>
    public double DefenseEpsilon { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to use input preprocessing for defense.
    /// </summary>
    public bool UseInputPreprocessing { get; set; } = false;

    /// <summary>
    /// Gets or sets the preprocessing method to use.
    /// </summary>
    public string PreprocessingMethod { get; set; } = "JPEG";

    /// <summary>
    /// Gets or sets whether to use ensemble defenses.
    /// </summary>
    public bool UseEnsembleDefense { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of models in the ensemble.
    /// </summary>
    public int EnsembleSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets the attack method to use during adversarial training.
    /// </summary>
    public string TrainingAttackMethod { get; set; } = "PGD";

    // ========================================================================
    // CERTIFIED ROBUSTNESS OPTIONS
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to enable certified robustness.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Certified robustness provides provable guarantees that the model's
    /// prediction won't change within a certain perturbation radius.</para>
    /// </remarks>
    public bool EnableCertifiedRobustness { get; set; } = false;

    /// <summary>
    /// Gets or sets the certification method to use.
    /// </summary>
    /// <value>Defaults to <see cref="CertifiedDefenseMethod.RandomizedSmoothing"/>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> A certification method proves that no change to the input within a given
    /// size can flip the prediction. Randomized smoothing works with any model by voting over noisy copies of
    /// the input; interval bound propagation and CROWN read a neural network's layers and bound its outputs
    /// directly, which is exact about what it proves but only works for the layer types they model.</para>
    /// </remarks>
    public CertifiedDefenseMethod CertificationMethod { get; set; } = CertifiedDefenseMethod.RandomizedSmoothing;

    /// <summary>
    /// Gets or sets the number of samples for randomized smoothing.
    /// </summary>
    public int CertificationSamples { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the noise standard deviation for randomized smoothing.
    /// </summary>
    public double CertificationNoiseSigma { get; set; } = 0.25;

    /// <summary>
    /// Gets or sets the confidence level for certification.
    /// </summary>
    public double CertificationConfidence { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets whether to use tight bounds computation.
    /// </summary>
    public bool UseTightCertificationBounds { get; set; } = false;

    /// <summary>
    /// Gets or sets the norm type for certification.
    /// </summary>
    /// <value>Defaults to <see cref="AdversarialNormType.L2"/>, the norm randomized smoothing certifies.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how the size of an input change is measured when a prediction is
    /// certified, and it is separate from <see cref="AttackNormType"/>, which sets the budget attacks are given.
    /// L2 measures the overall (Euclidean) size of the change; L-infinity measures the largest change to any
    /// single feature. A certificate only covers changes measured in the norm it was computed for.</para>
    /// </remarks>
    public AdversarialNormType CertificationNormType { get; set; } = AdversarialNormType.L2;

    // ========================================================================
    // CONTENT MODERATION OPTIONS (for LLMs)
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to enable content moderation for LLM outputs.
    /// </summary>
    public bool EnableContentModeration { get; set; } = true;

    /// <summary>
    /// Gets or sets the prompt injection detection sensitivity.
    /// </summary>
    public double PromptInjectionSensitivity { get; set; } = 0.8;

    /// <summary>
    /// Gets or sets whether to detect and block prompt injection attacks.
    /// </summary>
    public bool BlockPromptInjections { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to filter PII (personally identifiable information).
    /// </summary>
    public bool FilterPII { get; set; } = true;

    /// <summary>
    /// Gets or sets the types of PII to filter.
    /// </summary>
    public string[] PIITypes { get; set; } = new[]
    {
        "Email",
        "PhoneNumber",
        "SocialSecurityNumber",
        "CreditCardNumber",
        "Address"
    };

    /// <summary>
    /// Gets or sets whether to enable factuality checking.
    /// </summary>
    public bool EnableFactualityChecking { get; set; } = false;

    /// <summary>
    /// Gets or sets the hallucination detection threshold.
    /// </summary>
    public double HallucinationThreshold { get; set; } = 0.7;

    // ========================================================================
    // RED TEAMING OPTIONS
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to enable red teaming during evaluation.
    /// </summary>
    public bool EnableRedTeaming { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of red teaming prompts to generate.
    /// </summary>
    public int RedTeamingPromptCount { get; set; } = 100;

    /// <summary>
    /// Gets or sets the red teaming categories to test.
    /// </summary>
    public string[] RedTeamingCategories { get; set; } = new[]
    {
        "Jailbreaking",
        "HarmfulContent",
        "Misinformation",
        "PrivacyViolation",
        "BiasExploration"
    };

    // ========================================================================
    // GENERAL OPTIONS
    // ========================================================================

    /// <summary>
    /// Gets or sets the batch size for robustness operations.
    /// </summary>
    public int BatchSize { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to enable verbose logging.
    /// </summary>
    public bool VerboseLogging { get; set; } = false;

    // ========================================================================
    // FACTORY METHODS FOR COMMON CONFIGURATIONS
    // ========================================================================

    /// <summary>
    /// Creates options for basic safety filtering only.
    /// </summary>
    public static AdversarialRobustnessOptions<T> BasicSafety()
    {
        return new AdversarialRobustnessOptions<T>
        {
            EnableSafetyFiltering = true,
            EnableRobustnessTesting = false,
            EnableAdversarialTraining = false,
            EnableCertifiedRobustness = false,
            EnableContentModeration = true
        };
    }

    /// <summary>
    /// Creates options for comprehensive robustness with certified guarantees.
    /// </summary>
    public static AdversarialRobustnessOptions<T> ComprehensiveRobustness()
    {
        return new AdversarialRobustnessOptions<T>
        {
            EnableSafetyFiltering = true,
            EnableRobustnessTesting = true,
            EnableAdversarialTraining = true,
            EnableCertifiedRobustness = true,
            EnableContentModeration = true,
            EnableRedTeaming = true
        };
    }

    /// <summary>
    /// Creates options for LLM safety with content moderation.
    /// </summary>
    public static AdversarialRobustnessOptions<T> LLMSafety()
    {
        return new AdversarialRobustnessOptions<T>
        {
            EnableSafetyFiltering = true,
            EnableContentModeration = true,
            BlockPromptInjections = true,
            FilterPII = true,
            EnableFactualityChecking = true,
            EnableRedTeaming = true,
            EnableRobustnessTesting = false,
            EnableAdversarialTraining = false,
            EnableCertifiedRobustness = false
        };
    }

    /// <summary>
    /// Creates options for adversarial training focus.
    /// </summary>
    public static AdversarialRobustnessOptions<T> AdversarialTrainingFocus()
    {
        return new AdversarialRobustnessOptions<T>
        {
            EnableSafetyFiltering = true,
            EnableAdversarialTraining = true,
            AdversarialTrainingRatio = 0.5,
            TrainingAttackMethod = "PGD",
            DefenseEpsilon = 0.1,
            EnableRobustnessTesting = true,
            AttackMethods = new[]
            {
                AdversarialAttackMethod.Fgsm,
                AdversarialAttackMethod.Pgd,
                AdversarialAttackMethod.CarliniWagner,
                AdversarialAttackMethod.AutoAttack
            }
        };
    }
}
