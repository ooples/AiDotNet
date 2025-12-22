using AiDotNet.AdversarialRobustness.Safety;
using AiDotNet.Interfaces;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration for adversarial robustness and AI safety during model building and inference.
/// </summary>
/// <remarks>
/// <para>
/// This configuration controls all aspects of adversarial robustness and AI safety, replacing
/// the previous SafetyFilterConfiguration with a unified approach that includes:
/// - Safety filtering (input/output validation)
/// - Adversarial attacks and defenses
/// - Certified robustness
/// - Content moderation
/// - Red teaming
/// </para>
/// <para><b>For Beginners:</b> This is your complete safety and robustness configuration.
/// You can enable/disable features, customize options, or provide your own implementations.
/// All settings have sensible defaults based on industry best practices.</para>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
public class AdversarialRobustnessConfiguration<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets whether adversarial robustness features are enabled.
    /// </summary>
    /// <remarks>
    /// This is the master switch. When false, all robustness features are skipped.
    /// </remarks>
    public bool Enabled { get; set; } = true;

    /// <summary>
    /// Gets or sets the robustness options.
    /// </summary>
    public AdversarialRobustnessOptions<T> Options { get; set; } = new();

    // ========================================================================
    // CUSTOM IMPLEMENTATIONS (Optional)
    // ========================================================================

    /// <summary>
    /// Gets or sets a custom safety filter implementation.
    /// </summary>
    /// <remarks>
    /// When provided, this filter is used instead of the default implementation.
    /// </remarks>
    public ISafetyFilter<T>? CustomSafetyFilter { get; set; }

    /// <summary>
    /// Gets or sets custom adversarial attack implementations.
    /// </summary>
    /// <remarks>
    /// Additional attacks beyond the built-in FGSM, PGD, CW, and AutoAttack.
    /// These attacks can work with any input/output types supported by the model.
    /// </remarks>
    public IAdversarialAttack<T, TInput, TOutput>[]? CustomAttacks { get; set; }

    /// <summary>
    /// Gets or sets a custom adversarial defense implementation.
    /// </summary>
    /// <remarks>
    /// Custom defense mechanism that works with the model's input/output types.
    /// </remarks>
    public IAdversarialDefense<T, TInput, TOutput>? CustomDefense { get; set; }

    /// <summary>
    /// Gets or sets a custom certified defense implementation.
    /// </summary>
    /// <remarks>
    /// Custom certified defense that provides provable robustness guarantees.
    /// </remarks>
    public ICertifiedDefense<T, TInput, TOutput>? CustomCertifiedDefense { get; set; }

    /// <summary>
    /// Gets or sets a custom content classifier implementation.
    /// </summary>
    public IContentClassifier<T>? CustomContentClassifier { get; set; }

    // ========================================================================
    // INFERENCE-TIME OPTIONS
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to apply certified inference during prediction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled, predictions include certified robustness guarantees.
    /// This adds computational overhead but provides provable guarantees.
    /// </para>
    /// </remarks>
    public bool UseCertifiedInference { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to include robustness metrics in prediction results.
    /// </summary>
    public bool IncludeRobustnessMetrics { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum certified radius required for a prediction to be considered robust.
    /// </summary>
    public double MinimumCertifiedRadius { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets whether to reject predictions that don't meet the minimum certified radius.
    /// </summary>
    public bool RejectNonRobustPredictions { get; set; } = false;

    // ========================================================================
    // EVALUATION OPTIONS
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to include robustness evaluation in model evaluation.
    /// </summary>
    public bool IncludeRobustnessInEvaluation { get; set; } = true;

    /// <summary>
    /// Gets or sets the epsilon values to test during robustness evaluation.
    /// </summary>
    public double[] EvaluationEpsilons { get; set; } = new[] { 0.01, 0.05, 0.1, 0.2, 0.3 };

    /// <summary>
    /// Gets or sets the percentage of test data to use for robustness evaluation.
    /// </summary>
    /// <remarks>
    /// Robustness evaluation can be slow, so this allows testing on a subset.
    /// </remarks>
    public double RobustnessEvaluationSampleRatio { get; set; } = 0.1;

    // ========================================================================
    // MODEL CARD INTEGRATION
    // ========================================================================

    /// <summary>
    /// Gets or sets whether to auto-generate robustness sections in the model card.
    /// </summary>
    public bool AutoGenerateModelCardSections { get; set; } = true;

    /// <summary>
    /// Gets or sets custom model card robustness notes.
    /// </summary>
    public string? ModelCardRobustnessNotes { get; set; }

    // ========================================================================
    // FACTORY METHODS
    // ========================================================================

    /// <summary>
    /// Creates a disabled configuration (no robustness features).
    /// </summary>
    public static AdversarialRobustnessConfiguration<T, TInput, TOutput> Disabled()
    {
        return new AdversarialRobustnessConfiguration<T, TInput, TOutput>
        {
            Enabled = false
        };
    }

    /// <summary>
    /// Creates a configuration with basic safety filtering only.
    /// </summary>
    public static AdversarialRobustnessConfiguration<T, TInput, TOutput> BasicSafety()
    {
        return new AdversarialRobustnessConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = AdversarialRobustnessOptions<T>.BasicSafety(),
            UseCertifiedInference = false,
            IncludeRobustnessInEvaluation = false
        };
    }

    /// <summary>
    /// Creates a configuration with comprehensive robustness features.
    /// </summary>
    public static AdversarialRobustnessConfiguration<T, TInput, TOutput> Comprehensive()
    {
        return new AdversarialRobustnessConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = AdversarialRobustnessOptions<T>.ComprehensiveRobustness(),
            UseCertifiedInference = true,
            IncludeRobustnessInEvaluation = true,
            AutoGenerateModelCardSections = true
        };
    }

    /// <summary>
    /// Creates a configuration optimized for LLM safety.
    /// </summary>
    public static AdversarialRobustnessConfiguration<T, TInput, TOutput> ForLLM()
    {
        return new AdversarialRobustnessConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = AdversarialRobustnessOptions<T>.LLMSafety(),
            UseCertifiedInference = false,
            IncludeRobustnessInEvaluation = true,
            AutoGenerateModelCardSections = true
        };
    }

    /// <summary>
    /// Creates a configuration focused on adversarial training.
    /// </summary>
    public static AdversarialRobustnessConfiguration<T, TInput, TOutput> WithAdversarialTraining()
    {
        return new AdversarialRobustnessConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = AdversarialRobustnessOptions<T>.AdversarialTrainingFocus(),
            UseCertifiedInference = false,
            IncludeRobustnessInEvaluation = true
        };
    }

    /// <summary>
    /// Creates a configuration with certified robustness guarantees.
    /// </summary>
    /// <param name="certificationMethod">The certification method: "RandomizedSmoothing", "IBP", or "CROWN".</param>
    public static AdversarialRobustnessConfiguration<T, TInput, TOutput> WithCertification(
        string certificationMethod = "RandomizedSmoothing")
    {
        return new AdversarialRobustnessConfiguration<T, TInput, TOutput>
        {
            Enabled = true,
            Options = new AdversarialRobustnessOptions<T>
            {
                EnableSafetyFiltering = true,
                EnableCertifiedRobustness = true,
                CertificationMethod = certificationMethod
            },
            UseCertifiedInference = true,
            IncludeRobustnessInEvaluation = true,
            MinimumCertifiedRadius = 0.01
        };
    }
}

/// <summary>
/// Non-generic version for backward compatibility and simpler use cases.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
public class AdversarialRobustnessConfiguration<T> : AdversarialRobustnessConfiguration<T, object, object>
{
}
