namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for model interpretability and explainability features.
/// </summary>
/// <remarks>
/// <para>
/// This class configures which interpretability methods are available and how they behave.
/// It supports multiple model-agnostic explanation techniques that work with any model.
/// </para>
/// <para><b>For Beginners:</b> These options control how your model explains its predictions.
///
/// Why interpretability matters:
/// - Understanding why a model makes certain predictions builds trust
/// - Identifying which features drive predictions helps validate model logic
/// - Explaining individual predictions is often required for regulatory compliance
/// - Finding biases in feature importance can reveal unfair model behavior
///
/// Available explanation methods:
/// - <b>SHAP</b>: Shows how each feature contributed to each prediction (local and global)
/// - <b>LIME</b>: Explains individual predictions using simple approximations
/// - <b>Permutation Importance</b>: Shows which features matter most overall
/// - <b>Global Surrogate</b>: Trains a simple model to mimic the complex one
/// </para>
/// </remarks>
public class InterpretabilityOptions
{
    /// <summary>
    /// Gets or sets whether SHAP explanations are enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> SHAP (SHapley Additive exPlanations) comes from game theory
    /// and fairly distributes "credit" for a prediction among all features.
    ///
    /// - Positive SHAP values mean the feature pushed the prediction higher
    /// - Negative SHAP values mean the feature pushed the prediction lower
    /// - The magnitude shows how important that feature was
    ///
    /// SHAP values are considered the gold standard for feature attribution but can be
    /// computationally expensive for models with many features.
    /// </para>
    /// </remarks>
    public bool EnableSHAP { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of samples to use for Kernel SHAP approximation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More samples give more accurate SHAP values but take longer.
    /// Start with 100 and increase if you need more precision or see inconsistent results.</para>
    /// </remarks>
    public int SHAPSampleCount { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether LIME explanations are enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> LIME (Local Interpretable Model-agnostic Explanations)
    /// explains individual predictions by fitting a simple linear model around each prediction.
    ///
    /// It creates slightly modified versions of your input and sees how predictions change,
    /// then fits a simple linear model to understand the local behavior.
    ///
    /// Pros: Fast and intuitive
    /// Cons: Explanations can vary if you run it multiple times
    /// </para>
    /// </remarks>
    public bool EnableLIME { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of perturbed samples for LIME.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> LIME creates modified versions of your input to understand
    /// local behavior. More samples give more stable explanations but take longer.</para>
    /// </remarks>
    public int LIMESampleCount { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the kernel width for LIME (controls locality).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The kernel width controls how "local" the explanation is.
    /// Smaller values focus more on nearby samples; larger values consider more distant samples.
    /// If null, the explainer uses a sensible default based on the data.</para>
    /// </remarks>
    public double? LIMEKernelWidth { get; set; }

    /// <summary>
    /// Gets or sets whether Permutation Feature Importance is enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Permutation Feature Importance measures how important each
    /// feature is by randomly shuffling that feature's values and seeing how much worse the
    /// model performs.
    ///
    /// - If shuffling a feature hurts performance a lot, that feature is important
    /// - If shuffling doesn't change much, the feature isn't important
    ///
    /// This gives you a global view of which features matter most across all predictions.
    /// </para>
    /// </remarks>
    public bool EnablePermutationImportance { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of times to repeat permutation for more stable estimates.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Since shuffling is random, repeating it multiple times and
    /// averaging gives more stable importance estimates. More repeats = more stable but slower.</para>
    /// </remarks>
    public int PermutationRepeatCount { get; set; } = 5;

    /// <summary>
    /// Gets or sets whether Global Surrogate modeling is enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A global surrogate is a simple, interpretable model (like
    /// linear regression) that tries to mimic your complex model.
    ///
    /// If the surrogate has high fidelity (R² close to 1), you can use it to understand
    /// the complex model's overall behavior. The surrogate's coefficients tell you which
    /// features are most important and their direction of effect.
    ///
    /// Think of it like having a simple "translator" that explains what the complex model does.
    /// </para>
    /// </remarks>
    public bool EnableGlobalSurrogate { get; set; } = false;

    /// <summary>
    /// Gets or sets the random seed for reproducible explanations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Many explanation methods use randomness. Setting a seed
    /// ensures you get the same results each time you run the explanation, which is useful
    /// for testing and debugging.</para>
    /// </remarks>
    public int? RandomSeed { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of background samples for SHAP baseline calculation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> SHAP needs "background" data to compute expected predictions.
    /// Using too many background samples is slow; using too few can give unstable baselines.
    /// 100-500 samples usually works well.</para>
    /// </remarks>
    public int MaxBackgroundSamples { get; set; } = 100;

    /// <summary>
    /// Gets or sets optional feature names for more readable explanations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If you provide feature names, explanations will show
    /// "Age: +5000" instead of "Feature 3: +5000", making them much easier to understand.</para>
    /// </remarks>
    public string[]? FeatureNames { get; set; }
}
