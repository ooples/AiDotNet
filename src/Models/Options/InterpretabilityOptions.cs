namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for model interpretability and explainability features.
/// </summary>
/// <remarks>
/// <para>
/// This class configures which interpretability methods are available and how they behave.
/// It supports multiple model-agnostic explanation techniques that work with any model,
/// as well as specialized neural network explanation methods.
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
///
/// <b>Model-Agnostic Methods</b> (work with any model):
/// - <b>SHAP</b>: Shows how each feature contributed to each prediction (local and global)
/// - <b>LIME</b>: Explains individual predictions using simple approximations
/// - <b>Permutation Importance</b>: Shows which features matter most overall
/// - <b>Partial Dependence</b>: Shows how features affect predictions on average
/// - <b>Feature Interaction</b>: Measures how features interact using H-statistic
/// - <b>Counterfactual</b>: Shows what would need to change for a different prediction
/// - <b>Anchor</b>: Creates rule-based explanations
/// - <b>Global Surrogate</b>: Trains a simple model to mimic the complex one
///
/// <b>Neural Network Methods</b> (require gradient access):
/// - <b>Integrated Gradients</b>: Theoretically-grounded attribution method
/// - <b>DeepLIFT</b>: Fast attribution comparing to baseline
/// - <b>GradCAM</b>: Visual heatmaps for CNN explanations
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
    /// Gets or sets whether Partial Dependence Plots (PDP) are enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Partial dependence plots show how a feature affects predictions
    /// on average, while holding all other features constant.
    ///
    /// - Upward slope: increasing the feature increases predictions
    /// - Downward slope: increasing the feature decreases predictions
    /// - Flat line: feature has little average effect
    ///
    /// Also computes Individual Conditional Expectation (ICE) curves which show
    /// the same for individual instances, revealing heterogeneous effects.
    /// </para>
    /// </remarks>
    public bool EnablePartialDependence { get; set; } = false;

    /// <summary>
    /// Gets or sets whether Feature Interaction analysis (H-statistic) is enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Feature interaction analysis measures how much features
    /// interact with each other using Friedman's H-statistic.
    ///
    /// An interaction means the effect of one feature depends on another feature's value.
    /// For example: "Education increases salary, but MORE so for people with more Experience."
    ///
    /// H-statistic values:
    /// - H = 0: No interaction (features act independently)
    /// - H = 1: Pure interaction
    /// - H &lt; 0.05: Negligible interaction
    /// - H &gt; 0.5: Strong interaction
    /// </para>
    /// </remarks>
    public bool EnableFeatureInteraction { get; set; } = false;

    /// <summary>
    /// Gets or sets whether Counterfactual explanations are enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Counterfactual explanations answer "What would need to change
    /// to get a different prediction?"
    ///
    /// Example: "If your income was $5,000 higher, you would qualify for the loan."
    ///
    /// These are very intuitive for end users and useful in high-stakes decisions
    /// where people need actionable feedback.
    /// </para>
    /// </remarks>
    public bool EnableCounterfactual { get; set; } = false;

    /// <summary>
    /// Gets or sets whether Anchor explanations are enabled.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Anchor explanations find sufficient conditions (rules)
    /// that "anchor" the prediction with high precision.
    ///
    /// Example: "If age > 40 AND income > $50k, then approved (with 95% precision)."
    ///
    /// These create human-readable rules that explain predictions in terms users can
    /// easily understand and verify.
    /// </para>
    /// </remarks>
    public bool EnableAnchor { get; set; } = false;

    /// <summary>
    /// Gets or sets whether Integrated Gradients is enabled for neural network explanation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Integrated Gradients is a theoretically-grounded method
    /// for explaining neural network predictions. It satisfies important axioms:
    ///
    /// 1. <b>Completeness</b>: Attributions sum to (prediction - baseline_prediction)
    /// 2. <b>Sensitivity</b>: Important features get non-zero attributions
    ///
    /// It works by integrating gradients along a path from a baseline (usually zeros)
    /// to your actual input. More accurate than vanilla gradients but requires more computation.
    ///
    /// <b>Note:</b> Only works with neural network models.
    /// </para>
    /// </remarks>
    public bool EnableIntegratedGradients { get; set; } = false;

    /// <summary>
    /// Gets or sets the number of integration steps for Integrated Gradients.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More steps give more accurate attributions but take longer.
    /// 50 is usually sufficient; use 200-300 for high precision.</para>
    /// </remarks>
    public int IntegratedGradientsSteps { get; set; } = 50;

    /// <summary>
    /// Gets or sets whether DeepLIFT is enabled for neural network explanation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DeepLIFT (Deep Learning Important FeaTures) explains
    /// neural network predictions by comparing activations to a reference baseline.
    ///
    /// It's faster than Integrated Gradients and handles non-linearities (like ReLU)
    /// better than vanilla gradients.
    ///
    /// Two rules are available:
    /// - <b>Rescale</b>: Simpler, works well in most cases
    /// - <b>RevealCancel</b>: Better when you need to separate positive/negative contributions
    ///
    /// <b>Note:</b> Only works with neural network models.
    /// </para>
    /// </remarks>
    public bool EnableDeepLIFT { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use the RevealCancel rule for DeepLIFT (default: Rescale).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The Rescale rule is simpler and faster. The RevealCancel
    /// rule is more accurate when features have both positive and negative contributions
    /// that partially cancel each other out.</para>
    /// </remarks>
    public bool DeepLIFTUseRevealCancel { get; set; } = false;

    /// <summary>
    /// Gets or sets whether GradCAM is enabled for CNN visual explanations.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Grad-CAM (Gradient-weighted Class Activation Mapping)
    /// creates visual heatmaps showing which parts of an image were most important
    /// for a CNN's prediction.
    ///
    /// Bright regions in the heatmap indicate high importance. This is especially
    /// useful for debugging image classifiers and building trust in visual AI systems.
    ///
    /// <b>Note:</b> Only works with convolutional neural network models for images.
    /// </para>
    /// </remarks>
    public bool EnableGradCAM { get; set; } = false;

    /// <summary>
    /// Gets or sets whether to use GradCAM++ instead of standard GradCAM.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> GradCAM++ is an improved version that handles multiple
    /// instances of the same object in an image better. Use this when your images might
    /// contain multiple occurrences of the target class.</para>
    /// </remarks>
    public bool UseGradCAMPlusPlus { get; set; } = false;

    /// <summary>
    /// Gets or sets whether TreeSHAP is enabled for tree-based model explanation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> TreeSHAP computes exact (not approximate) SHAP values
    /// specifically for tree-based models like Decision Trees, Random Forests, and
    /// Gradient Boosting.
    ///
    /// Unlike Kernel SHAP which samples and approximates, TreeSHAP uses the tree structure
    /// to compute mathematically exact Shapley values. This makes it:
    /// - <b>Faster</b>: O(TLD²) complexity vs exponential for exact Kernel SHAP
    /// - <b>Exact</b>: No approximation, precise results every time
    /// - <b>No background data needed</b>: Uses tree structure itself
    ///
    /// <b>Note:</b> Only works with tree-based models (Decision Tree, Random Forest, Gradient Boosting).
    /// </para>
    /// </remarks>
    public bool EnableTreeSHAP { get; set; } = false;

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
