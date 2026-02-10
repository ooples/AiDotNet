using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Extended interface for meta-learning adapted models that carry task-specific adaptation state
/// beyond backbone parameters. Enables downstream code to access algorithm-specific adapted
/// representations for evaluation, analysis, or custom classification.
/// </summary>
/// <typeparam name="T">The numeric data type.</typeparam>
/// <remarks>
/// <para>
/// Meta-learning algorithms like EPNet, MCL, SetFeat, and ConstellationNet compute adapted
/// representations during task adaptation (e.g., propagated features, projected features,
/// set-encoded class representations). This interface exposes those representations so that
/// evaluation code, visualization tools, or custom classifiers can use them.
/// </para>
/// <para><b>For Beginners:</b> When a meta-learner adapts to a new task, it often computes
/// enriched feature representations beyond just the backbone model's output. This interface
/// lets you access those enriched features for custom use cases, like building a nearest-neighbor
/// classifier on the adapted features instead of relying solely on the backbone's predictions.
/// </para>
/// </remarks>
public interface IAdaptedMetaModel<T>
{
    /// <summary>
    /// Gets the adapted support features computed during task adaptation.
    /// These are algorithm-specific enriched representations (e.g., propagated, projected,
    /// set-encoded) that capture task-relevant information beyond raw backbone features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are the support examples' features AFTER the
    /// algorithm's special processing (like graph propagation in EPNet, or contrastive
    /// projection in MCL). They're typically more useful for classification than raw features.
    /// </para>
    /// </remarks>
    Vector<T>? AdaptedSupportFeatures { get; }

    /// <summary>
    /// Gets the parameter modulation factors computed during adaptation.
    /// When non-null, these factors are applied to the backbone parameters during Predict()
    /// to produce task-adapted predictions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Some algorithms (like NPBML) compute a set of numbers that
    /// scale the model's internal weights for each specific task. This property lets you
    /// inspect those scaling factors.
    /// </para>
    /// </remarks>
    double[]? ParameterModulationFactors { get; }
}
