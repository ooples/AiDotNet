namespace AiDotNet.Enums;

/// <summary>
/// Indicates the computational complexity and resource requirements of a machine learning model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tells you how much computing power and time a model needs.
/// If you're just getting started or have limited hardware, start with Low or Medium
/// complexity models. High and VeryHigh models may require GPUs and significant memory.
/// </para>
/// </remarks>
public enum ModelComplexity
{
    /// <summary>
    /// Minimal compute requirements. Runs quickly on any hardware including laptops.
    /// Examples: linear regression, naive Bayes, simple decision trees.
    /// Typical training time: seconds to minutes on small-to-medium datasets.
    /// </summary>
    Low,

    /// <summary>
    /// Moderate compute requirements. Runs on standard hardware but benefits from a GPU.
    /// Examples: random forests, gradient boosting, small neural networks.
    /// Typical training time: minutes to hours depending on dataset size.
    /// </summary>
    Medium,

    /// <summary>
    /// Significant compute requirements. GPU strongly recommended for reasonable training times.
    /// Examples: deep CNNs, RNNs, medium-sized transformers, most GANs.
    /// Typical training time: hours to days.
    /// </summary>
    High,

    /// <summary>
    /// Very high compute requirements. Multiple GPUs or specialized hardware typically needed.
    /// Examples: large language models, high-resolution diffusion models, large-scale RL agents.
    /// Typical training time: days to weeks.
    /// </summary>
    VeryHigh
}
