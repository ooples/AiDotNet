namespace AiDotNet.Data.Structures;

/// <summary>
/// Update rules for meta-optimization in Meta-SGD.
/// </summary>
/// <remarks>
/// <para>
/// Update rules determine how the meta-learner updates its parameters
/// based on the meta-gradients computed across tasks. Different rules
/// offer different trade-offs between stability and adaptability.
/// </para>
/// </remarks>
public enum UpdateRule
{
    /// <summary>
    /// Standard gradient descent update.
    /// </summary>
    SGD,

    /// <summary>
    /// Adam optimizer update.
    /// </summary>
    Adam,

    /// <summary>
    /// RMSprop update.
    /// </summary>
    RMSprop,

    /// <summary>
    /// Adagrad update.
    /// </summary>
    Adagrad
}