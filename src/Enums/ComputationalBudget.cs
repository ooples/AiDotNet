namespace AiDotNet.Enums;

/// <summary>
/// Describes the computational budget available for model training and search.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tells AutoML how much computing time and resources you're willing to spend.
/// Lower budgets prefer simpler, faster models. Higher budgets allow expensive ensembles and deep learning.
/// </para>
/// </remarks>
public enum ComputationalBudget
{
    /// <summary>
    /// Low budget — prefer fast, simple models (linear, small trees).
    /// </summary>
    Low,

    /// <summary>
    /// Moderate budget — typical training resources (ensembles, medium networks).
    /// </summary>
    Moderate,

    /// <summary>
    /// High budget — can afford expensive models and extensive hyperparameter search.
    /// </summary>
    High
}
