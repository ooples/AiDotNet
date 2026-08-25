using AiDotNet.Enums;
using AiDotNet.Tensors;

namespace AiDotNet.Interfaces;

/// <summary>
/// Describes and evaluates the mathematical objective optimized by a model whose
/// training contract is not ordinary supervised prediction against a supplied target.
/// </summary>
/// <typeparam name="T">The model's numeric type.</typeparam>
/// <remarks>
/// The contract is deliberately model-agnostic. Test generation, trainers, and
/// diagnostics can ask the model for its real target and scalar objective without
/// model-name allowlists or a second catalogue of paper-specific loss formulas.
/// Implementations must be deterministic and must not update parameters.
/// </remarks>
internal interface ITrainingObjectiveProvider<T>
{
    /// <summary>Gets the kind of objective optimized by the model.</summary>
    TrainingObjectiveKind TrainingObjectiveKind { get; }

    /// <summary>
    /// Resolves the target consumed by the model's real training algorithm.
    /// </summary>
    /// <param name="input">The training input.</param>
    /// <param name="proposedTarget">
    /// A caller-supplied supervised target. Unsupervised learners may replace it with
    /// the input or another target derived from the input.
    /// </param>
    Tensor<T> ResolveTrainingTarget(Tensor<T> input, Tensor<T> proposedTarget);

    /// <summary>
    /// Evaluates the scalar objective whose improvement constitutes successful training.
    /// </summary>
    /// <param name="input">The objective input.</param>
    /// <param name="target">The target returned by <see cref="ResolveTrainingTarget"/>.</param>
    T EvaluateTrainingObjective(Tensor<T> input, Tensor<T> target);
}
