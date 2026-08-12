namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Minimum geometry a generic caller must honor when constructing a model input probe.
/// </summary>
/// <param name="MinimumRank">Minimum tensor rank, including any batch axis.</param>
/// <param name="MinimumElementCount">Minimum total number of tensor elements.</param>
/// <param name="ExactRank">Required tensor rank, or zero when rank is not exact.</param>
public readonly record struct ModelInputShapeConstraint(
    int MinimumRank,
    int MinimumElementCount,
    int ExactRank = 0)
{
    /// <summary>No additional constraint.</summary>
    public static ModelInputShapeConstraint None { get; } = new(0, 0);

    /// <summary>Whether this declaration adds any constraint.</summary>
    public bool IsConstrained => ExactRank > 0 || MinimumRank > 0 || MinimumElementCount > 0;
}
