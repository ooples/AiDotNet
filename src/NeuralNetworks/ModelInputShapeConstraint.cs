namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Minimum geometry a generic caller must honor when constructing a model input probe.
/// </summary>
/// <param name="MinimumRank">Minimum tensor rank, including any batch axis.</param>
/// <param name="MinimumElementCount">Minimum total number of tensor elements.</param>
public readonly record struct ModelInputShapeConstraint(int MinimumRank, int MinimumElementCount)
{
    /// <summary>No additional constraint.</summary>
    public static ModelInputShapeConstraint None { get; } = new(0, 0);

    /// <summary>Whether this declaration adds any constraint.</summary>
    public bool IsConstrained => MinimumRank > 0 || MinimumElementCount > 0;
}
