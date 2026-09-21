namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Minimum geometry a generic caller must honor when constructing a model input probe.
/// </summary>
/// <param name="MinimumRank">Minimum tensor rank, including any batch axis.</param>
/// <param name="MinimumElementCount">Minimum total number of tensor elements.</param>
/// <param name="ExactRank">Required tensor rank, or zero when rank is not exact.</param>
/// <param name="ExactAxisSizes">
/// Per-axis sizes the model accepts and no other, or zero on an axis it leaves free. A minimum is
/// the wrong vocabulary for a structural extent such as a lookback window or an asset count: the
/// model rejects a longer axis exactly as it rejects a shorter one, so a generic caller that only
/// raises the axis to a floor still builds an input the model refuses.
/// </param>
public readonly record struct ModelInputShapeConstraint(
    int MinimumRank,
    int MinimumElementCount,
    int ExactRank = 0,
    int MaximumRank = 0,
    IReadOnlyList<int>? MinimumAxisSizes = null,
    IReadOnlyList<int>? AxisDivisors = null,
    IReadOnlyList<int>? ExactAxisSizes = null)
{
    /// <summary>No additional constraint.</summary>
    public static ModelInputShapeConstraint None { get; } = new(0, 0);

    /// <summary>Whether this declaration adds any constraint.</summary>
    public bool IsConstrained => ExactRank > 0 || MinimumRank > 0 || MaximumRank > 0
        || MinimumElementCount > 0
        || MinimumAxisSizes?.Any(value => value > 0) == true
        || AxisDivisors?.Any(value => value > 1) == true
        || ExactAxisSizes?.Any(value => value > 0) == true;
}
