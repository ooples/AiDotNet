using AiDotNet.Finance.Evaluation;
using AiDotNet.Interfaces;
using AiDotNet.Models.Results;

namespace AiDotNet.CrossValidators;

/// <summary>
/// Walk-forward cross-validation that purges training samples whose label windows overlap the
/// validation fold, and embargoes a gap after it.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
/// <remarks>
/// <para>
/// Every other cross-validator here — including <see cref="TimeSeriesCrossValidator{T, TInput, TOutput}"/>,
/// which is rolling-origin — leaks when labels look forward. A 20-bar forward return computed at the
/// last training index is derived from bars inside the validation fold, so the model is scored partly
/// on data it effectively saw, and the resulting estimate is optimistic. Purging removes those
/// overlapping training samples; the embargo additionally drops a gap after the fold to cover serial
/// correlation outliving the label window.
/// </para>
/// <para>
/// The fold geometry is <see cref="PurgedWalkForwardValidator"/>'s — the same logic behind
/// <c>PurgedWalkForwardSplitter</c>, so a purged split and a purged cross-validation agree rather
/// than being two separate implementations that can drift apart.
/// </para>
/// <para>
/// <b>For Beginners:</b> When the answer for "today" depends on the next 20 days, the rows just
/// before your validation period secretly contain validation-period information. Scoring against
/// them flatters the model. This removes them, so the number you get is one you can trust.
/// </para>
/// </remarks>
public class PurgedWalkForwardCrossValidator<T, TInput, TOutput> : CrossValidatorBase<T, TInput, TOutput>
{
    private readonly int _labelHorizon;
    private readonly int _nSplits;
    private readonly int _embargo;
    private readonly bool _expanding;

    /// <summary>
    /// Creates a purged, embargoed walk-forward cross-validator.
    /// </summary>
    /// <param name="labelHorizon">
    /// How many samples forward each label looks. Training samples whose window reaches into the
    /// validation fold are purged. Must be at least 1.
    /// </param>
    /// <param name="nSplits">Number of walk-forward folds. Must be at least 1.</param>
    /// <param name="embargo">Samples dropped after the validation fold. Must not be negative.</param>
    /// <param name="expanding">
    /// <c>true</c> to train on everything up to the fold; <c>false</c> for a fixed rolling window.
    /// </param>
    /// <param name="options">Cross-validation options.</param>
    public PurgedWalkForwardCrossValidator(
        int labelHorizon,
        int nSplits = 5,
        int embargo = 0,
        bool expanding = true,
        CrossValidationOptions? options = null)
        : base(options ?? new CrossValidationOptions())
    {
        if (labelHorizon < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(labelHorizon), labelHorizon,
                "Label horizon must be at least 1; it is how far each label looks forward, which is " +
                "what determines the training samples that must be purged.");
        }

        if (nSplits < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nSplits), nSplits, "Splits must be at least 1.");
        }

        if (embargo < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(embargo), embargo, "Embargo cannot be negative.");
        }

        _labelHorizon = labelHorizon;
        _nSplits = nSplits;
        _embargo = embargo;
        _expanding = expanding;
    }

    /// <inheritdoc />
    public override CrossValidationResult<T, TInput, TOutput> Validate(
        IFullModel<T, TInput, TOutput> model, TInput X, TOutput y,
        IOptimizer<T, TInput, TOutput> optimizer)
        => PerformCrossValidation(model, X, y, CreateFolds(X), optimizer);

    /// <summary>
    /// Builds purged, embargoed walk-forward folds, skipping any the geometry leaves degenerate.
    /// </summary>
    /// <remarks>
    /// Purging deliberately deletes training rows, so a fold can legitimately end up with none —
    /// most often the first fold of an expanding window, where little history precedes the
    /// validation block and the label horizon consumes what there is. Such a fold cannot be trained
    /// or scored, so it is skipped rather than passed on to fail deep in the fold loop with
    /// "Indices array cannot be empty".
    /// </remarks>
    private IEnumerable<(int[] trainIndices, int[] validationIndices)> CreateFolds(TInput X)
    {
        int nSamples = InputHelper<T, TInput>.GetInputSize(X);
        int usable = 0;

        foreach (var fold in PurgedWalkForwardValidator.Split(
            nSamples: nSamples,
            labelHorizon: _labelHorizon,
            nSplits: _nSplits,
            embargo: _embargo,
            expanding: _expanding))
        {
            if (fold.TrainIndices.Count == 0 || fold.TestIndices.Count == 0)
            {
                continue;
            }

            usable++;
            yield return (fold.TrainIndices.ToArray(), fold.TestIndices.ToArray());
        }

        if (usable == 0)
        {
            throw new InvalidOperationException(
                $"Purged walk-forward produced no usable folds from {nSamples} samples " +
                $"(label horizon {_labelHorizon}, {_nSplits} folds, embargo {_embargo}). Purging and " +
                "the embargo consume samples around every fold boundary, so short series need a " +
                "smaller label horizon, fewer folds, or a smaller embargo — silently validating on " +
                "nothing would be worse than saying so.");
        }
    }
}
