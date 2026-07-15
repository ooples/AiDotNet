using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Finance.Evaluation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.DataPreparation.Splitting.TimeSeries;

/// <summary>
/// Walk-forward splitter that purges training samples whose label windows overlap the test fold, and
/// embargoes a gap after it.
/// </summary>
/// <remarks>
/// <para>
/// The fold geometry is <see cref="PurgedWalkForwardValidator"/>'s, which is the only implementation
/// in the library carrying both <c>labelHorizon</c> and <c>embargo</c> — but it is a static class, so
/// it could not be handed to <c>ConfigureDataSplitter</c> and was unreachable from the facade. This
/// wraps it as an <see cref="IDataSplitter{T}"/> so the same logic is configurable like every other
/// splitter, rather than remaining a parallel API the builder cannot use.
/// </para>
/// <para>
/// <b>Why purge and embargo matter.</b> A plain chronological split leaks whenever a label spans the
/// boundary: a 20-bar forward return computed at the last training index is derived from bars inside
/// the test fold, so the model is scored partly on data it effectively saw. Purging drops those
/// overlapping training samples; the embargo additionally drops a gap after the fold, covering serial
/// correlation that outlives the label window. Standard practice in financial ML, and absent from
/// mainstream ML libraries.
/// </para>
/// <para>
/// <b>For Beginners:</b> When your answer for "today" depends on the next 20 days, the rows right
/// before your test period secretly contain test-period information. This removes those rows so the
/// score you get is honest.
/// </para>
/// </remarks>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class PurgedWalkForwardSplitter<T> : DataSplitterBase<T>
{
    private readonly int _labelHorizon;
    private readonly int _nSplits;
    private readonly int _embargo;
    private readonly bool _expanding;

    /// <summary>
    /// Creates a purged, embargoed walk-forward splitter.
    /// </summary>
    /// <param name="labelHorizon">
    /// How many samples forward each label looks. Training samples whose window reaches into the
    /// test fold are purged. Must be at least 1.
    /// </param>
    /// <param name="nSplits">Number of walk-forward folds. Must be at least 1.</param>
    /// <param name="embargo">Samples dropped after the test fold. Must not be negative.</param>
    /// <param name="expanding">
    /// <c>true</c> for an expanding window (train on everything up to the fold); <c>false</c> for a
    /// rolling window of fixed size.
    /// </param>
    public PurgedWalkForwardSplitter(int labelHorizon, int nSplits = 5, int embargo = 0, bool expanding = true)
        // Chronological by construction: shuffling would destroy the ordering purge/embargo depend on.
        : base(shuffle: false)
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
    public override int NumSplits => _nSplits;

    /// <inheritdoc />
    public override bool RequiresLabels => false;

    /// <inheritdoc />
    /// <remarks>
    /// The walk-forward geometry yields train and test only. A validation partition is produced by
    /// the caller re-applying this splitter to the training partition when one is needed, which
    /// keeps the purge and embargo in force at both levels.
    /// </remarks>
    public override bool SupportsValidation => false;

    /// <inheritdoc />
    public override string Description =>
        $"Purged walk-forward ({_nSplits} folds, label horizon {_labelHorizon}, embargo {_embargo}, " +
        $"{(_expanding ? "expanding" : "rolling")})";

    /// <inheritdoc />
    public override DataSplitResult<T> Split(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);
        return GetSplits(X, y).First();
    }

    /// <inheritdoc />
    public override IEnumerable<DataSplitResult<T>> GetSplits(Matrix<T> X, Vector<T>? y = null)
    {
        ValidateInputs(X, y);

        var folds = PurgedWalkForwardValidator.Split(
            nSamples: X.Rows,
            labelHorizon: _labelHorizon,
            nSplits: _nSplits,
            embargo: _embargo,
            expanding: _expanding);

        int foldIndex = 0;
        foreach (var fold in folds)
        {
            yield return BuildResult(
                X, y,
                trainIndices: fold.TrainIndices.ToArray(),
                testIndices: fold.TestIndices.ToArray(),
                foldIndex: foldIndex++,
                totalFolds: folds.Count);
        }
    }
}
