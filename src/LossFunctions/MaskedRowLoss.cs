using AiDotNet.Helpers;

namespace AiDotNet.LossFunctions;

/// <summary>
/// Restricts another loss to a subset of the rows of a <c>[rows, features]</c> prediction, so the
/// excluded rows contribute neither loss nor gradient.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This exists for TRANSDUCTIVE, semi-supervised training -- the setting graph networks train in,
/// where the forward pass must run over every node because message passing needs the whole graph,
/// but only the training nodes are allowed to supply a learning signal.
/// </para>
/// <para>
/// ZEROING THE LABEL ROW OF A HELD-OUT NODE DOES NOT HOLD IT OUT, and that is the mistake this type
/// is here to prevent. For softmax cross-entropy the gradient of an all-zero target row is
/// <c>softmax(logits) - 0 = softmax(logits)</c>, which is not zero: it pushes every logit of every
/// held-out node downward on every step. That is strictly worse than no signal, because it is a
/// consistent wrong signal rather than an absent one. Selecting the rows is what actually excludes
/// them -- an unselected row is not part of the graph the tape differentiates, so its gradient is
/// exactly zero rather than approximately so.
/// </para>
/// <para><b>For Beginners:</b> In semi-supervised learning you have labels for only some of your
/// data points, but you still have to run the model over all of them. This wrapper tells the loss
/// "score only these rows" so the unlabeled ones cannot accidentally teach the model anything.</para>
/// </remarks>
public sealed class MaskedRowLoss<T> : LossFunctionBase<T>
{
    private readonly LossFunctionBase<T> _inner;
    private readonly int[] _selectedRows;
    private readonly int _totalRows;

    /// <summary>
    /// Wraps <paramref name="inner"/> so it scores only the rows where <paramref name="mask"/> is true.
    /// </summary>
    /// <param name="inner">The loss to apply to the selected rows.</param>
    /// <param name="mask">One entry per row; true means the row contributes to the loss.</param>
    /// <exception cref="ArgumentException">No row is selected, so there would be nothing to learn from.</exception>
    public MaskedRowLoss(LossFunctionBase<T> inner, bool[] mask)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        if (mask is null) throw new ArgumentNullException(nameof(mask));

        var selected = new List<int>(mask.Length);
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i]) selected.Add(i);
        }

        if (selected.Count == 0)
        {
            throw new ArgumentException(
                "The mask selects no rows, so training would have no supervised signal at all.", nameof(mask));
        }

        _selectedRows = selected.ToArray();
        _totalRows = mask.Length;
    }

    /// <summary>Gets the number of rows that contribute to the loss.</summary>
    public int SelectedRowCount => _selectedRows.Length;

    /// <inheritdoc/>
    public override Tensor<T> ComputeTapeLoss(Tensor<T> predicted, Tensor<T> target)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (target is null) throw new ArgumentNullException(nameof(target));

        // A rank-1 prediction has no row axis to select along, and a row count that disagrees with the
        // mask means the mask describes different data than the model just produced. Both would
        // otherwise degrade into a silently-wrong subset rather than an error.
        RequireSelectableRows(predicted.Rank < 1 ? 0 : predicted.Shape[0], nameof(predicted));
        RequireSelectableRows(target.Rank < 1 ? 0 : target.Shape[0], nameof(target));

        var indices = BuildIndexTensor();
        var selectedPredicted = Engine.TensorIndexSelect(predicted, indices, 0);
        var selectedTarget = Engine.TensorIndexSelect(target, indices, 0);

        // The inner loss averages over the rows it is given, which are exactly the selected ones, so
        // the result is already the mean over training rows -- no rescaling needed here.
        return _inner.ComputeTapeLoss(selectedPredicted, selectedTarget);
    }

    /// <inheritdoc/>
    public override T CalculateLoss(Vector<T> predicted, Vector<T> actual)
    {
        var (p, a) = SelectRows(predicted, actual);
        return _inner.CalculateLoss(p, a);
    }

    /// <inheritdoc/>
    public override Vector<T> CalculateDerivative(Vector<T> predicted, Vector<T> actual)
    {
        var (p, a) = SelectRows(predicted, actual);
        var innerGradient = _inner.CalculateDerivative(p, a);

        // Scatter back into the full-width gradient. Unselected rows stay zero, which is the whole
        // point: they were never part of the loss, so they get no gradient.
        int valuesPerRow = predicted.Length / _totalRows;
        var gradient = new Vector<T>(predicted.Length);
        for (int i = 0; i < _selectedRows.Length; i++)
        {
            int sourceBase = i * valuesPerRow;
            int targetBase = _selectedRows[i] * valuesPerRow;
            for (int j = 0; j < valuesPerRow; j++)
            {
                gradient[targetBase + j] = innerGradient[sourceBase + j];
            }
        }

        return gradient;
    }

    private void RequireSelectableRows(int rows, string parameterName)
    {
        if (rows != _totalRows)
        {
            throw new ArgumentException(
                $"The mask describes {_totalRows} rows, but {parameterName} has {rows}.", parameterName);
        }
    }

    private Tensor<int> BuildIndexTensor()
    {
        var indices = new Tensor<int>([_selectedRows.Length]);
        for (int i = 0; i < _selectedRows.Length; i++)
        {
            indices[i] = _selectedRows[i];
        }

        return indices;
    }

    private (Vector<T> Predicted, Vector<T> Actual) SelectRows(Vector<T> predicted, Vector<T> actual)
    {
        if (predicted is null) throw new ArgumentNullException(nameof(predicted));
        if (actual is null) throw new ArgumentNullException(nameof(actual));
        if (predicted.Length != actual.Length)
        {
            throw new ArgumentException(
                $"Predicted has {predicted.Length} values but actual has {actual.Length}.", nameof(actual));
        }

        if (predicted.Length % _totalRows != 0)
        {
            throw new ArgumentException(
                $"A flat length of {predicted.Length} does not divide into the {_totalRows} rows the mask " +
                "describes, so the row boundaries cannot be recovered.", nameof(predicted));
        }

        int valuesPerRow = predicted.Length / _totalRows;
        var selectedPredicted = new Vector<T>(_selectedRows.Length * valuesPerRow);
        var selectedActual = new Vector<T>(_selectedRows.Length * valuesPerRow);
        for (int i = 0; i < _selectedRows.Length; i++)
        {
            int sourceBase = _selectedRows[i] * valuesPerRow;
            int targetBase = i * valuesPerRow;
            for (int j = 0; j < valuesPerRow; j++)
            {
                selectedPredicted[targetBase + j] = predicted[sourceBase + j];
                selectedActual[targetBase + j] = actual[sourceBase + j];
            }
        }

        return (selectedPredicted, selectedActual);
    }
}
