using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Sequential;

/// <summary>
/// Sequential Floating Forward Selection (SFFS).
/// </summary>
/// <remarks>
/// <para>
/// Combines forward selection with conditional backward steps to escape local optima,
/// allowing previously added features to be removed if they become redundant.
/// </para>
/// <para><b>For Beginners:</b> Regular forward selection only adds features but never
/// removes them. SFFS adds features but can also remove them if doing so improves
/// the result. This "floating" behavior helps find better feature combinations.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FloatingForwardSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxBacktracks;

    private Vector<T>? _featureImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureImportances => _featureImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FloatingForwardSelector(
        int nFeaturesToSelect = 10,
        int maxBacktracks = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxBacktracks = maxBacktracks;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FloatingForwardSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var x = Tensor<T>.FromMatrix(data);
        var y = Engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        var selected = new HashSet<int>();
        var remaining = new HashSet<int>(Enumerable.Range(0, p));
        var importances = new Dictionary<int, T>();
        T currentScore = NumOps.Zero;

        while (selected.Count < numToSelect && remaining.Count > 0)
        {
            // Forward step: add best feature
            int bestAdd = -1;
            T bestAddScore = currentScore;

            foreach (int j in remaining)
            {
                selected.Add(j);
                T score = SelectorTensorOps.RidgeRSquared(x, y, selected, 1e-6);
                if (PreprocessingTensorOps.IsAbove(score, bestAddScore))
                {
                    bestAddScore = score;
                    bestAdd = j;
                }
                selected.Remove(j);
            }

            if (bestAdd < 0) break;

            selected.Add(bestAdd);
            remaining.Remove(bestAdd);
            importances[bestAdd] = bestAddScore;
            currentScore = bestAddScore;

            // Backward steps: try removing features (over a snapshot, since each trial edits the set)
            for (int backtrack = 0; backtrack < _maxBacktracks && selected.Count > 1; backtrack++)
            {
                int bestRemove = -1;
                T bestRemoveScore = currentScore;

                foreach (int j in selected.ToArray())
                {
                    if (j == bestAdd) continue; // Don't remove just-added feature

                    selected.Remove(j);
                    T score = SelectorTensorOps.RidgeRSquared(x, y, selected, 1e-6);
                    if (PreprocessingTensorOps.IsAbove(score, bestRemoveScore))
                    {
                        bestRemoveScore = score;
                        bestRemove = j;
                    }
                    selected.Add(j);
                }

                if (bestRemove >= 0)
                {
                    selected.Remove(bestRemove);
                    remaining.Add(bestRemove);
                    currentScore = bestRemoveScore;
                }
                else
                {
                    break; // No improvement from removing
                }
            }
        }

        _selectedIndices = selected.OrderBy(index => index).ToArray();

        // Normalize feature importances
        _featureImportances = SelectorTensorOps.NormalizeByPositiveMax(SelectorTensorOps.FromSparse(importances, p)).ToVector();

        IsFitted = true;
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FloatingForwardSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FloatingForwardSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FloatingForwardSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
