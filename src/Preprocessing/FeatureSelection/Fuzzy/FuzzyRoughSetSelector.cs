using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Fuzzy;

/// <summary>
/// Fuzzy Rough Set Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Combines fuzzy set theory with rough set theory to handle both uncertainty
/// and vagueness in data. Uses fuzzy equivalence relations to compute fuzzy
/// positive regions and find feature reducts.
/// </para>
/// <para><b>For Beginners:</b> Traditional rough sets use sharp boundaries
/// (same/different), but real data often has degrees of similarity. Fuzzy
/// rough sets allow saying "these objects are 80% similar" instead of just
/// "same" or "different". This gives more nuanced feature selection.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class FuzzyRoughSetSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _maxFeatures;
    private readonly double _sigma;

    private Vector<T>? _fuzzyDependencies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MaxFeatures => _maxFeatures;
    public double Sigma => _sigma;
    public Vector<T>? FuzzyDependencies => _fuzzyDependencies;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public FuzzyRoughSetSelector(
        int maxFeatures = 20,
        double sigma = 0.2,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxFeatures < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeatures));
        if (sigma <= 0)
            throw new ArgumentException("Sigma must be positive.", nameof(sigma));

        _maxFeatures = maxFeatures;
        _sigma = sigma;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "FuzzyRoughSetSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var x = FeatureSelectionTensorOps.RangeScaled(FilterStatisticTensorOps.AsTensor(data));
        var classIndex = FilterStatisticTensorOps.ClassIndex(target, out int classCount);
        var classOneHot = FilterStatisticTensorOps.ClassOneHot<T>(classIndex, classCount);
        var decision = Engine.TensorMatMul(classOneHot, Engine.TensorTranspose(classOneHot));
        int n = data.Rows;

        // The [n, n] similarity of a feature is built when it is needed and reduced at once, so at most a few
        // [n, n] grids are alive (keeping all p of them was 40 GB of double at n = 10000, p = 50)
        // Individual feature fuzzy dependency, and the similarity of all features together (their minimum)
        var single = new T[p];
        var all = FilterStatisticTensorOps.Filled(new[] { n, n }, NumOps.One);
        for (int j = 0; j < p; j++)
        {
            var similarity = FeatureSimilarity(x, j);
            single[j] = FuzzyDependency(similarity, decision);
            all = Engine.TensorMin(all, similarity);
        }

        var individual = new Tensor<T>(single, new[] { p });
        _fuzzyDependencies = individual.ToVector();

        // Greedy forward selection based on fuzzy dependency
        var fullDependency = Scalar(NumOps.Subtract(FuzzyDependency(all, decision), NumOps.FromDouble(1e-6)));
        var current = Scalar(NumOps.Zero);
        var selectedSimilarity = FilterStatisticTensorOps.Filled(new[] { n, n }, NumOps.One);
        T excluded = NumOps.FromDouble(double.NegativeInfinity);
        var selected = new HashSet<int>();
        while (selected.Count < _maxFeatures && Engine.TensorLessThan(current, fullDependency).Cast<int>().ToArray()[0] == 1)
        {
            var candidates = new T[p];
            for (int j = 0; j < p; j++)
                candidates[j] = selected.Contains(j)
                    ? excluded
                    : FuzzyDependency(Engine.TensorMin(selectedSimilarity, FeatureSimilarity(x, j)), decision);

            // the first candidate with the highest dependency, if it improves on the current one
            var scores = new Tensor<T>(candidates, new[] { p });
            int bestFeature = FilterStatisticTensorOps.FirstArgMax(scores);
            var best = FeatureSelectionTensorOps.Entries(scores, new[] { bestFeature });
            if (Engine.TensorGreaterThan(best, current).Cast<int>().ToArray()[0] == 0)
                break;

            selected.Add(bestFeature);
            selectedSimilarity = Engine.TensorMin(selectedSimilarity, FeatureSimilarity(x, bestFeature));
            current = best;
        }

        _selectedIndices = selected.OrderBy(j => j).ToArray();

        // If not enough features selected, add by individual dependency
        if (_selectedIndices.Length < Math.Min(_maxFeatures, p))
        {
            var chosen = _selectedIndices;
            var remaining = PreprocessingTensorOps.StableOrder(individual, descending: true)
                .Except(chosen)
                .Take(_maxFeatures - chosen.Length);
            _selectedIndices = chosen.Concat(remaining).OrderBy(j => j).ToArray();
        }
        IsFitted = true;
    }

    // Gaussian similarity exp(-(x_i - x_k)^2 / (2 sigma^2)) of every pair of rows on feature j ([n, n])
    private Tensor<T> FeatureSimilarity(Tensor<T> x, int j)
    {
        var column = PreprocessingTensorOps.ColumnVector(x, j);
        int n = column.Length;
        var difference = Engine.TensorSubtract(
            FilterStatisticTensorOps.ColumnBroadcast(column, n), FilterStatisticTensorOps.RowBroadcast(column, n));
        return Engine.TensorExp(Engine.TensorMultiplyScalar(
            Engine.TensorSquare(difference), NumOps.FromDouble(-1.0 / (2 * _sigma * _sigma))));
    }

    // Fuzzy dependency: the mean over objects of the lower approximation min_{k != i} min(1, 1 - S_ik + D_ik)
    // (Lukasiewicz implication of the crisp decision relation D by the similarity S)
    private T FuzzyDependency(Tensor<T> similarity, Tensor<T> decision)
    {
        T one = NumOps.One;
        int n = similarity.Shape[0];
        var implication = Engine.TensorMin(Engine.TensorAddScalar(Engine.TensorSubtract(decision, similarity), one), one);
        implication = Engine.TensorWhere(Engine.TensorEye<T>(n), FilterStatisticTensorOps.FilledLike(implication, one), implication);
        var lower = Engine.TensorNegate(Engine.ReduceMax(Engine.TensorNegate(implication), new[] { 1 }, false));
        return Engine.TensorMean(lower);
    }

    private static Tensor<T> Scalar(T value) => new Tensor<T>(new[] { value }, new[] { 1 });

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
            throw new InvalidOperationException("FuzzyRoughSetSelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("FuzzyRoughSetSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("FuzzyRoughSetSelector has not been fitted.");

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
