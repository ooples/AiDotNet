using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Clustering;

/// <summary>
/// Silhouette Score based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that maximize the silhouette score of clusters, indicating
/// how well clusters are separated and cohesive.
/// </para>
/// <para><b>For Beginners:</b> The silhouette score measures how similar a point is
/// to its own cluster versus other clusters. Features that give high silhouette
/// scores create well-separated, tight clusters.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SilhouetteBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _silhouetteScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? SilhouetteScores => _silhouetteScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SilhouetteBasedSelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SilhouetteBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = Tensor<T>.FromMatrix(data);
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;

        // Half and BFloat16 run in float: a sample's summed distances to a class pass Half's 65504 (100 samples
        // 1000 away sum to 100000), and the own-class mask then multiplies that inf by 0
        var scores = PreprocessingTensorOps.IsNarrowFloat<T>()
            ? ComputeSilhouetteScores(X.Cast<float>(), classIndex, k)
            : ComputeSilhouetteScores(X, classIndex, k);
        _silhouetteScores = scores.Cast<T>().ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);
        IsFitted = true;
    }

    // Upper bound on the elements of one block of reference rows against every sample ([rows, n]).
    private const int PairBlockElements = 1 << 22;

    /// <summary>
    /// The mean silhouette (Rousseeuw 1987) of every column of <paramref name="X"/> ([n, p]) taken alone, with the
    /// samples clustered by class. Reference rows are processed in blocks against every sample, so the working
    /// storage is bounded rather than [n, n] per feature (3.2 GB of doubles for 20,000 rows).
    /// </summary>
    private static Tensor<double> ComputeSilhouetteScores<TC>(Tensor<TC> X, int[] classIndex, int k)
    {
        var engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<TC>();
        int n = X.Shape[0];
        int p = X.Shape[1];

        // Class membership, class sizes and each sample's own-class size
        var Y = PreprocessingTensorOps.ClassOneHot<TC>(classIndex, k);
        // Counted in int: a Half sum of the 0/1 membership stops at 2048
        var counts = engine.ReduceSum(Y.Cast<int>(), new[] { 0 }, false).Cast<TC>();
        var ownCount = engine.Reshape(engine.TensorMatMul(Y, engine.Reshape(counts, new[] { k, 1 })), new[] { n });
        var inMultiMemberClass = engine.TensorGreaterThan(ownCount, ops.One);
        var ownDenominator = engine.TensorMax(engine.TensorSubtractScalar(ownCount, ops.One), ops.One);
        TC maxValue = ops.MaxValue;
        TC zero = ops.Zero;
        int blockRows = Math.Max(1, PairBlockElements / Math.Max(1, n));

        var means = new double[p];
        for (int j = 0; j < p; j++)
        {
            var column = PreprocessingTensorOps.ColumnVector(X, j);
            var everySample = engine.Reshape(column, new[] { 1, n });
            double total = 0;
            for (int start = 0; start < n; start += blockRows)
            {
                int rows = Math.Min(blockRows, n - start);
                var blockShape = new[] { rows, n };
                var classShape = new[] { rows, k };
                var zeros = new Tensor<TC>(new[] { rows });

                // Pairwise 1-D distances of the block's rows to every sample and their per-class sums
                var reference = engine.TensorSlice(column, new[] { start }, new[] { rows });
                var distances = engine.TensorAbs(engine.TensorSubtract(
                    engine.TensorBroadcastTo(engine.Reshape(reference, new[] { rows, 1 }), blockShape),
                    engine.TensorBroadcastTo(everySample, blockShape)));
                var classSums = engine.TensorMatMul(distances, Y);
                var blockY = engine.TensorSlice(Y, new[] { start, 0 }, classShape);
                var multiMember = engine.TensorSlice(inMultiMemberClass, new[] { start }, new[] { rows });

                // a(i) - average distance to own cluster (self excluded)
                var a = engine.TensorWhere(multiMember,
                    engine.TensorDivide(engine.ReduceSum(engine.TensorMultiply(classSums, blockY), new[] { 1 }, false),
                        engine.TensorSlice(ownDenominator, new[] { start }, new[] { rows })),
                    zeros);

                // b(i) - min average distance to other clusters (0 when there is no other cluster)
                var classSizes = engine.TensorBroadcastTo(engine.Reshape(counts, new[] { 1, k }), classShape);
                var otherAverages = engine.TensorWhere(blockY,
                    PreprocessingTensorOps.Filled(classShape, maxValue), engine.TensorDivide(classSums, classSizes));
                var b = engine.TensorNegate(engine.ReduceMax(engine.TensorNegate(otherAverages), new[] { 1 }, false));
                b = engine.TensorWhere(engine.TensorEquals(b, maxValue), zeros, b);

                // s(i) = (b - a) / max(a, b); 0 for a sample alone in its cluster (Rousseeuw 1987)
                var largest = engine.TensorMax(a, b);
                var s = engine.TensorWhere(engine.TensorGreaterThan(largest, zero),
                    engine.TensorDivide(engine.TensorSubtract(b, a), largest), zeros);
                s = engine.TensorWhere(multiMember, s, zeros);

                // Summed in double: a Half sum of n silhouettes stalls once it passes 2048 times their spacing
                total += engine.TensorSum(s.Cast<double>());
            }

            means[j] = total / n;
        }

        return new Tensor<double>(means, new[] { p });
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
            throw new InvalidOperationException("SilhouetteBasedSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SilhouetteBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SilhouetteBasedSelector has not been fitted.");

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
