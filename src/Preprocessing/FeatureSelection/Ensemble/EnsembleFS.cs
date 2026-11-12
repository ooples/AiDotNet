using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Ensemble;

/// <summary>
/// Ensemble Feature Selection combining multiple methods.
/// </summary>
/// <remarks>
/// <para>
/// Ensemble Feature Selection combines the results of multiple feature selection
/// methods using voting or rank aggregation. Features consistently selected across
/// multiple methods are more likely to be truly important.
/// </para>
/// <para><b>For Beginners:</b> Different feature selection methods have different
/// biases and catch different patterns. By combining multiple methods (like asking
/// multiple experts), we get a more reliable answer. Features that everyone agrees
/// are important are more trustworthy than those only one method likes.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class EnsembleFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _threshold;
    private readonly AggregationMethod _aggregation;

    private Vector<T>? _aggregatedScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public enum AggregationMethod
    {
        MajorityVoting,
        WeightedVoting,
        RankAggregation,
        BordaCount
    }

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Threshold => _threshold;
    public AggregationMethod Aggregation => _aggregation;
    public Vector<T>? AggregatedScores => _aggregatedScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public EnsembleFS(
        int nFeaturesToSelect = 10,
        double threshold = 0.5,
        AggregationMethod aggregation = AggregationMethod.MajorityVoting,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (threshold <= 0 || threshold > 1)
            throw new ArgumentException("Threshold must be between 0 and 1.", nameof(threshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _threshold = threshold;
        _aggregation = aggregation;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "EnsembleFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var X = Tensor<T>.FromMatrix(data);
        var y = Tensor<T>.FromVector(target);

        // Run multiple base methods and collect rankings (0 = best, ties by index)
        var methodRankings = new List<Tensor<T>>();
        var methodScores = new List<Tensor<T>>();

        // Method 1: Correlation-based
        var corrScores = engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(X, y));
        methodRankings.Add(GetRanking(corrScores));
        methodScores.Add(corrScores);

        // Method 2: Variance-based (combined with correlation)
        var varianceScores = ComputeVarianceScores(X, n, p);
        var combinedVar = engine.TensorMultiply(corrScores, varianceScores);
        methodRankings.Add(GetRanking(combinedVar));
        methodScores.Add(combinedVar);

        // Method 3: Fisher score approximation
        var fisherScores = ComputeFisherScoreApprox(X, y, n, p);
        methodRankings.Add(GetRanking(fisherScores));
        methodScores.Add(fisherScores);

        // Method 4: Information gain approximation (correlation; they're related for continuous data)
        var igScores = corrScores;
        methodRankings.Add(GetRanking(igScores));
        methodScores.Add(igScores);

        // Aggregate results
        var aggregatedScores = _aggregation switch
        {
            AggregationMethod.MajorityVoting => AggregateMajorityVoting(methodRankings, p),
            AggregationMethod.WeightedVoting => AggregateWeightedVoting(methodScores, p),
            AggregationMethod.RankAggregation => AggregateRanks(methodRankings, p),
            AggregationMethod.BordaCount => AggregateBordaCount(methodRankings, p),
            _ => AggregateMajorityVoting(methodRankings, p)
        };
        _aggregatedScores = aggregatedScores.ToVector();

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(aggregatedScores, numToSelect);

        IsFitted = true;
    }

    private Tensor<T> ComputeVarianceScores(Tensor<T> X, int n, int p)
    {
        // Population variance, divided by the largest variance when that is positive
        var engine = Engine;
        var scores = engine.ReduceMean(engine.TensorSquare(engine.TensorSubtract(X,
            engine.TensorBroadcastTo(engine.ReduceMean(X, new[] { 0 }, true), new[] { n, p }))), new[] { 0 }, false);
        T maxVar = engine.TensorMaxValue(scores);
        return PreprocessingTensorOps.IsAbove(maxVar, NumOps.Zero) ? engine.TensorDivideScalar(scores, maxVar) : scores;
    }

    private Tensor<T> ComputeFisherScoreApprox(Tensor<T> X, Tensor<T> y, int n, int p)
    {
        // Binary split of the rows at the target mean: (mean0 - mean1)^2 / (var0 / n0 + var1 / n1)
        var engine = Engine;
        var shape = new[] { n, p };
        var upper = engine.TensorGreaterThan(y, engine.TensorMean(y));
        var lower = PreprocessingTensorOps.Not(upper);
        var one = NumOps.One;

        Tensor<T> GroupMean(Tensor<T> mask, out Tensor<T> rowMask, out T divisor)
        {
            rowMask = engine.TensorBroadcastTo(engine.Reshape(mask, new[] { n, 1 }), shape);
            divisor = PreprocessingTensorOps.ScalarMax(one, engine.TensorSum(mask));
            return engine.TensorDivideScalar(engine.ReduceSum(engine.TensorMultiply(X, rowMask), new[] { 0 }, true), divisor);
        }

        var mean0 = GroupMean(lower, out var mask0, out T count0);
        var mean1 = GroupMean(upper, out var mask1, out T count1);
        var var0 = engine.ReduceSum(engine.TensorMultiply(engine.TensorSquare(
            engine.TensorSubtract(X, engine.TensorBroadcastTo(mean0, shape))), mask0), new[] { 0 }, false);
        var var1 = engine.ReduceSum(engine.TensorMultiply(engine.TensorSquare(
            engine.TensorSubtract(X, engine.TensorBroadcastTo(mean1, shape))), mask1), new[] { 0 }, false);

        var between = engine.Reshape(engine.TensorSquare(engine.TensorSubtract(mean0, mean1)), new[] { p });
        var within = engine.TensorAdd(engine.TensorDivideScalar(var0, count0), engine.TensorDivideScalar(var1, count1));
        var usable = engine.TensorGreaterThan(within, NumOps.FromDouble(1e-10));
        var safe = engine.TensorWhere(usable, within, PreprocessingTensorOps.Filled(new[] { p }, one));
        return engine.TensorWhere(usable, engine.TensorDivide(between, safe), new Tensor<T>(new[] { p }));
    }

    private Tensor<T> GetRanking(Tensor<T> scores)
    {
        // Rank of every feature in the stable descending order of its score
        int p = scores.Length;
        var order = PreprocessingTensorOps.StableOrder(scores, descending: true);
        var ranks = new int[p];
        for (int rank = 0; rank < p; rank++)
            ranks[order[rank]] = rank;
        return Engine.TensorIndexSelect(PreprocessingTensorOps.Range<T>(p), new Tensor<int>(ranks, new[] { p }), 0);
    }

    private Tensor<T> AggregateMajorityVoting(List<Tensor<T>> rankings, int p)
    {
        // One vote per method for each of its top-k ranked features
        var votes = new Tensor<T>(new[] { p });
        T topK = NumOps.FromDouble(_nFeaturesToSelect);

        foreach (var ranking in rankings)
            votes = Engine.TensorAdd(votes, Engine.TensorLessThan(ranking, topK));

        return votes;
    }

    private Tensor<T> AggregateWeightedVoting(List<Tensor<T>> allScores, int p)
    {
        var aggregated = new Tensor<T>(new[] { p });
        T zero = NumOps.Zero;

        foreach (var scores in allScores)
        {
            T max = Engine.TensorMaxValue(scores);
            if (PreprocessingTensorOps.IsAbove(max, zero))
                aggregated = Engine.TensorAdd(aggregated, Engine.TensorDivideScalar(scores, max));
        }

        return aggregated;
    }

    private Tensor<T> AggregateRanks(List<Tensor<T>> rankings, int p)
    {
        var rankSums = new Tensor<T>(new[] { p });

        foreach (var ranking in rankings)
            rankSums = Engine.TensorAdd(rankSums, ranking);

        // Convert to scores (lower rank = higher score)
        return Engine.TensorSubtract(PreprocessingTensorOps.Filled(new[] { p }, Engine.TensorMaxValue(rankSums)), rankSums);
    }

    private Tensor<T> AggregateBordaCount(List<Tensor<T>> rankings, int p)
    {
        var bordaScores = new Tensor<T>(new[] { p });
        var features = PreprocessingTensorOps.Filled(new[] { p }, NumOps.FromDouble(p));

        foreach (var ranking in rankings)
            bordaScores = Engine.TensorAdd(bordaScores, Engine.TensorSubtract(features, ranking));

        return bordaScores;
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
            throw new InvalidOperationException("EnsembleFS has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("EnsembleFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EnsembleFS has not been fitted.");

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
