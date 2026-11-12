using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection;

/// <summary>
/// Plug-in (count based) entropies and mutual informations of discretised features and a discrete target, for every
/// feature at once with engine operations. All logarithms are natural.
/// </summary>
/// <remarks>
/// The features are an [n, p] int tensor of codes in <c>[0, levels)</c>; the target is a one-hot [n, k] class
/// membership. Methods taking a feature index <c>s</c> return the quantity for every feature j paired with s.
/// </remarks>
internal sealed class DiscreteInformation<T>
{
    private readonly IEngine _engine = AiDotNetEngine.Current;
    private readonly INumericOperations<T> _ops = MathHelper.GetNumericOperations<T>();
    private readonly Tensor<int> _codes;
    private readonly int _levels;
    private readonly Tensor<T> _targetOneHot;
    private readonly int _n;
    private readonly int _p;
    private readonly T _total;
    private Tensor<T>? _featureEntropy;
    private Tensor<T>? _featureTargetEntropy;

    public DiscreteInformation(Tensor<int> codes, int levels, Tensor<T> targetOneHot)
    {
        _codes = codes;
        _levels = levels;
        _targetOneHot = targetOneHot;
        _n = codes.Shape[0];
        _p = codes.Shape[1];
        _total = _ops.FromDouble(Math.Max(_n, 1));
        TargetEntropy = _engine.TensorSum(FeatureSelectionTensorOps.Entropy(
            _engine.Reshape(FeatureSelectionTensorOps.Counts(targetOneHot, 0), new[] { 1, targetOneHot.Shape[1] }), _total, 1));
    }

    /// <summary>
    /// Discrete information of <paramref name="codes"/> against the classes of <paramref name="target"/>, each target
    /// value rounded to the nearest integer label like <c>(int)Math.Round(value)</c>.
    /// </summary>
    public static DiscreteInformation<T> ForTarget(Tensor<int> codes, int levels, Vector<T> target)
    {
        var classIndex = FilterStatisticTensorOps.ClassIndex(target, out int classCount);
        return new DiscreteInformation<T>(codes, levels, FilterStatisticTensorOps.ClassOneHot<T>(classIndex, classCount));
    }

    /// <summary>
    /// Greedy forward selection of up to <paramref name="count"/> of <paramref name="p"/> features. Each step,
    /// <paramref name="candidateScores"/> receives the features chosen so far (in selection order) and returns a [p]
    /// score; the highest score among the features not yet chosen wins, ties going to the lowest index.
    /// </summary>
    /// <returns>The chosen features in ascending order and a [p] vector of each chosen feature's winning score (zero
    /// for features never chosen).</returns>
    public static (int[] Selected, Vector<T> Scores) Greedy(int p, int count, Func<IReadOnlyList<int>, Tensor<T>> candidateScores)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        T zero = ops.Zero;
        T one = ops.One;
        T negativeInfinity = ops.FromDouble(double.NegativeInfinity);
        var available = new T[p];
        var won = new T[p];
        for (int j = 0; j < p; j++)
        {
            available[j] = one;
            won[j] = zero;
        }

        var selected = new List<int>();
        int target = Math.Min(count, p);
        while (selected.Count < target)
        {
            var masked = FeatureSelectionTensorOps.KeepOrFill(
                candidateScores(selected), new Tensor<T>((T[])available.Clone(), new[] { p }), negativeInfinity);
            int best = FilterStatisticTensorOps.FirstArgMax(masked);
            won[best] = masked.ToArray()[best];
            available[best] = zero;
            selected.Add(best);
        }

        return (selected.OrderBy(j => j).ToArray(), new Tensor<T>(won, new[] { p }).ToVector());
    }

    /// <summary>
    /// The running mean after a <paramref name="count"/>-th term: <c>mean + (next - mean) / count</c>. The mean of
    /// the per-step terms of a greedy criterion, kept as a mean rather than a sum over a step counter in T: in Half
    /// a counter stops at 2048 (2048 + 1 rounds back to 2048) and a growing sum stops taking terms below half its
    /// spacing, while the mean stays on the scale of a single term.
    /// </summary>
    public static Tensor<T> RunningMean(Tensor<T> mean, Tensor<T> next, int count)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        return engine.TensorAdd(mean, engine.TensorDivideScalar(engine.TensorSubtract(next, mean), ops.FromDouble(count)));
    }

    /// <summary>Number of features.</summary>
    public int FeatureCount => _p;

    /// <summary>H(Y).</summary>
    public T TargetEntropy { get; }

    /// <summary>H(X_j) for every feature ([p]).</summary>
    public Tensor<T> FeatureEntropy()
    {
        if (_featureEntropy is null)
            _featureEntropy = FeatureSelectionTensorOps.Entropy(FeatureSelectionTensorOps.CodeCounts<T>(_codes, _levels), _total, 1);
        return _featureEntropy;
    }

    /// <summary>H(X_j, Y) for every feature ([p]).</summary>
    public Tensor<T> FeatureTargetEntropy()
    {
        if (_featureTargetEntropy is null)
        {
            int k = _targetOneHot.Shape[1];
            var joint = FeatureSelectionTensorOps.JointCounts(_codes, _levels, _targetOneHot);
            _featureTargetEntropy = FeatureSelectionTensorOps.Entropy(
                _engine.Reshape(joint, new[] { _p, _levels * k }), _total, 1);
        }

        return _featureTargetEntropy;
    }

    /// <summary>I(X_j; Y) = H(X_j) + H(Y) - H(X_j, Y) for every feature ([p]).</summary>
    public Tensor<T> Relevance() =>
        _engine.TensorAddScalar(_engine.TensorSubtract(FeatureEntropy(), FeatureTargetEntropy()), TargetEntropy);

    /// <summary>
    /// Symmetric uncertainty with the target, <c>2 I(X_j; Y) / (H(X_j) + H(Y))</c>, for every feature ([p]); 0 where
    /// the entropy sum is not above 1e-10.
    /// </summary>
    public Tensor<T> SymmetricUncertainty() =>
        FilterStatisticTensorOps.GuardedDivide(
            _engine.TensorMultiplyScalar(Relevance(), _ops.FromDouble(2)),
            _engine.TensorAddScalar(FeatureEntropy(), TargetEntropy), 1e-10, _ops.Zero);

    /// <summary>
    /// Symmetric uncertainty between features, <c>2 I(X_j; X_s) / (H(X_j) + H(X_s))</c>, for every feature j ([p]);
    /// 0 where the entropy sum is not above 1e-10.
    /// </summary>
    public Tensor<T> FeatureSymmetricUncertainty(int s)
    {
        var hj = FeatureEntropy();
        var hs = Broadcast(FeatureSelectionTensorOps.Entries(hj, new[] { s }));
        return FilterStatisticTensorOps.GuardedDivide(
            _engine.TensorMultiplyScalar(FeatureMutualInformation(s), _ops.FromDouble(2)),
            _engine.TensorAdd(hj, hs), 1e-10, _ops.Zero);
    }

    /// <summary>H(X_j | Y) = H(X_j, Y) - H(Y) for every feature ([p]).</summary>
    public Tensor<T> FeatureGivenTarget() =>
        _engine.TensorSubtractScalar(FeatureTargetEntropy(), TargetEntropy);

    /// <summary>H(X_j, X_s) for every feature j ([p]).</summary>
    public Tensor<T> PairEntropy(int s) =>
        FeatureSelectionTensorOps.Entropy(
            _engine.Reshape(FeatureSelectionTensorOps.PairedJointCounts<T>(_codes, _levels, Partner(s), _levels),
                new[] { _p, _levels * _levels }),
            _total, 1);

    /// <summary>H(X_j, X_s, Y) for every feature j ([p]).</summary>
    public Tensor<T> PairTargetEntropy(int s)
    {
        int k = _targetOneHot.Shape[1];
        var combined = _engine.TensorAdd(_engine.TensorMultiplyScalar(_codes, _levels), Partner(s));
        var joint = FeatureSelectionTensorOps.JointCounts(combined, _levels * _levels, _targetOneHot);
        return FeatureSelectionTensorOps.Entropy(_engine.Reshape(joint, new[] { _p, _levels * _levels * k }), _total, 1);
    }

    /// <summary>I(X_j; X_s) = H(X_j) + H(X_s) - H(X_j, X_s) for every feature j ([p]).</summary>
    public Tensor<T> FeatureMutualInformation(int s)
    {
        var hj = FeatureEntropy();
        var hs = Broadcast(FeatureSelectionTensorOps.Entries(hj, new[] { s }));
        return _engine.TensorSubtract(_engine.TensorAdd(hj, hs), PairEntropy(s));
    }

    /// <summary>
    /// I(X_j; X_s | Y) = H(X_j, Y) + H(X_s, Y) - H(X_j, X_s, Y) - H(Y) for every feature j ([p]).
    /// </summary>
    public Tensor<T> ConditionalFeatureMutualInformation(int s)
    {
        var hjy = FeatureTargetEntropy();
        var hsy = Broadcast(FeatureSelectionTensorOps.Entries(hjy, new[] { s }));
        return _engine.TensorSubtractScalar(_engine.TensorSubtract(_engine.TensorAdd(hjy, hsy), PairTargetEntropy(s)), TargetEntropy);
    }

    /// <summary>I(X_j, X_s; Y) = H(X_j, X_s) + H(Y) - H(X_j, X_s, Y) for every feature j ([p]).</summary>
    public Tensor<T> JointRelevance(int s) =>
        _engine.TensorAddScalar(_engine.TensorSubtract(PairEntropy(s), PairTargetEntropy(s)), TargetEntropy);

    /// <summary>
    /// I(X_j; Y | X_s) = H(X_j, X_s) + H(X_s, Y) - H(X_j, X_s, Y) - H(X_s) for every feature j ([p]).
    /// </summary>
    public Tensor<T> RelevanceGiven(int s)
    {
        var hs = Broadcast(FeatureSelectionTensorOps.Entries(FeatureEntropy(), new[] { s }));
        var hsy = Broadcast(FeatureSelectionTensorOps.Entries(FeatureTargetEntropy(), new[] { s }));
        return _engine.TensorSubtract(
            _engine.TensorSubtract(_engine.TensorAdd(PairEntropy(s), hsy), PairTargetEntropy(s)), hs);
    }

    // feature s's codes repeated for every feature column ([n, p])
    private Tensor<int> Partner(int s)
    {
        var column = _engine.TensorIndexSelect(_codes, new Tensor<int>(new[] { s }, new[] { 1 }), 1);
        return _engine.TensorBroadcastTo(column, new[] { _n, _p });
    }

    private Tensor<T> Broadcast(Tensor<T> single) => _engine.TensorBroadcastTo(single, new[] { _p });
}
