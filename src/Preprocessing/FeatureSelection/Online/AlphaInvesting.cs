using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Online;

/// <summary>
/// Alpha-Investing for online feature selection with FDR control.
/// </summary>
/// <remarks>
/// <para>
/// Alpha-Investing is an online algorithm for controlling the false discovery rate
/// when testing features sequentially. It maintains a "wealth" that increases when
/// null hypotheses are rejected and decreases when tests are performed.
/// </para>
/// <para><b>For Beginners:</b> Think of it like managing a budget for discoveries.
/// You start with some "discovery wealth." Each time you test a feature, you spend
/// some wealth. When you find a significant feature, you earn wealth back. This
/// ensures you don't make too many false discoveries while still finding real ones.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class AlphaInvesting<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _initialWealth;
    private readonly double _omega; // Wealth earned when rejecting null
    private readonly double _eta; // Fraction of wealth to invest per test

    private Vector<T>? _pValues;
    private Vector<T>? _testAlphas;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double InitialWealth => _initialWealth;
    public double Omega => _omega;
    public double Eta => _eta;
    public Vector<T>? PValues => _pValues;
    public Vector<T>? TestAlphas => _testAlphas;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public AlphaInvesting(
        int nFeaturesToSelect = 10,
        double initialWealth = 0.5,
        double omega = 0.05,
        double eta = 0.25,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (initialWealth <= 0)
            throw new ArgumentException("Initial wealth must be positive.", nameof(initialWealth));
        if (omega <= 0 || omega > 1)
            throw new ArgumentException("Omega must be between 0 and 1.", nameof(omega));
        if (eta <= 0 || eta > 1)
            throw new ArgumentException("Eta must be between 0 and 1.", nameof(eta));

        _nFeaturesToSelect = nFeaturesToSelect;
        _initialWealth = initialWealth;
        _omega = omega;
        _eta = eta;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AlphaInvesting requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var selected = new List<int>();

        // Compute p-values for all features
        var pValues = ComputePValue(data, target, n);
        _pValues = pValues.ToVector();

        // Order features by p-value for more efficient testing (ties by index)
        var orderedFeatures = FeatureSelectionTensorOps.StableOrder(pValues, descending: false);

        // Alpha-investing procedure (Foster & Stine 2008): each test spends alpha / (1 - alpha) of the wealth
        // when it fails and earns omega when it rejects; alpha = eta * w / (1 + eta * w) invests a fraction eta
        // of the current wealth w.
        var wealth = FeatureSelectionTensorOps.Filled(new[] { 1 }, NumOps.FromDouble(_initialWealth));
        T eta = NumOps.FromDouble(_eta);
        T omega = NumOps.FromDouble(_omega);
        var one = FeatureSelectionTensorOps.Filled(new[] { 1 }, NumOps.One);
        var testAlphas = new Tensor<T>(new[] { p });

        foreach (int featureIdx in orderedFeatures)
        {
            if (selected.Count >= _nFeaturesToSelect)
                break;

            if (FeatureSelectionTensorOps.CountNonZero(engine.TensorGreaterThan(wealth, NumOps.Zero)) == 0)
                break;

            // Compute alpha for this test
            var investment = engine.TensorMultiplyScalar(wealth, eta);
            var alpha = engine.TensorDivide(investment, engine.TensorAdd(one, investment));
            testAlphas = engine.TensorAdd(testAlphas, engine.TensorMultiply(
                FeatureSelectionTensorOps.OneHot<T>(featureIdx, p), engine.TensorBroadcastTo(alpha, new[] { p })));

            // Test: reject the null hypothesis (the feature is significant) when p <= alpha
            var pValue = engine.TensorSlice(pValues, new[] { featureIdx }, new[] { 1 });
            bool rejected = FeatureSelectionTensorOps.CountNonZero(engine.TensorLessThan(pValue, alpha))
                + FeatureSelectionTensorOps.CountNonZero(engine.TensorEquals(pValue, alpha)) > 0;
            if (rejected)
            {
                selected.Add(featureIdx);

                // Gain wealth from discovery
                wealth = engine.TensorAddScalar(wealth, omega);
            }
            else
            {
                // Pay for the failed test
                wealth = engine.TensorSubtract(wealth, engine.TensorDivide(alpha, engine.TensorSubtract(one, alpha)));
            }
        }

        _testAlphas = testAlphas.ToVector();
        _selectedIndices = selected.OrderBy(x => x).ToArray();

        // If not enough selected, add top features by p-value
        if (_selectedIndices.Length < _nFeaturesToSelect)
        {
            var additional = orderedFeatures
                .Where(x => !selected.Contains(x))
                .Take(_nFeaturesToSelect - _selectedIndices.Length);
            _selectedIndices = selected.Concat(additional).OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    /// <summary>
    /// Two-sided p-value of every feature's Pearson correlation with the target (the t-test on n - 2 degrees of
    /// freedom, as scipy.stats.pearsonr); a feature or target without variance has r = 0 and p = 1.
    /// </summary>
    private Tensor<T> ComputePValue(Matrix<T> data, Vector<T> target, int n)
    {
        var r = FeatureSelectionInformationOps.AbsCorrelationWithTarget(
            Tensor<T>.FromMatrix(data), Engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), NumOps.Zero);
        return FeatureSelectionInformationOps.CorrelationPValue(r, n - 2);
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
            throw new InvalidOperationException("AlphaInvesting has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AlphaInvesting does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AlphaInvesting has not been fitted.");

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
