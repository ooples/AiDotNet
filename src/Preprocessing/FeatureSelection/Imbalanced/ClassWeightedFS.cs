using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Imbalanced;

/// <summary>
/// Class-Weighted Feature Selection for imbalanced datasets.
/// </summary>
/// <remarks>
/// <para>
/// Class-Weighted Feature Selection applies class weights when computing feature
/// scores to give more importance to minority class samples. This helps select
/// features that are discriminative for underrepresented classes.
/// </para>
/// <para><b>For Beginners:</b> In datasets where one class has few examples,
/// normal feature selection might favor features that only work for the common class.
/// This method weights the rare class more heavily, ensuring selected features
/// can identify both common and rare cases.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ClassWeightedFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly ClassWeightingStrategy _weightingStrategy;

    private Vector<T>? _featureScores;
    private Vector<T>? _classWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public ClassWeightingStrategy WeightingStrategy => _weightingStrategy;
    public Vector<T>? FeatureScores => _featureScores;

    /// <summary>
    /// The weight of each class, in ascending order of the (rounded) class labels, like scikit-learn's
    /// <c>compute_class_weight</c> over <c>np.unique(y)</c>.
    /// </summary>
    public Vector<T>? ClassWeights => _classWeights;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ClassWeightedFS(
        int nFeaturesToSelect = 10,
        ClassWeightingStrategy weightingStrategy = ClassWeightingStrategy.Balanced,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        if (!Enum.IsDefined(typeof(ClassWeightingStrategy), weightingStrategy))
            throw new ArgumentException("Weighting strategy must be Balanced, Sqrt or Log.", nameof(weightingStrategy));

        _nFeaturesToSelect = nFeaturesToSelect;
        _weightingStrategy = weightingStrategy;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ClassWeightedFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        if (n == 0)
            throw new InvalidOperationException("Sequence contains no elements");

        var engine = Engine;

        // Classes are the rounded target values, in ascending order
        var classIds = FeatureSelectionInformationOps.ClassIndices(engine.TensorRound(Tensor<T>.FromVector(target)), out int nClasses);
        var membership = engine.TensorOneHot<T>(classIds, nClasses); // [n, C]
        var classCounts = engine.ReduceSum(membership, new[] { 0 }, false); // [C]

        // Compute class weights
        var classWeights = ComputeClassWeights(classCounts, n, nClasses);
        _classWeights = classWeights.ToVector();

        // Compute weighted feature scores
        var scores = ComputeWeightedScores(Tensor<T>.FromMatrix(data), membership, classCounts, classWeights, p);
        _featureScores = scores.ToVector();

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <summary>Class weights from the balanced ratio n / (classes * count) of every class ([C]).</summary>
    private Tensor<T> ComputeClassWeights(Tensor<T> classCounts, int n, int nClasses)
    {
        var engine = Engine;
        var ratio = engine.TensorDivide(
            FeatureSelectionTensorOps.Filled(new[] { nClasses }, FeatureSelectionTensorOps.Count<T>(n)),
            engine.TensorMultiplyScalar(classCounts, FeatureSelectionTensorOps.Count<T>(nClasses)));

        return _weightingStrategy switch
        {
            ClassWeightingStrategy.Sqrt => engine.TensorSqrt(ratio),
            ClassWeightingStrategy.Log => engine.TensorLog(engine.TensorAddScalar(ratio, NumOps.One)),
            _ => ratio
        };
    }

    /// <summary>
    /// Class-weighted Fisher score per feature: the weighted between-class scatter over the weighted within-class
    /// scatter (population class variances), each class weighted by weight * count; 0 when the within-class
    /// scatter is not above 1e-10.
    /// </summary>
    private Tensor<T> ComputeWeightedScores(Tensor<T> X, Tensor<T> membership, Tensor<T> classCounts, Tensor<T> classWeights, int p)
    {
        var engine = Engine;
        int nClasses = classCounts.Length;
        var counts = engine.Reshape(classCounts, new[] { nClasses, 1 });
        var membershipT = engine.TensorTranspose(membership); // [C, n]

        // Compute class means and (population) class variances
        var classMeans = engine.TensorDivide(engine.TensorMatMul(membershipT, X), counts); // [C, p]
        var residual = engine.TensorSubtract(X, engine.TensorMatMul(membership, classMeans));
        var classVars = engine.TensorDivide(engine.TensorMatMul(membershipT, engine.TensorSquare(residual)), counts); // [C, p]

        // Compute weighted Fisher score for each feature
        var classMass = engine.Reshape(engine.TensorMultiply(classWeights, classCounts), new[] { 1, nClasses }); // weight * count
        T totalWeight = engine.TensorSum(classMass);
        var overallMean = engine.TensorDivideScalar(engine.TensorMatMul(classMass, classMeans), totalWeight); // [1, p]
        var betweenVar = engine.Reshape(engine.TensorMatMul(classMass, engine.TensorSquare(engine.TensorSubtract(classMeans, overallMean))), new[] { p });
        var withinVar = engine.Reshape(engine.TensorMatMul(classMass, classVars), new[] { p });

        var defined = engine.TensorGreaterThan(withinVar, NumOps.FromDouble(1e-10));
        var safeWithin = engine.TensorWhere(defined, withinVar, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One));
        return engine.TensorWhere(defined, engine.TensorDivide(betweenVar, safeWithin), new Tensor<T>(new[] { p }));
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
            throw new InvalidOperationException("ClassWeightedFS has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ClassWeightedFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ClassWeightedFS has not been fitted.");

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
