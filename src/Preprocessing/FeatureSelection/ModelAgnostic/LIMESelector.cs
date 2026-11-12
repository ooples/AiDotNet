using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.ModelAgnostic;

/// <summary>
/// LIME (Local Interpretable Model-agnostic Explanations) based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// LIME explains individual predictions by approximating the model locally with
/// an interpretable linear model. Feature importance is derived from the linear
/// coefficients of these local explanations.
/// </para>
/// <para>
/// The algorithm:
/// 1. For each instance to explain:
///    a. Generate perturbed samples around the instance
///    b. Weight samples by proximity to original instance
///    c. Fit weighted linear regression on perturbed samples
///    d. Extract feature importance from coefficients
/// 2. Aggregate importance across all explained instances
/// </para>
/// <para><b>For Beginners:</b> LIME answers: "For this specific prediction,
/// which features mattered most?"
///
/// It works by creating slightly modified versions of your data point,
/// seeing how predictions change, and fitting a simple linear model to
/// understand which features drive the prediction locally.
///
/// Unlike global methods, LIME provides instance-specific explanations.
/// For feature selection, we average importance across many instances.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LIMESelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nSamplesToExplain;
    private readonly int _nPerturbations;
    private readonly double _kernelWidth;
    private readonly int? _randomState;
    private readonly Func<Matrix<T>, Vector<T>>? _predictFunc;

    // Fitted parameters
    private Vector<T>? _importances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the number of features to select.
    /// </summary>
    public int NFeaturesToSelect => _nFeaturesToSelect;

    /// <summary>
    /// Gets the computed LIME importance values (mean absolute coefficients).
    /// </summary>
    public Vector<T>? Importances => _importances;

    /// <summary>
    /// Gets the indices of selected features.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="LIMESelector{T}"/>.
    /// </summary>
    /// <param name="predictFunc">Function that makes predictions from data.</param>
    /// <param name="nFeaturesToSelect">Number of features to select. Defaults to 10.</param>
    /// <param name="nSamplesToExplain">Number of instances to explain. Defaults to 100.</param>
    /// <param name="nPerturbations">Number of perturbations per instance. Defaults to 500.</param>
    /// <param name="kernelWidth">Width of the exponential kernel. Defaults to 0.75.</param>
    /// <param name="randomState">Random seed for reproducibility.</param>
    /// <param name="columnIndices">The column indices to evaluate, or null for all columns.</param>
    public LIMESelector(
        Func<Matrix<T>, Vector<T>>? predictFunc = null,
        int nFeaturesToSelect = 10,
        int nSamplesToExplain = 100,
        int nPerturbations = 500,
        double kernelWidth = 0.75,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
        {
            throw new ArgumentException("Number of features to select must be at least 1.", nameof(nFeaturesToSelect));
        }

        if (nSamplesToExplain < 1)
        {
            throw new ArgumentException("Number of samples to explain must be at least 1.", nameof(nSamplesToExplain));
        }

        if (nPerturbations < 1)
        {
            throw new ArgumentException("Number of perturbations must be at least 1.", nameof(nPerturbations));
        }

        if (kernelWidth <= 0)
        {
            throw new ArgumentException("Kernel width must be positive.", nameof(kernelWidth));
        }

        _predictFunc = predictFunc;
        _nFeaturesToSelect = nFeaturesToSelect;
        _nSamplesToExplain = nSamplesToExplain;
        _nPerturbations = nPerturbations;
        _kernelWidth = kernelWidth;
        _randomState = randomState;
    }

    /// <summary>
    /// Fits the selector (requires target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "LIMESelector requires target values for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits LIME selector by computing local explanations and aggregating importance.
    /// </summary>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The target values.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // Compute feature means and (population) stds for perturbation; a constant feature uses std 1
        var x = Tensor<T>.FromMatrix(data);
        var means = engine.ReduceMean(x, new[] { 0 }, true);
        var stds = engine.TensorSqrt(engine.ReduceMean(
            engine.TensorSquare(engine.TensorSubtract(x, engine.TensorBroadcastTo(means, new[] { n, p }))), new[] { 0 }, true));
        stds = engine.TensorWhere(
            engine.TensorLessThan(stds, NumOps.FromDouble(1e-10)), PreprocessingTensorOps.Filled(new[] { 1, p }, NumOps.One), stds);

        // Sample instances to explain
        int nExplain = Math.Min(_nSamplesToExplain, n);
        var explainIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(nExplain).ToArray();

        // Accumulate |coefficient| of the local surrogate of every explained instance
        var importanceAccum = new Tensor<T>(new[] { p });
        foreach (int explainIdx in explainIndices)
        {
            var instance = PopulationSearchOps.Row(x, explainIdx);
            var coefficients = ExplainInstance(instance, means, stds, random);
            importanceAccum = engine.TensorAdd(importanceAccum, engine.TensorAbs(coefficients));
        }

        // Average importance
        var importances = engine.TensorDivideScalar(importanceAccum, NumOps.FromDouble(nExplain));
        _importances = importances.ToVector();

        // Select top features by importance
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(importances, nToSelect);

        IsFitted = true;
    }

    /// <summary>
    /// The local surrogate of LIME (Ribeiro, Singh and Guestrin 2016) for one instance [p]: each perturbation keeps
    /// feature j when a uniform draw exceeds 1/2 and otherwise replaces it with mean + std (2u - 1) (one more
    /// draw), in the order of the scalar loop; it is weighted by exp(-d^2 / width^2) with d the fraction of
    /// replaced features, and the coefficients (without intercept) of the weighted least-squares fit (ridge
    /// 1e-6) of the predictions on the binary keep-mask are returned ([p]).
    /// </summary>
    private Tensor<T> ExplainInstance(Tensor<T> instance, Tensor<T> means, Tensor<T> stds, Random random)
    {
        var engine = Engine;
        int p = instance.Length;
        int count = _nPerturbations;

        // Draws in the scalar order: the keep test of every feature, then the replacement value of a replaced one.
        // keep[cell] = 1 for a kept cell; source[cell] points at the replacement draw of a replaced cell.
        var keep = new int[count * p];
        var source = new int[count * p];
        var replacements = new List<Tensor<T>> { new Tensor<T>(new[] { 1 }) };
        for (int cell = 0; cell < keep.Length; cell++)
        {
            if (random.NextDouble() > 0.5)
            {
                keep[cell] = 1;
            }
            else
            {
                source[cell] = replacements.Count;
                replacements.Add(PopulationSearchOps.Uniform<T>(random, 1));
            }
        }

        var shape = new[] { count, p };
        var mask = engine.Reshape(new Tensor<int>(keep, new[] { count * p }).Cast<T>(), shape);
        var draws = engine.Reshape(
            engine.TensorIndexSelect(
                engine.TensorConcatenate(replacements.ToArray(), 0), new Tensor<int>(source, new[] { count * p }), 0),
            shape);
        var sampled = engine.TensorAdd(
            engine.TensorBroadcastTo(means, shape),
            engine.TensorMultiply(engine.TensorBroadcastTo(stds, shape), PopulationSearchOps.Affine(draws, 2, -1)));
        var perturbed = engine.TensorWhere(
            mask, engine.TensorBroadcastTo(engine.Reshape(instance, new[] { 1, p }), shape), sampled);

        // Exponential kernel on the fraction of replaced features
        var distance = ReplacedFraction(mask);
        var weights = engine.TensorExp(engine.TensorMultiplyScalar(
            engine.TensorSquare(distance), NumOps.FromDouble(-1.0 / (_kernelWidth * _kernelWidth))));

        // Weighted least squares on [1, mask]: (D^T W D + 1e-6 I)^-1 D^T W y
        var predictions = Predict(perturbed);
        var design = SubsetRegressionOps.WithIntercept(mask);
        var weighted = PopulationSearchOps.ScaleRows(design, weights);
        var transposed = engine.TensorTranspose(weighted);
        var gram = engine.TensorAdd(
            engine.TensorMatMul(transposed, design), engine.TensorMultiplyScalar(engine.TensorEye<T>(p + 1), NumOps.FromDouble(1e-6)));
        var rhs = engine.TensorMatMul(transposed, engine.Reshape(predictions, new[] { count, 1 }));
        var coefficients = engine.Reshape(LinalgDispatch<T>.Solve(gram, rhs), new[] { p + 1 });
        return engine.TensorSlice(coefficients, new[] { 1 }, new[] { p });
    }

    /// <summary>
    /// The fraction of replaced features of every perturbation row of a 0/1 keep-mask [m, p] ([m]); the replaced
    /// features are counted in int (a 0/1 sum in T stops counting at 2048 for Half).
    /// </summary>
    internal static Tensor<T> ReplacedFraction(Tensor<T> mask)
    {
        var engine = AiDotNetEngine.Current;
        int p = mask.Shape[1];
        var replaced = engine.ReduceSum(PreprocessingTensorOps.OneMinus(mask).Cast<int>(), new[] { 1 }, false).Cast<T>();
        return engine.TensorDivideScalar(replaced, MathHelper.GetNumericOperations<T>().FromDouble(p));
    }

    /// <summary>
    /// Predictions [m] for the rows of <paramref name="rows"/> [m, p]: the user's model, or by default the mean
    /// of the features as a simple proxy.
    /// </summary>
    private Tensor<T> Predict(Tensor<T> rows)
    {
        if (_predictFunc is not null)
            return Tensor<T>.FromVector(_predictFunc(rows.ToMatrix()));

        return Engine.ReduceMean(rows, new[] { 1 }, false);
    }

    /// <summary>
    /// Fits and transforms the data.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <summary>
    /// Transforms the data by selecting LIME-important features.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("LIMESelector has not been fitted.");
        }

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LIMESelector does not support inverse transformation.");
    }

    /// <summary>
    /// Gets a boolean mask indicating which features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
        {
            throw new InvalidOperationException("LIMESelector has not been fitted.");
        }

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
        {
            mask[idx] = true;
        }

        return mask;
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
        {
            return Array.Empty<string>();
        }

        if (inputFeatureNames is null)
        {
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();
        }

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
