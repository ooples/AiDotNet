using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.Encoders;

/// <summary>
/// Encodes categorical features using Weight of Evidence (WOE).
/// </summary>
/// <remarks>
/// <para>
/// Weight of Evidence is commonly used in credit scoring and binary classification.
/// It measures the strength of the relationship between a category and the binary target.
/// WOE = ln(Distribution of Events / Distribution of Non-Events)
/// </para>
/// <para>
/// Higher WOE values indicate categories more associated with the positive class,
/// while lower (negative) values indicate association with the negative class.
/// </para>
/// <para><b>For Beginners:</b> WOE tells you how "good" or "bad" a category is for prediction:
/// - WOE &gt; 0: Category is more likely to have positive outcomes
/// - WOE &lt; 0: Category is more likely to have negative outcomes
/// - WOE ≈ 0: Category has no predictive power
///
/// Example in loan default prediction:
/// - "Employed" might have WOE = -0.5 (less likely to default)
/// - "Unemployed" might have WOE = +0.8 (more likely to default)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.Encoder)]
[PipelineStage(PipelineStage.Preprocessing)]
public class WOEEncoder<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly double _regularization;
    private readonly WOEHandleUnknown _handleUnknown;

    // Fitted parameters
    private Dictionary<int, CategoryValueMap<T>>? _woeValues;
    private int _nInputFeatures;

    /// <summary>
    /// Gets the regularization parameter to prevent infinite WOE values.
    /// </summary>
    public double Regularization => _regularization;

    /// <summary>
    /// Gets how unknown categories are handled.
    /// </summary>
    public WOEHandleUnknown HandleUnknown => _handleUnknown;

    /// <summary>
    /// Gets the WOE values for each category.
    /// </summary>
    public Dictionary<int, CategoryValueMap<T>>? WOEValues => _woeValues;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="WOEEncoder{T}"/>.
    /// </summary>
    /// <param name="regularization">Regularization to add to counts to prevent division by zero. Defaults to 0.5.</param>
    /// <param name="handleUnknown">How to handle unknown categories. Defaults to UseZero.</param>
    /// <param name="columnIndices">The column indices to encode, or null for all columns.</param>
    public WOEEncoder(
        double regularization = 0.5,
        WOEHandleUnknown handleUnknown = WOEHandleUnknown.UseZero,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (regularization < 0)
        {
            throw new ArgumentException("Regularization must be non-negative.", nameof(regularization));
        }

        _regularization = regularization;
        _handleUnknown = handleUnknown;
    }

    /// <summary>
    /// Fits the encoder (requires binary target via specialized Fit method).
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WOEEncoder requires binary target values (0/1) for fitting. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the encoder by computing WOE values for each category.
    /// </summary>
    /// <param name="data">The feature matrix to fit.</param>
    /// <param name="target">The binary target values (0 or 1).</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
        {
            throw new ArgumentException("Target length must match the number of rows in data.");
        }

        _nInputFeatures = data.Columns;
        var columnsToProcess = GetColumnsToProcess(_nInputFeatures);

        // Count total events (1s) and non-events (0s)
        var y = Tensor<T>.FromVector(target);
        var events = IsOne(y);
        var nonEvents = Engine.TensorLessThan(Engine.TensorAbs(y), PreprocessingTensorOps.Tolerance<T>(1e-10));
        var invalid = PreprocessingTensorOps.MaskPositions(PreprocessingTensorOps.Not(Engine.TensorAdd(events, nonEvents)));
        if (invalid.Length > 0)
            throw new ArgumentException($"Target must be binary (0 or 1). Found value: {y.GetFlat(invalid[0])}");

        int totalEvents = PreprocessingTensorOps.CountNonZero(events);
        int totalNonEvents = PreprocessingTensorOps.CountNonZero(nonEvents);
        if (totalEvents == 0 || totalNonEvents == 0)
        {
            throw new ArgumentException("Target must contain both classes (0 and 1).");
        }

        // WOE per category from its event and non-event counts
        _woeValues = new Dictionary<int, CategoryValueMap<T>>();
        foreach (int col in columnsToProcess)
        {
            var column = PreprocessingTensorOps.Column(data, col);
            var categories = PreprocessingTensorOps.SortedUnique(column);
            var woe = WeightOfEvidence(column, events, categories, totalEvents, totalNonEvents, out _, out _);
            _woeValues[col] = new CategoryValueMap<T>(categories, woe);
        }

        IsFitted = true;
    }

    /// <summary>A 0/1 mask of the targets within 1e-10 of 1 (the events).</summary>
    private Tensor<T> IsOne(Tensor<T> y) =>
        Engine.TensorLessThan(Engine.TensorAbs(Engine.TensorSubtractScalar(y, NumOps.One)), PreprocessingTensorOps.Tolerance<T>(1e-10));

    /// <summary>
    /// Regularized event and non-event distributions per category and their clamped weight of evidence
    /// <c>clamp(log(distEvents / distNonEvents), -5, 5)</c>; rows that are not events count as non-events.
    /// </summary>
    private Tensor<T> WeightOfEvidence(Tensor<T> column, Tensor<T> events, Tensor<T> categories, int totalEvents,
        int totalNonEvents, out Tensor<T> distEvents, out Tensor<T> distNonEvents)
    {
        var engine = Engine;
        int k = categories.Length;
        var position = PreprocessingTensorOps.LookupSorted(categories, column, out _);
        var counts = PreprocessingTensorOps.CategoryCounts<T>(position, k);
        var eventCounts = PreprocessingTensorOps.CategorySums(position, events, k);
        var nonEventCounts = engine.TensorSubtract(counts, eventCounts);

        // Add regularization to prevent log(0)
        T regularization = NumOps.FromDouble(_regularization);
        distEvents = engine.TensorDivideScalar(
            engine.TensorAddScalar(eventCounts, regularization), NumOps.FromDouble(totalEvents + _regularization * k));
        distNonEvents = engine.TensorDivideScalar(
            engine.TensorAddScalar(nonEventCounts, regularization), NumOps.FromDouble(totalNonEvents + _regularization * k));

        // Clamp to prevent extreme values
        return engine.TensorClamp(engine.TensorLog(engine.TensorDivide(distEvents, distNonEvents)),
            NumOps.FromDouble(-5), NumOps.FromDouble(5));
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
    /// Transforms the data by replacing categories with WOE values.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_woeValues is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        // Unknown categories get 0 (or throw)
        var processSet = new HashSet<int>(GetColumnsToProcess(data.Columns));
        return CategoryValueMap<T>.EncodeColumns(data, processSet, _woeValues, _ => NumOps.Zero,
            _handleUnknown == WOEHandleUnknown.Error);
    }

    /// <summary>
    /// Calculates Information Value (IV) for each feature.
    /// </summary>
    /// <remarks>
    /// IV measures the overall predictive power of a feature.
    /// IV &lt; 0.02: Not useful for prediction
    /// 0.02 &lt; IV &lt; 0.1: Weak predictor
    /// 0.1 &lt; IV &lt; 0.3: Medium predictor
    /// 0.3 &lt; IV &lt; 0.5: Strong predictor
    /// IV &gt; 0.5: Suspicious (possible overfitting)
    /// </remarks>
    /// <param name="data">The feature matrix.</param>
    /// <param name="target">The binary target.</param>
    /// <returns>Dictionary mapping column index to IV value.</returns>
    public Dictionary<int, T> CalculateInformationValue(Matrix<T> data, Vector<T> target)
    {
        if (_woeValues is null)
        {
            throw new InvalidOperationException("Encoder has not been fitted.");
        }

        // Count total events and non-events (every target that is not 1 is a non-event)
        var events = IsOne(Tensor<T>.FromVector(target));
        int totalEvents = PreprocessingTensorOps.CountNonZero(events);
        int totalNonEvents = target.Length - totalEvents;

        var ivValues = new Dictionary<int, T>();
        foreach (var colKvp in _woeValues)
        {
            int col = colKvp.Key;
            var column = PreprocessingTensorOps.Column(data, col);
            var categories = PreprocessingTensorOps.SortedUnique(column);
            WeightOfEvidence(column, events, categories, totalEvents, totalNonEvents, out var distEvents, out var distNonEvents);

            // IV = sum over categories of (distEvents - distNonEvents) * fitted WOE (0 for categories not seen in fit)
            var woe = colKvp.Value.Map(categories, new Tensor<T>(new[] { categories.Length }), out _);
            ivValues[col] = Engine.TensorSum(Engine.TensorMultiply(Engine.TensorSubtract(distEvents, distNonEvents), woe));
        }

        return ivValues;
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("WOEEncoder does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        return inputFeatureNames ?? Array.Empty<string>();
    }
}

/// <summary>
/// Specifies how to handle unknown categories during transformation.
/// </summary>
public enum WOEHandleUnknown
{
    /// <summary>
    /// Use WOE = 0 for unknown categories (neutral evidence).
    /// </summary>
    UseZero,

    /// <summary>
    /// Raise an error when an unknown category is encountered.
    /// </summary>
    Error
}
