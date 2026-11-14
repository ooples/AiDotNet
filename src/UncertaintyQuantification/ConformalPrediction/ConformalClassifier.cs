namespace AiDotNet.UncertaintyQuantification.ConformalPrediction;

/// <summary>
/// Implements Conformal Prediction for classification tasks.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Conformal Prediction for classification provides prediction sets
/// with guaranteed coverage, not just a single class prediction.
///
/// Key differences from regression conformal prediction:
/// - Instead of an interval, you get a SET of possible classes
/// - The set is guaranteed to contain the true class with specified probability
///
/// Example with 90% confidence:
/// - Traditional classifier: "This is a cat" (might be wrong)
/// - Conformal classifier: "This is {cat, dog}" (90% guaranteed to include correct class)
///
/// Benefits:
/// - When the model is uncertain, you get a larger prediction set (e.g., {cat, dog, rabbit})
/// - When the model is confident, you get a smaller set (e.g., {cat})
/// - You can defer to a human expert when the prediction set is too large
///
/// This is invaluable for:
/// - Medical diagnosis: Know when to seek specialist opinion
/// - Autonomous systems: Know when to hand control back to operator
/// - Quality control: Flag uncertain cases for manual review
/// </para>
/// </remarks>
public class ConformalClassifier<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly INeuralNetwork<T> _model;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _calibrationScores;
    private readonly int _numClasses;
    private bool _isCalibrated;

    /// <summary>
    /// Initializes a new instance of the ConformalClassifier class.
    /// </summary>
    /// <param name="model">The underlying classification model (must output class probabilities).</param>
    /// <param name="numClasses">The number of classes in the classification problem.</param>
    public ConformalClassifier(INeuralNetwork<T> model, int numClasses)
    {
        Guard.NotNull(model);
        _model = model;
        _numClasses = numClasses;
        _numOps = MathHelper.GetNumericOperations<T>();
        _isCalibrated = false;
    }

    /// <summary>
    /// Calibrates the conformal classifier using a calibration dataset.
    /// </summary>
    /// <param name="calibrationInputs">Input features from the calibration set.</param>
    /// <param name="calibrationLabels">True class labels from the calibration set.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Calibration computes "conformity scores" for each calibration example.
    ///
    /// The conformity score for an example is typically the predicted probability for the true class.
    /// Higher score = model was more confident in the correct class = more conforming
    ///
    /// These scores are then used to determine prediction sets: classes with scores above a threshold
    /// are included in the prediction set.
    /// </para>
    /// </remarks>
    public void Calibrate(Matrix<T> calibrationInputs, Vector<int> calibrationLabels)
    {
        if (calibrationInputs.Rows != calibrationLabels.Length)
            throw new ArgumentException("Number of inputs must match number of labels");

        var numCalibration = calibrationInputs.Rows;
        if (numCalibration == 0)
            throw new ArgumentException("Calibration set must contain at least one sample.", nameof(calibrationInputs));

        var engine = AiDotNetEngine.Current;
        var labels = Tensor<int>.FromVector(calibrationLabels, new[] { numCalibration, 1 });
        var outOfRange = engine.TensorAdd(engine.TensorLessThan(labels, 0), engine.TensorGreaterThan(labels, _numClasses - 1));
        if (engine.TensorSum(outOfRange) > 0)
        {
            int firstBad = engine.TensorArgMax(engine.Reshape(outOfRange, new[] { numCalibration }), 0).ToArray()[0];
            throw new ArgumentException($"Label {calibrationLabels[firstBad]} is out of range [0, {_numClasses})");
        }

        // Conformity score is the probability assigned to the true class
        var probabilities = PredictRows(calibrationInputs, includeSampleIndex: true);           // [n, C]
        var scores = engine.Reshape(engine.TensorTakeAlongDim(probabilities, labels, 1), new[] { numCalibration });

        // Sort scores (ascending order)
        var (sorted, _) = engine.TensorSort(scores, 0, false);
        _calibrationScores = sorted.ToVector();

        _isCalibrated = true;
    }

    /// <summary>
    /// Predicts with a conformal prediction set.
    /// </summary>
    /// <param name="input">The input features for prediction.</param>
    /// <param name="confidenceLevel">The desired coverage level (e.g., 0.9 for 90%).</param>
    /// <returns>A set of class indices that form the prediction set.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This returns a set of classes that are "plausible" for the input.
    ///
    /// The confidence level determines the size of the set:
    /// - Higher confidence (e.g., 0.95) = Larger prediction sets = More conservative
    /// - Lower confidence (e.g., 0.80) = Smaller prediction sets = More risky
    ///
    /// The guarantee: With 90% confidence, at least 90% of prediction sets will contain the true class.
    ///
    /// Interpreting the results:
    /// - Prediction set = {2}: Model is very confident in class 2
    /// - Prediction set = {2, 5}: Model thinks it's either class 2 or 5
    /// - Prediction set = {0, 1, 2, 3, 4, 5}: Model is very uncertain (might need more data or manual review)
    /// </para>
    /// </remarks>
    public HashSet<int> PredictSet(Tensor<T> input, double confidenceLevel = 0.9)
    {
        if (!_isCalibrated || _calibrationScores == null)
            throw new InvalidOperationException("Must calibrate classifier before making predictions");

        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentException("Confidence level must be between 0 and 1", nameof(confidenceLevel));

        // Get predicted probabilities for all classes
        var probabilities = _model.Predict(input);
        if (probabilities.Length != _numClasses)
            throw new InvalidOperationException($"Model must return exactly {_numClasses} class probabilities. Got {probabilities.Length}.");

        var engine = AiDotNetEngine.Current;
        var threshold = ConformalThreshold(_calibrationScores, confidenceLevel);
        var membership = PredictionSetMask(engine.Reshape(probabilities, new[] { 1, _numClasses }), threshold);
        var classes = engine.TensorMaskedSelect(
            engine.TensorArange<int>(0, _numClasses, 1),
            engine.TensorEqScalar(engine.Reshape(membership, new[] { _numClasses }), _numOps.One));

        return new HashSet<int>(classes.ToArray());
    }

    /// <summary>
    /// 0/1 membership of every class in the prediction set of every row of an [n, C] probability tensor.
    /// </summary>
    /// <remarks>
    /// A class is in the set when its probability is at least the threshold. A row with no such class
    /// gets its top class (the first one on ties), so every set is non-empty.
    /// </remarks>
    private Tensor<T> PredictionSetMask(Tensor<T> probabilities, T threshold)
    {
        var engine = AiDotNetEngine.Current;
        int rows = probabilities._shape[0];

        // p >= threshold, written as (p > t) + (p == t) so a NaN probability stays out of the set.
        var included = engine.TensorAdd(engine.TensorGreaterThan(probabilities, threshold), engine.TensorEquals(probabilities, threshold));
        var setSize = engine.ReduceSum(included, new[] { 1 }, keepDims: true);                    // [n, 1]
        var isEmpty = engine.TensorEquals(setSize, _numOps.Zero);                                 // [n, 1]
        var topClass = engine.TensorOneHot<T>(engine.TensorArgMax(probabilities, 1), _numClasses); // [n, C]

        return engine.TensorAdd(included,
            engine.TensorMultiply(engine.TensorBroadcastTo(isEmpty, new[] { rows, _numClasses }), topClass));
    }

    /// <summary>
    /// Runs the model on every row and returns the class probabilities as an [n, C] tensor.
    /// </summary>
    private Tensor<T> PredictRows(Matrix<T> inputs, bool includeSampleIndex)
    {
        var outputs = new Tensor<T>[inputs.Rows];
        for (int i = 0; i < inputs.Rows; i++)
        {
            var probabilities = _model.Predict(new Tensor<T>([inputs.Columns], inputs.GetRow(i)));
            if (probabilities.Length != _numClasses)
            {
                string where = includeSampleIndex ? $" (sample index {i})" : string.Empty;
                throw new InvalidOperationException($"Model must return exactly {_numClasses} class probabilities. Got {probabilities.Length}{where}.");
            }

            outputs[i] = AiDotNetEngine.Current.Reshape(probabilities, new[] { 1, _numClasses });
        }

        return outputs.Length == 0
            ? new Tensor<T>([0, _numClasses])
            : AiDotNetEngine.Current.TensorConcatenate(outputs, 0);
    }

    /// <summary>
    /// Computes the threshold for prediction set inclusion.
    /// </summary>
    /// <remarks>
    /// Least-ambiguous set-valued classifier (Sadinle, Lei and Wasserman 2019; Angelopoulos and Bates 2021, Sec. 1):
    /// with non-conformity s = 1 - p_y, the threshold is the k-th smallest s, k = ceil((n + 1) * confidenceLevel).
    /// On the ascending conformity scores p that is element n - k (0-based). When k &gt; n no finite threshold
    /// guarantees the coverage, so every class is included (threshold -infinity).
    /// </remarks>
    internal static T ConformalThreshold(Vector<T> sortedConformityScores, double confidenceLevel)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        int n = sortedConformityScores.Length;
        int k = (int)Math.Ceiling((n + 1) * confidenceLevel);
        if (k > n)
            return numOps.FromDouble(double.NegativeInfinity);

        return sortedConformityScores[n - Math.Max(k, 1)];
    }

    /// <summary>
    /// Evaluates the empirical coverage on a test set.
    /// </summary>
    /// <param name="testInputs">Test input features.</param>
    /// <param name="testLabels">True test class labels.</param>
    /// <param name="confidenceLevel">The confidence level used for prediction sets.</param>
    /// <returns>The empirical coverage (proportion of sets containing true class).</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This verifies that your prediction sets achieve the desired coverage.
    /// For 90% confidence, coverage should be ≥ 0.90. If it's lower, check your calibration set.
    /// </remarks>
    public T EvaluateCoverage(Matrix<T> testInputs, Vector<int> testLabels, double confidenceLevel = 0.9)
    {
        if (testLabels.Length != testInputs.Rows)
            throw new ArgumentException(
                $"testLabels has {testLabels.Length} entries but testInputs has {testInputs.Rows} rows.", nameof(testLabels));

        var membership = PredictionSetMasks(testInputs, confidenceLevel, "Must calibrate classifier before evaluation");
        var engine = AiDotNetEngine.Current;
        int totalCount = testInputs.Rows;

        // A label outside [0, C) is never in a set; it is clamped for the lookup and masked out.
        var labels = Tensor<int>.FromVector(testLabels, new[] { testLabels.Length, 1 });
        var valid = engine.ScalarMinusTensor(1, engine.TensorAdd(engine.TensorLessThan(labels, 0), engine.TensorGreaterThan(labels, _numClasses - 1)));
        var inSet = engine.TensorTakeAlongDim(membership, engine.TensorClamp(labels, 0, _numClasses - 1), 1).Cast<int>();
        int coveredCount = engine.TensorSum(engine.TensorMultiply(inSet, valid));

        return _numOps.Divide(_numOps.FromDouble(coveredCount), _numOps.FromDouble(totalCount));
    }

    /// <summary>
    /// Computes the average size of prediction sets on a test set.
    /// </summary>
    /// <param name="testInputs">Test input features.</param>
    /// <param name="confidenceLevel">The confidence level used for prediction sets.</param>
    /// <returns>The average number of classes in prediction sets.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Smaller prediction sets are better (more precise) while maintaining coverage.
    /// - Average size = 1.0: Perfect model (always predicts single correct class)
    /// - Average size = 2-3: Good model with some uncertainty
    /// - Average size close to num_classes: Poor model (very uncertain)
    /// </remarks>
    public T ComputeAverageSetSize(Matrix<T> testInputs, double confidenceLevel = 0.9)
    {
        var membership = PredictionSetMasks(testInputs, confidenceLevel, "Must calibrate classifier before computing set size");
        int totalSize = AiDotNetEngine.Current.TensorSum(membership.Cast<int>());
        return _numOps.Divide(_numOps.FromDouble(totalSize), _numOps.FromDouble(testInputs.Rows));
    }

    private Tensor<T> PredictionSetMasks(Matrix<T> testInputs, double confidenceLevel, string notCalibratedMessage)
    {
        if (!_isCalibrated || _calibrationScores == null)
            throw new InvalidOperationException(notCalibratedMessage);

        if (confidenceLevel <= 0 || confidenceLevel >= 1)
            throw new ArgumentException("Confidence level must be between 0 and 1", nameof(confidenceLevel));

        var threshold = ConformalThreshold(_calibrationScores, confidenceLevel);
        return PredictionSetMask(PredictRows(testInputs, includeSampleIndex: false), threshold);
    }
}
