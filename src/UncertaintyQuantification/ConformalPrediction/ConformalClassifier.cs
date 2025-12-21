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
        _model = model ?? throw new ArgumentNullException(nameof(model));
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
        _calibrationScores = new Vector<T>(numCalibration);

        // Compute conformity scores (probability of true class)
        for (int i = 0; i < numCalibration; i++)
        {
            var input = new Tensor<T>([calibrationInputs.Columns], calibrationInputs.GetRow(i));
            var probabilities = _model.Predict(input);
            if (probabilities.Length != _numClasses)
                throw new InvalidOperationException($"Model must return exactly {_numClasses} class probabilities. Got {probabilities.Length} (sample index {i}).");

            var trueLabel = calibrationLabels[i];
            if (trueLabel < 0 || trueLabel >= _numClasses)
                throw new ArgumentException($"Label {trueLabel} is out of range [0, {_numClasses})");

            // Conformity score is the probability assigned to the true class
            _calibrationScores[i] = probabilities[trueLabel];
        }

        // Sort scores (ascending order)
        var scoresArray = _calibrationScores.ToArray();
        Array.Sort(scoresArray);
        _calibrationScores = new Vector<T>(scoresArray);

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

        // Compute the threshold score
        var threshold = ComputeThreshold(confidenceLevel);

        // Build prediction set: include classes with probability >= threshold
        var predictionSet = new HashSet<int>();
        for (int c = 0; c < _numClasses; c++)
        {
            if (_numOps.GreaterThanOrEquals(probabilities[c], threshold))
            {
                predictionSet.Add(c);
            }
        }

        // Ensure prediction set is non-empty (include at least the top predicted class)
        if (predictionSet.Count == 0)
        {
            var maxProb = probabilities[0];
            var maxClass = 0;
            for (int c = 1; c < _numClasses; c++)
            {
                if (_numOps.GreaterThan(probabilities[c], maxProb))
                {
                    maxProb = probabilities[c];
                    maxClass = c;
                }
            }
            predictionSet.Add(maxClass);
        }

        return predictionSet;
    }

    /// <summary>
    /// Computes the threshold for prediction set inclusion.
    /// </summary>
    private T ComputeThreshold(double confidenceLevel)
    {
        var n = _calibrationScores!.Length;

        // Compute quantile index (using floor for conservative estimates)
        var quantileLevel = 1.0 - confidenceLevel;
        var index = (int)Math.Floor(n * quantileLevel);

        if (index < 0) index = 0;
        if (index >= n) index = n - 1;

        return _calibrationScores[index];
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
    public double EvaluateCoverage(Matrix<T> testInputs, Vector<int> testLabels, double confidenceLevel = 0.9)
    {
        if (!_isCalibrated)
            throw new InvalidOperationException("Must calibrate classifier before evaluation");

        int coveredCount = 0;
        int totalCount = testInputs.Rows;

        for (int i = 0; i < totalCount; i++)
        {
            var input = new Tensor<T>([testInputs.Columns], testInputs.GetRow(i));
            var predictionSet = PredictSet(input, confidenceLevel);

            if (predictionSet.Contains(testLabels[i]))
            {
                coveredCount++;
            }
        }

        return (double)coveredCount / totalCount;
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
    public double ComputeAverageSetSize(Matrix<T> testInputs, double confidenceLevel = 0.9)
    {
        if (!_isCalibrated)
            throw new InvalidOperationException("Must calibrate classifier before computing set size");

        int totalSize = 0;
        int count = testInputs.Rows;

        for (int i = 0; i < count; i++)
        {
            var input = new Tensor<T>([testInputs.Columns], testInputs.GetRow(i));
            var predictionSet = PredictSet(input, confidenceLevel);
            totalSize += predictionSet.Count;
        }

        return (double)totalSize / count;
    }
}
