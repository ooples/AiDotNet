using System.Text;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

using AiDotNet.Models.Parameters;
namespace AiDotNet.Classification.Calibration;

/// <summary>
/// Wrapper that adds probability calibration to any probabilistic classifier.
/// </summary>
/// <remarks>
/// <para>
/// CalibratedClassifier wraps any probabilistic classifier and applies post-hoc probability
/// calibration to improve the reliability of predicted probabilities.
/// </para>
/// <para>
/// <b>For Beginners:</b> Many classifiers give poor probability estimates:
/// - Random Forest tends to push probabilities towards 0.5
/// - SVM's margin-based scores aren't true probabilities
/// - Neural networks without proper training can be overconfident
///
/// This wrapper fixes that by learning to transform raw scores into
/// well-calibrated probabilities that match actual event frequencies.
///
/// Example: If the model says "80% probability", approximately 80% of
/// such predictions should actually be positive.
///
/// Usage:
/// <code>
/// var baseClassifier = new RandomForestClassifier&lt;double&gt;(...);
/// var calibrated = new CalibratedClassifier&lt;double&gt;(
///     baseClassifier,
///     new CalibratedClassifierOptions&lt;double&gt;
///     {
///         CalibrationMethod = ProbabilityCalibrationMethod.IsotonicRegression,
///         CrossValidationFolds = 5
///     });
/// calibrated.Train(X, y);
/// var probs = calibrated.PredictProbabilities(X_test);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Predicting Good Probabilities with Supervised Learning", "https://doi.org/10.1145/1102351.1102430")]
public partial class CalibratedClassifier<T> : ProbabilisticClassifierBase<T>,
    IParameterizable<T, Matrix<T>, Vector<T>>
{

    /// <inheritdoc />
    /// <remarks>All four calibration coefficients -- Platt A and B, then beta A and B. Registering the two Platt scalars by hand missed the beta pair and made a restore throw "Expected 2 parameters, got 4"; keeping the original packing cannot miss anything. PackParameters and UnpackParameters ARE the previous
    /// GetParameters and SetParameters, renamed rather than rewritten, so the layout is
    /// byte-for-byte unchanged. What changes is that the COUNT now folds this same
    /// component instead of ClassifierBase's NumFeatures * NumClasses formula, which
    /// described the classifier's shape rather than what it serializes.</remarks>
    protected override void RegisterComponents()
    {
        RegisterParameterComponent(new VariableLengthParameterSource<T>(
            () => PackParameters().Length,
            PackParameters,
            UnpackParameters));
    }

    /// <summary>
    /// The base classifier being calibrated.
    /// </summary>
    private IProbabilisticClassifier<T> _baseClassifier;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly CalibratedClassifierOptions<T> _options;

    /// <summary>
    /// Platt scaling parameters (A, B) for sigmoid calibration.
    /// P_calibrated = 1 / (1 + exp(-(A * logit(P) + B)))
    /// </summary>
    private T _plattA;
    private T _plattB;

    /// <summary>
    /// Beta calibration parameters (a, b, c).
    /// </summary>
    private T _betaA;
    private T _betaB;
    private T _betaC;

    /// <summary>
    /// Temperature scaling parameter.
    /// </summary>
    private T _temperature;

    /// <summary>
    /// Isotonic regression mapping (sorted by probability).
    /// </summary>
    private (T prob, T calibrated)[]? _isotonicMapping;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Whether the model has been trained.
    /// </summary>
    private bool _isTrained;

    /// <summary>
    /// Initializes a new instance with default settings using Gaussian Naive Bayes as the base classifier.
    /// </summary>
    public CalibratedClassifier()
        : this(new AiDotNet.Classification.NaiveBayes.GaussianNaiveBayes<T>())
    {
    }

    /// <summary>
    /// Initializes a new CalibratedClassifier.
    /// </summary>
    /// <param name="baseClassifier">The probabilistic classifier to calibrate.</param>
    /// <param name="options">Calibration configuration options.</param>
    /// <param name="regularization">Optional regularization.</param>
    public CalibratedClassifier(
        IProbabilisticClassifier<T> baseClassifier,
        CalibratedClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ??= new CalibratedClassifierOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
        _baseClassifier = baseClassifier;
        _options = options;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
        _isTrained = false;
        _plattA = NumOps.One;
        _plattB = NumOps.Zero;
        _betaA = NumOps.One;
        _betaB = NumOps.One;
        _betaC = NumOps.Zero;
        _temperature = NumOps.One;
    }

    /// <summary>
    /// Gets the base classifier.
    /// </summary>
    public IProbabilisticClassifier<T> BaseClassifier => _baseClassifier;

    /// <summary>
    /// Gets the calibration method.
    /// </summary>
    public ProbabilityCalibrationMethod CalibrationMethod => _options.CalibrationMethod;

    /// <summary>
    /// Gets whether the model is trained.
    /// </summary>
    public bool IsTrained => _isTrained;

    /// <summary>
    /// Trains the base classifier and fits calibration.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        // Extract class labels
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        NumFeatures = x.Columns;
        TaskType = InferTaskType(y);

        if (_options.CalibrationMethod == ProbabilityCalibrationMethod.None
            || _options.CalibrationMethod == ProbabilityCalibrationMethod.Auto)
        {
            // No calibration - just train base
            _baseClassifier.Train(x, y);
            _isTrained = true;
            return;
        }

        int n = x.Rows;

        if (_options.CrossValidationFolds > 1 && n >= _options.CrossValidationFolds * 2)
        {
            // K-fold cross-validation approach
            TrainWithCrossValidation(x, y);
        }
        else
        {
            // Simple holdout approach
            int calibSize = Math.Max(2, (int)(n * _options.CalibrationSetFraction));
            int trainSize = n - calibSize;

            // Create shuffled indices
            var indices = Enumerable.Range(0, n).ToArray();
            for (int i = n - 1; i > 0; i--)
            {
                int j = _random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Split data
            var trainIndices = new int[trainSize];
            var calibIndices = new int[calibSize];
            Array.Copy(indices, 0, trainIndices, 0, trainSize);
            Array.Copy(indices, trainSize, calibIndices, 0, calibSize);
            var trainX = x.GetRows(trainIndices);
            var trainY = GatherElements(y, trainIndices);
            var calibX = x.GetRows(calibIndices);
            var calibY = GatherElements(y, calibIndices);

            // Train base classifier
            _baseClassifier.Train(trainX, trainY);

            // Get uncalibrated predictions on calibration set
            var uncalibrated = _baseClassifier.PredictProbabilities(calibX);

            // Fit calibration
            FitCalibration(uncalibrated, calibY);
        }

        // Retrain base classifier on all data for final model
        _baseClassifier.Train(x, y);
        _isTrained = true;
    }

    /// <summary>
    /// Trains with cross-validation to use all data for calibration.
    /// </summary>
    private void TrainWithCrossValidation(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int k = _options.CrossValidationFolds;

        // Create shuffled indices
        var indices = Enumerable.Range(0, n).ToArray();
        for (int i = n - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Collect out-of-fold predictions; fold blocks are stacked in shuffled order and un-shuffled at the end
        var foldPredictions = new Tensor<T>[k];
        int foldSize = n / k;

        for (int fold = 0; fold < k; fold++)
        {
            int foldStart = fold * foldSize;
            int foldEnd = (fold == k - 1) ? n : (fold + 1) * foldSize;
            int testSize = foldEnd - foldStart;
            int trainSize = n - testSize;

            var testIndices = new int[testSize];
            var trainIndices = new int[trainSize];
            Array.Copy(indices, foldStart, testIndices, 0, testSize);
            Array.Copy(indices, 0, trainIndices, 0, foldStart);
            Array.Copy(indices, foldEnd, trainIndices, foldStart, n - foldEnd);

            var trainX = x.GetRows(trainIndices);
            var trainY = GatherElements(y, trainIndices);
            var testX = x.GetRows(testIndices);

            // Clone and train on this fold's training data
            IFullModel<T, Matrix<T>, Vector<T>> foldModel;
            if (_baseClassifier is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
            {
                foldModel = fullModel.Clone();
            }
            else
            {
                throw new InvalidOperationException("Base classifier must implement IFullModel for cross-validation.");
            }

            foldModel.Train(trainX, trainY);

            // Get predictions on test fold
            var foldPreds = ((IProbabilisticClassifier<T>)foldModel).PredictProbabilities(testX);
            foldPredictions[fold] = Tensor<T>.FromMatrix(foldPreds.GetSubMatrix(0, 0, testSize, NumClasses));
        }

        // Row r of the stacked folds belongs to sample indices[r]; invert the shuffle to restore sample order
        var inverse = new int[n];
        for (int r = 0; r < n; r++)
        {
            inverse[indices[r]] = r;
        }

        var stacked = Engine.TensorConcatenate(foldPredictions, 0);
        var allPredictions = Engine.TensorIndexSelect(stacked, new Tensor<int>(inverse, new[] { n }), 0).ToMatrix();

        // Fit calibration on all out-of-fold predictions
        FitCalibration(allPredictions, y);
    }

    /// <summary>
    /// Fits the calibration model to uncalibrated predictions.
    /// </summary>
    private void FitCalibration(Matrix<T> uncalibrated, Vector<T> actuals)
    {
        int n = uncalibrated.Rows;

        if (NumClasses != 2)
        {
            throw new NotSupportedException(
                "Per-class calibration is not implemented. Use binary classification or add per-class parameters.");
        }

        // Use last class as the "positive" class for calibration
        int positiveClassIdx = NumClasses - 1;
        var classLabels = ClassLabels ?? throw new InvalidOperationException("Class labels not initialized.");
        var positiveLabel = classLabels[positiveClassIdx];

        // For binary classification, use probability of positive class (last column)
        var probs = n == 0 ? new Vector<T>(0) : uncalibrated.GetColumn(positiveClassIdx);
        var targets = n == 0 ? new Vector<T>(0) : Engine.TensorEquals(actuals.AsTensor(), positiveLabel).ToVector();

        switch (_options.CalibrationMethod)
        {
            case ProbabilityCalibrationMethod.PlattScaling:
                FitPlattScaling(probs, targets);
                break;
            case ProbabilityCalibrationMethod.IsotonicRegression:
                FitIsotonicRegression(probs, targets);
                break;
            case ProbabilityCalibrationMethod.BetaCalibration:
                FitBetaCalibration(probs, targets);
                break;
            case ProbabilityCalibrationMethod.TemperatureScaling:
                FitTemperatureScaling(probs, targets);
                break;
            case ProbabilityCalibrationMethod.Auto:
            case ProbabilityCalibrationMethod.None:
                // No calibration
                break;
        }
    }

    /// <summary>
    /// Fits Platt scaling (sigmoid calibration).
    /// </summary>
    private void FitPlattScaling(Vector<T> probs, Vector<T> targets)
    {
        // Fit sigmoid: P_calibrated = 1 / (1 + exp(-(A * logit(P) + B))) by gradient descent on the
        // design matrix [logit(P), 1] with parameters [A, B].
        var logit = Logit(ClampProbabilities(probs.AsTensor()));
        var design = StackColumns(new[] { logit, FilledVector(probs.Length, NumOps.One) });
        var parameters = FitLogisticByGradientDescent(design, targets, new Vector<T>(2));

        _plattA = parameters[0];
        _plattB = parameters[1];
    }

    /// <summary>
    /// Fits isotonic regression calibration using PAVA.
    /// </summary>
    private void FitIsotonicRegression(Vector<T> probs, Vector<T> targets)
    {
        int n = probs.Length;

        // Sort by predicted probability
        var indexed = new (T prob, T target)[n];
        for (int i = 0; i < n; i++)
        {
            indexed[i] = (probs[i], targets[i]);
        }
        Array.Sort(indexed, (a, b) => NumOps.Compare(a.prob, b.prob));

        // Pool Adjacent Violators Algorithm (PAVA) - block-based implementation
        var blockValues = new Vector<T>(n);
        var blockWeights = new Vector<T>(n);
        var blockEnds = new int[n];
        int numBlocks = n;

        for (int i = 0; i < n; i++)
        {
            blockValues[i] = indexed[i].target;
            blockWeights[i] = NumOps.One;
            blockEnds[i] = i;
        }

        // Forward pass: merge violating adjacent blocks
        int current = 0;
        var blockStarts = new List<int> { 0 };

        while (current < n - 1)
        {
            int next = blockEnds[current] + 1;
            if (next >= n) break;

            if (NumOps.GreaterThan(blockValues[current], blockValues[next]))
            {
                // Merge current and next blocks
                T totalWeight = NumOps.Add(blockWeights[current], blockWeights[next]);
                blockValues[current] = NumOps.Divide(
                    NumOps.Add(
                        NumOps.Multiply(blockValues[current], blockWeights[current]),
                        NumOps.Multiply(blockValues[next], blockWeights[next])),
                    totalWeight);
                blockWeights[current] = totalWeight;
                blockEnds[current] = blockEnds[next];

                // Check if we need to merge backwards
                while (blockStarts.Count > 1)
                {
                    int prev = blockStarts[^2];
                    if (NumOps.GreaterThan(blockValues[prev], blockValues[current]))
                    {
                        T tw = NumOps.Add(blockWeights[prev], blockWeights[current]);
                        blockValues[prev] = NumOps.Divide(
                            NumOps.Add(
                                NumOps.Multiply(blockValues[prev], blockWeights[prev]),
                                NumOps.Multiply(blockValues[current], blockWeights[current])),
                            tw);
                        blockWeights[prev] = tw;
                        blockEnds[prev] = blockEnds[current];
                        blockStarts.RemoveAt(blockStarts.Count - 1);
                        current = prev;
                    }
                    else
                    {
                        break;
                    }
                }
            }
            else
            {
                current = next;
                blockStarts.Add(current);
            }
        }

        // Expand blocks to per-element calibrated values
        var calibrated = new Vector<T>(n);
        foreach (int start in blockStarts)
        {
            int end = blockEnds[start];
            for (int i = start; i <= end; i++)
            {
                calibrated[i] = blockValues[start];
            }
        }

        // Store mapping for prediction (remove duplicates, keep unique probability bins)
        var uniqueMapping = new List<(T prob, T calibrated)>();
        T lastProb = NumOps.MinValue;
        T threshold = Tolerance(1e-10);
        for (int i = 0; i < n; i++)
        {
            if (NumOps.GreaterThan(NumOps.Subtract(indexed[i].prob, lastProb), threshold))
            {
                uniqueMapping.Add((indexed[i].prob, calibrated[i]));
                lastProb = indexed[i].prob;
            }
        }
        _isotonicMapping = uniqueMapping.ToArray();
    }

    /// <summary>
    /// Fits beta calibration.
    /// </summary>
    private void FitBetaCalibration(Vector<T> probs, Vector<T> targets)
    {
        // Beta calibration: P_calibrated = sigmoid(a * log(P) - b * log(1-P) + c), fitted by gradient descent
        // on the design matrix [log(P), -log(1-P), 1] with parameters [a, b, c].
        var engine = Engine;
        var p = ClampProbabilities(probs.AsTensor());
        var logP = engine.TensorLog(p).ToVector();
        var negLog1mP = engine.TensorNegate(engine.TensorLog(engine.ScalarMinusTensor(NumOps.One, p))).ToVector();
        var design = StackColumns(new[] { logP, negLog1mP, FilledVector(probs.Length, NumOps.One) });
        var initial = Vector<T>.Concatenate(FilledVector(2, NumOps.One), new Vector<T>(1));
        var parameters = FitLogisticByGradientDescent(design, targets, initial);

        _betaA = parameters[0];
        _betaB = parameters[1];
        _betaC = parameters[2];
    }

    /// <summary>
    /// Fits temperature scaling.
    /// </summary>
    private void FitTemperatureScaling(Vector<T> probs, Vector<T> targets)
    {
        // Find temperature T that minimizes NLL via grid search over T = 0.1, 0.15, ..., 9.95 (or up to 10).
        int gridSize = 0;
        for (double tVal = 0.1; tVal <= 10.0; tVal += 0.05)
        {
            gridSize++;
        }

        int n = probs.Length;
        if (n == 0 || gridSize == 0)
        {
            _temperature = NumOps.One;
            return;
        }

        // Every grid temperature is evaluated at once: z[i, g] = logit(p_i) / T_g
        var engine = Engine;
        var temperatures = engine.TensorArange<T>(0.1, 0.1 + (gridSize - 0.5) * 0.05, 0.05);
        var logit = Logit(ClampProbabilities(probs.AsTensor()));
        var z = engine.TensorDivide(logit.AsTensor().Reshape(n, 1), temperatures.Reshape(1, gridSize));
        var calibrated = ClampProbabilities(engine.Sigmoid(z));

        // NLL per temperature: -sum_i (y_i log(q) + (1 - y_i) log(1 - q))
        var yColumn = targets.AsTensor().Reshape(n, 1);
        var oneMinusY = engine.ScalarMinusTensor(NumOps.One, yColumn);
        var perSample = engine.TensorAdd(
            engine.TensorMultiply(engine.TensorLog(calibrated), yColumn),
            engine.TensorMultiply(engine.TensorLog(engine.ScalarMinusTensor(NumOps.One, calibrated)), oneMinusY));
        var losses = engine.TensorNegate(engine.ReduceSum(perSample, new[] { 0 }, false));

        // First temperature with the smallest loss
        int best = engine.TensorArgMin(losses, 0).ToArray()[0];
        _temperature = engine.TensorGather(temperatures, new Tensor<int>(new[] { best }, new[] { 1 }), 0).ToArray()[0];
    }

    /// <summary>
    /// Clamps probabilities to [1e-10, 1 - 1e-10] (moved inward where float or Half cannot hold the bounds) so
    /// logarithms stay finite.
    /// </summary>
    private Tensor<T> ClampProbabilities(Tensor<T> probabilities)
    {
        // 1 - 1e-10 rounds to 1 in float and Half, which made logit(p) and log(1 - p) infinite for p = 1
        return Engine.TensorClamp(probabilities, Tolerance(1e-10), OneMinusTolerance(1e-10));
    }

    /// <summary>
    /// logit(p) = log(p / (1 - p)).
    /// </summary>
    private Vector<T> Logit(Tensor<T> p)
    {
        var engine = Engine;
        return engine.TensorLog(engine.TensorDivide(p, engine.ScalarMinusTensor(NumOps.One, p))).ToVector();
    }

    /// <summary>
    /// Fits sigmoid(design * w) to the 0/1 targets by full-batch gradient descent on the mean cross-entropy
    /// (learning rate 0.01, at most 1000 iterations, stopping when the loss changes by less than 1e-8).
    /// </summary>
    private Vector<T> FitLogisticByGradientDescent(Matrix<T> design, Vector<T> targets, Vector<T> initial)
    {
        var engine = Engine;
        var weights = initial;
        T step = NumOps.Divide(NumOps.FromDouble(0.01), NumOps.FromDouble(targets.Length));
        T prevLoss = NumOps.MaxValue;
        var designT = engine.MatrixTranspose(design);
        var targetsT = targets.AsTensor();
        var oneMinusTargets = engine.ScalarMinusTensor(NumOps.One, targetsT);

        for (int iter = 0; iter < 1000; iter++)
        {
            var predicted = engine.Sigmoid(engine.MatrixVectorMultiply(design, weights));
            var error = engine.Subtract(predicted, targets);
            var gradient = engine.MatrixVectorMultiply(designT, error);

            // Cross-entropy loss on the clamped predictions
            var q = ClampProbabilities(predicted.AsTensor());
            var perSample = engine.TensorAdd(
                engine.TensorMultiply(targetsT, engine.TensorLog(q)),
                engine.TensorMultiply(oneMinusTargets, engine.TensorLog(engine.ScalarMinusTensor(NumOps.One, q))));
            T loss = engine.TensorSum(engine.TensorNegate(perSample));

            weights = engine.Subtract(weights, engine.Multiply(gradient, step));

            // Check convergence
            if (HasConverged(loss, ref prevLoss))
            {
                break;
            }
        }

        return weights;
    }

    /// <summary>
    /// True when |loss - previous| &lt; 1e-8; otherwise records the loss as the new previous value.
    /// </summary>
    private bool HasConverged(T loss, ref T previous)
    {
        if (NumOps.LessThan(NumOps.Abs(NumOps.Subtract(loss, previous)), Tolerance(1e-8)))
        {
            return true;
        }

        previous = loss;
        return false;
    }

    /// <summary>
    /// Selects the elements of <paramref name="values"/> at <paramref name="indices"/>.
    /// </summary>
    private Vector<T> GatherElements(Vector<T> values, int[] indices)
    {
        if (indices.Length == 0)
        {
            return new Vector<T>(0);
        }

        return Engine.TensorGather(values.AsTensor(), new Tensor<int>(indices, new[] { indices.Length }), 0).ToVector();
    }

    /// <summary>
    /// Gets calibrated probability predictions.
    /// </summary>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (!_isTrained)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        var uncalibrated = _baseClassifier.PredictProbabilities(input);

        if (_options.CalibrationMethod == ProbabilityCalibrationMethod.None
            || _options.CalibrationMethod == ProbabilityCalibrationMethod.Auto)
        {
            // No calibration, return raw probabilities
            return uncalibrated;
        }

        if (NumClasses != 2)
        {
            throw new NotSupportedException(
                "Calibration currently supports binary classification only. " +
                "For multiclass, apply one-vs-rest calibration externally.");
        }

        if (input.Rows == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        // Binary classification: calibrate positive class
        var calibP = CalibrateProbabilities(uncalibrated.GetColumn(1));
        var calibN = Engine.ScalarMinusTensor(NumOps.One, calibP.AsTensor()).ToVector();
        return StackColumns(new[] { calibN, calibP });
    }

    /// <summary>
    /// Applies the fitted calibration to every probability in <paramref name="probabilities"/>.
    /// </summary>
    private Vector<T> CalibrateProbabilities(Vector<T> probabilities)
    {
        var engine = Engine;
        var p = ClampProbabilities(probabilities.AsTensor());

        switch (_options.CalibrationMethod)
        {
            case ProbabilityCalibrationMethod.PlattScaling:
                return engine.Sigmoid(engine.TensorAddScalar(
                    engine.TensorMultiplyScalar(Logit(p).AsTensor(), _plattA), _plattB)).ToVector();

            case ProbabilityCalibrationMethod.IsotonicRegression:
                return InterpolateIsotonic(p);

            case ProbabilityCalibrationMethod.BetaCalibration:
                var logP = engine.TensorMultiplyScalar(engine.TensorLog(p), _betaA);
                var log1mP = engine.TensorMultiplyScalar(
                    engine.TensorLog(engine.ScalarMinusTensor(NumOps.One, p)), _betaB);
                return engine.Sigmoid(engine.TensorAddScalar(engine.TensorSubtract(logP, log1mP), _betaC)).ToVector();

            case ProbabilityCalibrationMethod.TemperatureScaling:
                return engine.Sigmoid(engine.TensorDivideScalar(Logit(p).AsTensor(), _temperature)).ToVector();

            default:
                return p.ToVector();
        }
    }

    /// <summary>
    /// Interpolates isotonic regression mapping.
    /// </summary>
    private Vector<T> InterpolateIsotonic(Tensor<T> p)
    {
        if (_isotonicMapping == null || _isotonicMapping.Length == 0)
            return p.ToVector();

        var engine = Engine;
        int m = _isotonicMapping.Length;
        var knotProbs = new T[m];
        var knotValues = new T[m];
        for (int k = 0; k < m; k++)
        {
            (knotProbs[k], knotValues[k]) = _isotonicMapping[k];
        }

        var xs = new Tensor<T>(knotProbs, new[] { m });
        var ys = new Tensor<T>(knotValues, new[] { m });
        if (m == 1)
        {
            return FilledVector(p.Length, knotValues[0]);
        }

        // Bracketing knots: low = last knot with prob <= p (clamped so that low + 1 exists)
        var low = engine.TensorClamp(
            engine.TensorAddScalar(engine.TensorSearchSorted(xs, p, true), -1), 0, m - 2);
        var high = engine.TensorAddScalar(low, 1);
        var xLow = engine.TensorGather(xs, low, 0);
        var xHigh = engine.TensorGather(xs, high, 0);
        var yLow = engine.TensorGather(ys, low, 0);
        var yHigh = engine.TensorGather(ys, high, 0);

        // Linear interpolation (falls back to the low knot when the bracket is narrower than 1e-10)
        var range = engine.TensorSubtract(xHigh, xLow);
        var narrow = engine.TensorLessThan(range, Tolerance(1e-10));
        var safeRange = engine.TensorWhere(narrow, FilledVector(p.Length, NumOps.One).AsTensor(), range);
        var t = engine.TensorDivide(engine.TensorSubtract(p, xLow), safeRange);
        var interpolated = engine.TensorAdd(
            engine.TensorMultiply(yLow, engine.ScalarMinusTensor(NumOps.One, t)),
            engine.TensorMultiply(yHigh, t));
        var result = engine.TensorWhere(narrow, yLow, interpolated);

        // Boundary cases: at or below the first knot / at or above the last knot
        var belowFirst = engine.TensorLessEqual(p, FilledVector(p.Length, knotProbs[0]).AsTensor());
        var aboveLast = engine.TensorGreaterEqual(p, FilledVector(p.Length, knotProbs[m - 1]).AsTensor());
        result = engine.TensorWhere(belowFirst, FilledVector(p.Length, knotValues[0]).AsTensor(), result);
        result = engine.TensorWhere(aboveLast, FilledVector(p.Length, knotValues[m - 1]).AsTensor(), result);
        return result.ToVector();
    }

    /// <inheritdoc/>
    private Vector<T> PackParameters()
    {
        // Return calibration parameters
        return new Vector<T>(new[]
        {
            _plattA,
            _plattB,
            _betaA,
            _betaB,
            _betaC,
            _temperature
        });
    }

    /// <inheritdoc/>
    private void UnpackParameters(Vector<T> parameters)
    {
        if (parameters.Length < 6)
        {
            // Accept silently for untrained models — calibration parameters come from training
            return;
        }

        _plattA = parameters[0];
        _plattB = parameters[1];
        _betaA = parameters[2];
        _betaB = parameters[3];
        _betaC = parameters[4];
        _temperature = parameters[5];
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        var newModel = Clone();
        if (newModel is CalibratedClassifier<T> calibrated)
        {
            calibrated.SetParameters(parameters);
        }
        return newModel;
    }

    private static T DeserializeValue(JToken? token, T defaultValue)
    {
        if (token is null || token.Type == JTokenType.Null)
            return defaultValue;

        var result = token.ToObject<T>();
        return result is not null ? result : defaultValue;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["CalibrationMethod"] = _options.CalibrationMethod.ToString();
        metadata.AdditionalInfo["CrossValidationFolds"] = _options.CrossValidationFolds;
        metadata.AdditionalInfo["PlattA"] = NumOps.ToDouble(_plattA);
        metadata.AdditionalInfo["PlattB"] = NumOps.ToDouble(_plattB);
        metadata.AdditionalInfo["BetaA"] = NumOps.ToDouble(_betaA);
        metadata.AdditionalInfo["BetaB"] = NumOps.ToDouble(_betaB);
        metadata.AdditionalInfo["BetaC"] = NumOps.ToDouble(_betaC);
        metadata.AdditionalInfo["Temperature"] = NumOps.ToDouble(_temperature);
        return metadata;
    }
}
