using System.Text;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Validation;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

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
public class CalibratedClassifier<T> : ProbabilisticClassifierBase<T>
{
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
        : base(options ??= new CalibratedClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
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
            var trainX = new Matrix<T>(trainSize, x.Columns);
            var trainY = new Vector<T>(trainSize);
            var calibX = new Matrix<T>(calibSize, x.Columns);
            var calibY = new Vector<T>(calibSize);

            for (int i = 0; i < trainSize; i++)
            {
                for (int j = 0; j < x.Columns; j++)
                    trainX[i, j] = x[indices[i], j];
                trainY[i] = y[indices[i]];
            }
            for (int i = 0; i < calibSize; i++)
            {
                for (int j = 0; j < x.Columns; j++)
                    calibX[i, j] = x[indices[trainSize + i], j];
                calibY[i] = y[indices[trainSize + i]];
            }

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

        // Collect out-of-fold predictions
        var allPredictions = new Matrix<T>(n, NumClasses);
        int foldSize = n / k;

        for (int fold = 0; fold < k; fold++)
        {
            int foldStart = fold * foldSize;
            int foldEnd = (fold == k - 1) ? n : (fold + 1) * foldSize;
            int testSize = foldEnd - foldStart;
            int trainSize = n - testSize;

            var trainX = new Matrix<T>(trainSize, x.Columns);
            var trainY = new Vector<T>(trainSize);
            var testX = new Matrix<T>(testSize, x.Columns);

            int trainIdx = 0;
            for (int i = 0; i < n; i++)
            {
                if (i >= foldStart && i < foldEnd)
                {
                    int testIdx = i - foldStart;
                    for (int j = 0; j < x.Columns; j++)
                        testX[testIdx, j] = x[indices[i], j];
                }
                else
                {
                    for (int j = 0; j < x.Columns; j++)
                        trainX[trainIdx, j] = x[indices[i], j];
                    trainY[trainIdx] = y[indices[i]];
                    trainIdx++;
                }
            }

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
            for (int i = foldStart; i < foldEnd; i++)
            {
                for (int c = 0; c < NumClasses; c++)
                    allPredictions[indices[i], c] = foldPreds[i - foldStart, c];
            }
        }

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

        // For binary classification, use probability of positive class (last column)
        var probs = new Vector<T>(n);
        var targets = new Vector<T>(n);

        // Use last class as the "positive" class for calibration
        int positiveClassIdx = NumClasses - 1;
        var classLabels = ClassLabels ?? throw new InvalidOperationException("Class labels not initialized.");
        var positiveLabel = classLabels[positiveClassIdx];

        for (int i = 0; i < n; i++)
        {
            probs[i] = uncalibrated[i, positiveClassIdx];
            targets[i] = NumOps.Compare(actuals[i], positiveLabel) == 0 ? NumOps.One : NumOps.Zero;
        }

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
        // Fit sigmoid: P_calibrated = 1 / (1 + exp(-(A * logit(P) + B)))
        T a = NumOps.Zero;
        T b = NumOps.Zero;
        T lr = NumOps.FromDouble(0.01);
        int maxIter = 1000;
        T tolerance = NumOps.FromDouble(1e-8);
        T prevLoss = NumOps.MaxValue;
        T eps = NumOps.FromDouble(1e-10);
        T oneMinusEps = NumOps.Subtract(NumOps.One, eps);
        T nT = NumOps.FromDouble(probs.Length);

        for (int iter = 0; iter < maxIter; iter++)
        {
            T gradA = NumOps.Zero, gradB = NumOps.Zero;
            T loss = NumOps.Zero;

            for (int i = 0; i < probs.Length; i++)
            {
                // Clamp probability to avoid log(0)
                T p = probs[i];
                if (NumOps.LessThan(p, eps)) p = eps;
                if (NumOps.GreaterThan(p, oneMinusEps)) p = oneMinusEps;
                T logit = NumOps.Log(NumOps.Divide(p, NumOps.Subtract(NumOps.One, p)));

                T z = NumOps.Add(NumOps.Multiply(a, logit), b);
                T sigmoid = SigmoidT(z);

                T error = NumOps.Subtract(sigmoid, targets[i]);
                gradA = NumOps.Add(gradA, NumOps.Multiply(error, logit));
                gradB = NumOps.Add(gradB, error);

                // Cross-entropy loss
                T clampedSigmoid = sigmoid;
                if (NumOps.LessThan(clampedSigmoid, eps)) clampedSigmoid = eps;
                if (NumOps.GreaterThan(clampedSigmoid, oneMinusEps)) clampedSigmoid = oneMinusEps;
                loss = NumOps.Subtract(loss,
                    NumOps.Add(
                        NumOps.Multiply(targets[i], NumOps.Log(clampedSigmoid)),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, targets[i]),
                            NumOps.Log(NumOps.Subtract(NumOps.One, clampedSigmoid)))));
            }

            a = NumOps.Subtract(a, NumOps.Multiply(lr, NumOps.Divide(gradA, nT)));
            b = NumOps.Subtract(b, NumOps.Multiply(lr, NumOps.Divide(gradB, nT)));

            // Check convergence
            T lossDiff = NumOps.Abs(NumOps.Subtract(loss, prevLoss));
            if (NumOps.LessThan(lossDiff, tolerance))
            {
                break;
            }
            prevLoss = loss;
        }

        _plattA = a;
        _plattB = b;
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
        T threshold = NumOps.FromDouble(1e-10);
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
        // Beta calibration: P_calibrated = sigmoid(a * log(P) - b * log(1-P) + c)
        T a = NumOps.One, b = NumOps.One, c = NumOps.Zero;
        T lr = NumOps.FromDouble(0.01);
        int maxIter = 1000;
        T tolerance = NumOps.FromDouble(1e-8);
        T prevLoss = NumOps.MaxValue;
        T eps = NumOps.FromDouble(1e-10);
        T oneMinusEps = NumOps.Subtract(NumOps.One, eps);
        T nT = NumOps.FromDouble(probs.Length);

        for (int iter = 0; iter < maxIter; iter++)
        {
            T gradA = NumOps.Zero, gradB = NumOps.Zero, gradC = NumOps.Zero;
            T loss = NumOps.Zero;

            for (int i = 0; i < probs.Length; i++)
            {
                T p = probs[i];
                if (NumOps.LessThan(p, eps)) p = eps;
                if (NumOps.GreaterThan(p, oneMinusEps)) p = oneMinusEps;

                T logP = NumOps.Log(p);
                T log1mP = NumOps.Log(NumOps.Subtract(NumOps.One, p));
                T z = NumOps.Add(NumOps.Subtract(NumOps.Multiply(a, logP),
                    NumOps.Multiply(b, log1mP)), c);
                T calibratedP = SigmoidT(z);

                T error = NumOps.Subtract(calibratedP, targets[i]);

                gradA = NumOps.Add(gradA, NumOps.Multiply(error, logP));
                gradB = NumOps.Add(gradB, NumOps.Multiply(error, NumOps.Negate(log1mP)));
                gradC = NumOps.Add(gradC, error);

                // Cross-entropy loss
                T clampedCalib = calibratedP;
                if (NumOps.LessThan(clampedCalib, eps)) clampedCalib = eps;
                if (NumOps.GreaterThan(clampedCalib, oneMinusEps)) clampedCalib = oneMinusEps;
                loss = NumOps.Subtract(loss,
                    NumOps.Add(
                        NumOps.Multiply(targets[i], NumOps.Log(clampedCalib)),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, targets[i]),
                            NumOps.Log(NumOps.Subtract(NumOps.One, clampedCalib)))));
            }

            a = NumOps.Subtract(a, NumOps.Multiply(lr, NumOps.Divide(gradA, nT)));
            b = NumOps.Subtract(b, NumOps.Multiply(lr, NumOps.Divide(gradB, nT)));
            c = NumOps.Subtract(c, NumOps.Multiply(lr, NumOps.Divide(gradC, nT)));

            T lossDiff = NumOps.Abs(NumOps.Subtract(loss, prevLoss));
            if (NumOps.LessThan(lossDiff, tolerance))
            {
                break;
            }
            prevLoss = loss;
        }

        _betaA = a;
        _betaB = b;
        _betaC = c;
    }

    /// <summary>
    /// Fits temperature scaling.
    /// </summary>
    private void FitTemperatureScaling(Vector<T> probs, Vector<T> targets)
    {
        // Find temperature T that minimizes NLL via grid search
        T bestTemp = NumOps.One;
        T bestLoss = NumOps.MaxValue;
        T eps = NumOps.FromDouble(1e-10);
        T oneMinusEps = NumOps.Subtract(NumOps.One, eps);

        // Grid search over temperature
        for (double tVal = 0.1; tVal <= 10.0; tVal += 0.05)
        {
            T t = NumOps.FromDouble(tVal);
            T loss = NumOps.Zero;
            for (int i = 0; i < probs.Length; i++)
            {
                T p = probs[i];
                if (NumOps.LessThan(p, eps)) p = eps;
                if (NumOps.GreaterThan(p, oneMinusEps)) p = oneMinusEps;
                T logit = NumOps.Log(NumOps.Divide(p, NumOps.Subtract(NumOps.One, p)));
                T calibratedP = SigmoidT(NumOps.Divide(logit, t));

                T clampedCalib = calibratedP;
                if (NumOps.LessThan(clampedCalib, eps)) clampedCalib = eps;
                if (NumOps.GreaterThan(clampedCalib, oneMinusEps)) clampedCalib = oneMinusEps;
                loss = NumOps.Subtract(loss,
                    NumOps.Add(
                        NumOps.Multiply(targets[i], NumOps.Log(clampedCalib)),
                        NumOps.Multiply(NumOps.Subtract(NumOps.One, targets[i]),
                            NumOps.Log(NumOps.Subtract(NumOps.One, clampedCalib)))));
            }

            if (NumOps.LessThan(loss, bestLoss))
            {
                bestLoss = loss;
                bestTemp = t;
            }
        }

        _temperature = bestTemp;
    }

    /// <summary>
    /// Numerically stable sigmoid in type T.
    /// </summary>
    private T SigmoidT(T x)
    {
        if (NumOps.GreaterThanOrEquals(x, NumOps.Zero))
        {
            T ez = NumOps.Exp(NumOps.Negate(x));
            return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, ez));
        }
        else
        {
            T ez = NumOps.Exp(x);
            return NumOps.Divide(ez, NumOps.Add(NumOps.One, ez));
        }
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

        var calibrated = new Matrix<T>(input.Rows, NumClasses);

        for (int i = 0; i < input.Rows; i++)
        {
            // Binary classification: calibrate positive class
            T p = uncalibrated[i, 1];
            T calibP = CalibrateProb(p);

            calibrated[i, 0] = NumOps.Subtract(NumOps.One, calibP);
            calibrated[i, 1] = calibP;
        }

        return calibrated;
    }

    /// <summary>
    /// Applies the fitted calibration to a single probability.
    /// </summary>
    private T CalibrateProb(T p)
    {
        T eps = NumOps.FromDouble(1e-10);
        T oneMinusEps = NumOps.Subtract(NumOps.One, eps);
        if (NumOps.LessThan(p, eps)) p = eps;
        if (NumOps.GreaterThan(p, oneMinusEps)) p = oneMinusEps;

        return _options.CalibrationMethod switch
        {
            ProbabilityCalibrationMethod.PlattScaling =>
                SigmoidT(NumOps.Add(NumOps.Multiply(_plattA,
                    NumOps.Log(NumOps.Divide(p, NumOps.Subtract(NumOps.One, p)))), _plattB)),

            ProbabilityCalibrationMethod.IsotonicRegression =>
                InterpolateIsotonic(p),

            ProbabilityCalibrationMethod.BetaCalibration =>
                SigmoidT(NumOps.Add(NumOps.Subtract(
                    NumOps.Multiply(_betaA, NumOps.Log(p)),
                    NumOps.Multiply(_betaB, NumOps.Log(NumOps.Subtract(NumOps.One, p)))),
                    _betaC)),

            ProbabilityCalibrationMethod.TemperatureScaling =>
                SigmoidT(NumOps.Divide(
                    NumOps.Log(NumOps.Divide(p, NumOps.Subtract(NumOps.One, p))),
                    _temperature)),

            ProbabilityCalibrationMethod.Auto => p,
            ProbabilityCalibrationMethod.None => p,
            _ => p
        };
    }

    /// <summary>
    /// Interpolates isotonic regression mapping.
    /// </summary>
    private T InterpolateIsotonic(T p)
    {
        if (_isotonicMapping == null || _isotonicMapping.Length == 0)
            return p;

        int low = 0, high = _isotonicMapping.Length - 1;

        // Boundary cases
        if (NumOps.LessThanOrEquals(p, _isotonicMapping[low].prob))
            return _isotonicMapping[low].calibrated;
        if (NumOps.GreaterThanOrEquals(p, _isotonicMapping[high].prob))
            return _isotonicMapping[high].calibrated;

        // Binary search for bracketing points
        while (high - low > 1)
        {
            int mid = (low + high) / 2;
            if (NumOps.LessThanOrEquals(_isotonicMapping[mid].prob, p))
                low = mid;
            else
                high = mid;
        }

        // Linear interpolation
        T range = NumOps.Subtract(_isotonicMapping[high].prob, _isotonicMapping[low].prob);
        T rangeEps = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(range, rangeEps))
            return _isotonicMapping[low].calibrated;

        T t = NumOps.Divide(NumOps.Subtract(p, _isotonicMapping[low].prob), range);
        return NumOps.Add(
            NumOps.Multiply(_isotonicMapping[low].calibrated, NumOps.Subtract(NumOps.One, t)),
            NumOps.Multiply(_isotonicMapping[high].calibrated, t));
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.CalibratedClassifier;

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        IProbabilisticClassifier<T> clonedBase;
        if (_baseClassifier is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
        {
            clonedBase = (IProbabilisticClassifier<T>)fullModel.Clone();
        }
        else
        {
            clonedBase = _baseClassifier;
        }

        return new CalibratedClassifier<T>(clonedBase, new CalibratedClassifierOptions<T>
        {
            CalibrationMethod = _options.CalibrationMethod,
            CrossValidationFolds = _options.CrossValidationFolds,
            CalibrationSetFraction = _options.CalibrationSetFraction,
            Seed = _options.Seed
        });
    }

    /// <inheritdoc/>
    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = (CalibratedClassifier<T>)CreateNewInstance();

        clone.NumFeatures = NumFeatures;
        clone.NumClasses = NumClasses;
        clone.TaskType = TaskType;
        clone._isTrained = _isTrained;

        // Copy calibration parameters
        clone._plattA = _plattA;
        clone._plattB = _plattB;
        clone._betaA = _betaA;
        clone._betaB = _betaB;
        clone._betaC = _betaC;
        clone._temperature = _temperature;

        if (_isotonicMapping != null)
        {
            clone._isotonicMapping = new (T, T)[_isotonicMapping.Length];
            Array.Copy(_isotonicMapping, clone._isotonicMapping, _isotonicMapping.Length);
        }

        if (ClassLabels != null)
        {
            clone.ClassLabels = new Vector<T>(ClassLabels.Length);
            for (int i = 0; i < ClassLabels.Length; i++)
            {
                clone.ClassLabels[i] = ClassLabels[i];
            }
        }

        return clone;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
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
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length < 6)
        {
            throw new ArgumentException(
                $"Expected at least 6 parameters, but received {parameters.Length}.", nameof(parameters));
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

    /// <inheritdoc/>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Calibration wrappers don't use gradient-based optimization
        // The calibration is fitted using closed-form solutions or simple optimization
        return new Vector<T>(6); // 6 calibration parameters
    }

    /// <inheritdoc/>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Calibration wrappers don't use gradient-based optimization
        // This is a no-op for compatibility
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        var (baseTypeName, baseData) = ClassifierRegistry<T>.SerializeClassifier((IClassifier<T>)_baseClassifier);

        var modelDict = new Dictionary<string, object?>
        {
            { "ClassLabels", ClassLabels?.ToArray().Select(NumOps.ToDouble).ToArray() },
            { "NumClasses", NumClasses },
            { "NumFeatures", NumFeatures },
            { "TaskType", (int)TaskType },
            { "CalibrationMethod", (int)_options.CalibrationMethod },
            { "PlattA", _plattA },
            { "PlattB", _plattB },
            { "BetaA", _betaA },
            { "BetaB", _betaB },
            { "BetaC", _betaC },
            { "Temperature", _temperature },
            { "IsotonicMapping", _isotonicMapping?.Select(m => new[] { m.prob, m.calibrated }).ToArray() },
            { "IsTrained", _isTrained },
            { "BaseClassifierType", baseTypeName },
            { "BaseClassifierData", baseData }
        };

        var metadata = GetModelMetadata();
        metadata.ModelData = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(modelDict));
        return Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(metadata));
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        var jsonString = Encoding.UTF8.GetString(modelData);
        var metadata = JsonConvert.DeserializeObject<ModelMetadata<T>>(jsonString);
        if (metadata?.ModelData is null)
            throw new InvalidOperationException("Invalid serialized data: missing model metadata.");

        var dataString = Encoding.UTF8.GetString(metadata.ModelData);
        var jObj = JsonConvert.DeserializeObject<JObject>(dataString);
        if (jObj is null)
            throw new InvalidOperationException("Invalid serialized data: model data is not a valid JSON object.");

        var classLabelsArr = jObj["ClassLabels"]?.ToObject<double[]>();
        if (classLabelsArr is not null)
        {
            ClassLabels = new Vector<T>(classLabelsArr.Length);
            for (int i = 0; i < classLabelsArr.Length; i++)
                ClassLabels[i] = NumOps.FromDouble(classLabelsArr[i]);
        }
        NumClasses = jObj["NumClasses"]?.ToObject<int>() ?? 0;
        NumFeatures = jObj["NumFeatures"]?.ToObject<int>() ?? 0;
        TaskType = (ClassificationTaskType)(jObj["TaskType"]?.ToObject<int>() ?? 0);
        _options.CalibrationMethod = (ProbabilityCalibrationMethod)(jObj["CalibrationMethod"]?.ToObject<int>()
            ?? (int)ProbabilityCalibrationMethod.PlattScaling);
        _plattA = jObj["PlattA"]?.ToObject<double>() ?? 1.0;
        _plattB = jObj["PlattB"]?.ToObject<double>() ?? 0.0;
        _betaA = jObj["BetaA"]?.ToObject<double>() ?? 1.0;
        _betaB = jObj["BetaB"]?.ToObject<double>() ?? 1.0;
        _betaC = jObj["BetaC"]?.ToObject<double>() ?? 0.0;
        _temperature = jObj["Temperature"]?.ToObject<double>() ?? 1.0;
        _isTrained = jObj["IsTrained"]?.ToObject<bool>() ?? false;

        var isoArr = jObj["IsotonicMapping"]?.ToObject<double[][]>();
        if (isoArr is not null)
        {
            _isotonicMapping = isoArr
                .Where(m => m is not null && m.Length >= 2)
                .Select(m => (m[0], m[1]))
                .ToArray();
        }
        else
        {
            // Clear stale isotonic calibration state when not present in the payload
            _isotonicMapping = null;
        }

        // Restore wrapped base classifier
        var baseType = jObj["BaseClassifierType"]?.ToObject<string>();
        var baseData = jObj["BaseClassifierData"]?.ToObject<string>();
        if (baseType is null || baseData is null)
            throw new InvalidOperationException(
                "Invalid serialized data: missing BaseClassifierType or BaseClassifierData for CalibratedClassifier.");

        var restoredBase = ClassifierRegistry<T>.DeserializeClassifier(baseType, baseData);
        if (restoredBase is not IProbabilisticClassifier<T> probClassifier)
            throw new InvalidOperationException(
                $"Deserialized base classifier of type '{baseType}' does not implement IProbabilisticClassifier<T>.");

        _baseClassifier = probClassifier;
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
