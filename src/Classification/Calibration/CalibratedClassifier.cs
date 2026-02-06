using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

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
    private readonly IProbabilisticClassifier<T> _baseClassifier;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly CalibratedClassifierOptions<T> _options;

    /// <summary>
    /// Platt scaling parameters (A, B) for sigmoid calibration.
    /// P_calibrated = 1 / (1 + exp(-(A * logit(P) + B)))
    /// </summary>
    private double _plattA = 1.0;
    private double _plattB = 0.0;

    /// <summary>
    /// Beta calibration parameters (a, b, c).
    /// </summary>
    private double _betaA = 1.0;
    private double _betaB = 1.0;
    private double _betaC = 0.0;

    /// <summary>
    /// Temperature scaling parameter.
    /// </summary>
    private double _temperature = 1.0;

    /// <summary>
    /// Isotonic regression mapping (sorted by probability).
    /// </summary>
    private (double prob, double calibrated)[]? _isotonicMapping;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Whether the model has been trained.
    /// </summary>
    private bool _isTrained;

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
        : base(options ?? new CalibratedClassifierOptions<T>(), regularization)
    {
        _baseClassifier = baseClassifier ?? throw new ArgumentNullException(nameof(baseClassifier));
        _options = options ?? new CalibratedClassifierOptions<T>();

        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Trains the base classifier and fits the calibration model.
    /// </summary>
    /// <param name="x">The feature matrix where each row is a sample.</param>
    /// <param name="y">The target labels.</param>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        if (_options.CalibrationMethod == ProbabilityCalibrationMethod.None
            || _options.CalibrationMethod == ProbabilityCalibrationMethod.Auto)
        {
            _baseClassifier.Train(x, y);
            _isTrained = true;
            return;
        }

        int n = x.Rows;

        if (_options.CrossValidationFolds > 1)
        {
            // Use cross-validation to get out-of-fold predictions for calibration
            TrainWithCrossValidation(x, y);
        }
        else
        {
            // Split data: train base model on part, calibrate on holdout
            TrainWithHoldout(x, y);
        }

        _isTrained = true;
    }

    /// <summary>
    /// Trains using cross-validation for calibration (preferred approach).
    /// </summary>
    private void TrainWithCrossValidation(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        int numFolds = _options.CrossValidationFolds;

        // Create fold assignments
        var foldAssignments = new int[n];
        for (int i = 0; i < n; i++)
        {
            foldAssignments[i] = i % numFolds;
        }

        // Shuffle fold assignments
        for (int i = n - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (foldAssignments[i], foldAssignments[j]) = (foldAssignments[j], foldAssignments[i]);
        }

        // Store OOF predictions
        var oofProbabilities = new Matrix<T>(n, NumClasses);

        // Process each fold
        for (int fold = 0; fold < numFolds; fold++)
        {
            // Count samples in train and test
            int trainCount = 0;
            int testCount = 0;
            for (int i = 0; i < n; i++)
            {
                if (foldAssignments[i] == fold)
                    testCount++;
                else
                    trainCount++;
            }

            // Create train and test splits
            var xTrain = new Matrix<T>(trainCount, NumFeatures);
            var yTrain = new Vector<T>(trainCount);
            var xTest = new Matrix<T>(testCount, NumFeatures);
            var testIndices = new int[testCount];

            int trainIdx = 0;
            int testIdx = 0;
            for (int i = 0; i < n; i++)
            {
                if (foldAssignments[i] == fold)
                {
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        xTest[testIdx, j] = x[i, j];
                    }
                    testIndices[testIdx] = i;
                    testIdx++;
                }
                else
                {
                    for (int j = 0; j < NumFeatures; j++)
                    {
                        xTrain[trainIdx, j] = x[i, j];
                    }
                    yTrain[trainIdx] = y[i];
                    trainIdx++;
                }
            }

            // Clone and train on this fold
            IProbabilisticClassifier<T> foldClassifier;
            if (_baseClassifier is IFullModel<T, Matrix<T>, Vector<T>> fullModel)
            {
                foldClassifier = (IProbabilisticClassifier<T>)fullModel.Clone();
            }
            else
            {
                throw new InvalidOperationException(
                    "Base classifier does not implement IFullModel and cannot be cloned. " +
                    "Cross-validation requires clonable classifiers. " +
                    "Set CrossValidationFolds=1 to use holdout validation instead.");
            }

            foldClassifier.Train(xTrain, yTrain);

            // Get predictions for test fold
            var foldProbs = foldClassifier.PredictProbabilities(xTest);

            // Store OOF predictions at original indices
            for (int t = 0; t < testCount; t++)
            {
                int origIdx = testIndices[t];
                for (int c = 0; c < NumClasses; c++)
                {
                    oofProbabilities[origIdx, c] = foldProbs[t, c];
                }
            }
        }

        // Train final base model on all data (for prediction time)
        _baseClassifier.Train(x, y);

        // Fit calibration model on OOF predictions
        FitCalibration(oofProbabilities, y);
    }

    /// <summary>
    /// Trains using holdout set for calibration.
    /// </summary>
    private void TrainWithHoldout(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        if (_options.CalibrationSetFraction <= 0 || _options.CalibrationSetFraction >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(_options.CalibrationSetFraction),
                "CalibrationSetFraction must be between 0 and 1 (exclusive).");
        }
        int calibSize = (int)(n * _options.CalibrationSetFraction);
        int trainSize = n - calibSize;

        if (trainSize < 1)
        {
            throw new ArgumentException("Training set too small after calibration split.");
        }

        if (calibSize < 10)
        {
            throw new ArgumentException(
                $"Calibration set too small ({calibSize} samples). " +
                $"Use more data or increase CalibrationSetFraction.");
        }

        // Create shuffled indices
        var indices = Enumerable.Range(0, n).ToArray();
        for (int i = n - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Split into train and calibration sets
        var trainX = new Matrix<T>(trainSize, NumFeatures);
        var trainY = new Vector<T>(trainSize);
        var calibX = new Matrix<T>(calibSize, NumFeatures);
        var calibY = new Vector<T>(calibSize);

        for (int i = 0; i < trainSize; i++)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                trainX[i, j] = x[indices[i], j];
            }
            trainY[i] = y[indices[i]];
        }

        for (int i = 0; i < calibSize; i++)
        {
            for (int j = 0; j < NumFeatures; j++)
            {
                calibX[i, j] = x[indices[trainSize + i], j];
            }
            calibY[i] = y[indices[trainSize + i]];
        }

        // Train base classifier on training set
        _baseClassifier.Train(trainX, trainY);

        // Get uncalibrated predictions on calibration set
        var uncalibrated = _baseClassifier.PredictProbabilities(calibX);

        // Fit calibration model
        FitCalibration(uncalibrated, calibY);
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
        var probs = new double[n];
        var targets = new double[n];

        // Use last class as the "positive" class for calibration
        int positiveClassIdx = NumClasses - 1;
        var classLabels = ClassLabels ?? throw new InvalidOperationException("Class labels not initialized.");
        var positiveLabel = classLabels[positiveClassIdx];

        for (int i = 0; i < n; i++)
        {
            probs[i] = NumOps.ToDouble(uncalibrated[i, positiveClassIdx]);
            targets[i] = NumOps.Compare(actuals[i], positiveLabel) == 0 ? 1.0 : 0.0;
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
    private void FitPlattScaling(double[] probs, double[] targets)
    {
        // Fit sigmoid: P_calibrated = 1 / (1 + exp(-(A * logit(P) + B)))
        double a = 0.0, b = 0.0;
        double lr = 0.01;
        int maxIter = 1000;
        double tolerance = 1e-8;
        double prevLoss = double.MaxValue;

        for (int iter = 0; iter < maxIter; iter++)
        {
            double gradA = 0, gradB = 0;
            double loss = 0;

            for (int i = 0; i < probs.Length; i++)
            {
                // Clamp probability to avoid log(0)
                double p = Math.Max(1e-10, Math.Min(1 - 1e-10, probs[i]));
                double logit = Math.Log(p / (1 - p));

                double z = a * logit + b;
                double sigmoid = 1.0 / (1.0 + Math.Exp(-z));

                double error = sigmoid - targets[i];
                gradA += error * logit;
                gradB += error;

                // Cross-entropy loss
                double clampedSigmoid = Math.Max(1e-10, Math.Min(1 - 1e-10, sigmoid));
                loss += -targets[i] * Math.Log(clampedSigmoid)
                       - (1 - targets[i]) * Math.Log(1 - clampedSigmoid);
            }

            a -= lr * gradA / probs.Length;
            b -= lr * gradB / probs.Length;

            // Check convergence
            if (Math.Abs(loss - prevLoss) < tolerance)
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
    private void FitIsotonicRegression(double[] probs, double[] targets)
    {
        int n = probs.Length;

        // Sort by predicted probability
        var indexed = new (double prob, double target)[n];
        for (int i = 0; i < n; i++)
        {
            indexed[i] = (probs[i], targets[i]);
        }
        Array.Sort(indexed, (a, b) => a.prob.CompareTo(b.prob));

        // Pool Adjacent Violators Algorithm (PAVA) - block-based implementation
        // Each block tracks: weighted sum, weight, and start/end indices
        var blockValues = new double[n];
        var blockWeights = new double[n];
        var blockEnds = new int[n]; // blockEnds[i] = last index in block starting at i
        int numBlocks = n;

        for (int i = 0; i < n; i++)
        {
            blockValues[i] = indexed[i].target;
            blockWeights[i] = 1.0;
            blockEnds[i] = i;
        }

        // Forward pass: merge violating adjacent blocks
        int current = 0;
        var blockStarts = new List<int> { 0 };

        while (current < n - 1)
        {
            int next = blockEnds[current] + 1;
            if (next >= n) break;

            if (blockValues[current] > blockValues[next])
            {
                // Merge current and next blocks
                double totalWeight = blockWeights[current] + blockWeights[next];
                blockValues[current] = (blockValues[current] * blockWeights[current]
                                      + blockValues[next] * blockWeights[next]) / totalWeight;
                blockWeights[current] = totalWeight;
                blockEnds[current] = blockEnds[next];

                // Check if we need to merge backwards
                while (blockStarts.Count > 1)
                {
                    int prev = blockStarts[^2];
                    if (blockValues[prev] > blockValues[current])
                    {
                        // Merge prev and current
                        double tw = blockWeights[prev] + blockWeights[current];
                        blockValues[prev] = (blockValues[prev] * blockWeights[prev]
                                           + blockValues[current] * blockWeights[current]) / tw;
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
        var calibrated = new double[n];
        foreach (int start in blockStarts)
        {
            int end = blockEnds[start];
            for (int i = start; i <= end; i++)
            {
                calibrated[i] = blockValues[start];
            }
        }

        // Store mapping for prediction (remove duplicates, keep unique probability bins)
        var uniqueMapping = new List<(double prob, double calibrated)>();
        double lastProb = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            if (indexed[i].prob > lastProb + 1e-10)
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
    private void FitBetaCalibration(double[] probs, double[] targets)
    {
        // Beta calibration: P_calibrated = sigmoid(a * log(P) - b * log(1-P) + c)
        double a = 1.0, b = 1.0, c = 0.0;
        double lr = 0.01;
        int maxIter = 1000;
        double tolerance = 1e-8;
        double prevLoss = double.MaxValue;

        for (int iter = 0; iter < maxIter; iter++)
        {
            double gradA = 0, gradB = 0, gradC = 0;
            double loss = 0;

            for (int i = 0; i < probs.Length; i++)
            {
                double p = Math.Max(1e-10, Math.Min(1 - 1e-10, probs[i]));

                double logP = Math.Log(p);
                double log1mP = Math.Log(1 - p);
                double z = a * logP - b * log1mP + c;
                double calibrated = 1.0 / (1.0 + Math.Exp(-z));

                // Cross-entropy gradient through sigmoid: dL/dz = sigma(z) - target
                double error = calibrated - targets[i];

                gradA += error * logP;
                gradB += error * (-log1mP);
                gradC += error;

                // Cross-entropy loss
                double clampedCalib = Math.Max(1e-10, Math.Min(1 - 1e-10, calibrated));
                loss += -targets[i] * Math.Log(clampedCalib)
                       - (1 - targets[i]) * Math.Log(1 - clampedCalib);
            }

            a -= lr * gradA / probs.Length;
            b -= lr * gradB / probs.Length;
            c -= lr * gradC / probs.Length;

            if (Math.Abs(loss - prevLoss) < tolerance)
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
    private void FitTemperatureScaling(double[] probs, double[] targets)
    {
        // Find temperature T that minimizes NLL
        double bestTemp = 1.0;
        double bestLoss = double.MaxValue;

        // Grid search over temperature
        for (double t = 0.1; t <= 10.0; t += 0.05)
        {
            double loss = 0;
            for (int i = 0; i < probs.Length; i++)
            {
                double p = Math.Max(1e-10, Math.Min(1 - 1e-10, probs[i]));
                double logit = Math.Log(p / (1 - p));
                double calibrated = 1.0 / (1.0 + Math.Exp(-logit / t));

                double clampedCalib = Math.Max(1e-10, Math.Min(1 - 1e-10, calibrated));
                loss += -targets[i] * Math.Log(clampedCalib)
                       - (1 - targets[i]) * Math.Log(1 - clampedCalib);
            }

            if (loss < bestLoss)
            {
                bestLoss = loss;
                bestTemp = t;
            }
        }

        _temperature = bestTemp;
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
            double p = NumOps.ToDouble(uncalibrated[i, 1]);
            double calibP = CalibrateProb(p);

            calibrated[i, 0] = NumOps.FromDouble(1 - calibP);
            calibrated[i, 1] = NumOps.FromDouble(calibP);
        }

        return calibrated;
    }

    /// <summary>
    /// Applies the fitted calibration to a single probability.
    /// </summary>
    private double CalibrateProb(double p)
    {
        p = Math.Max(1e-10, Math.Min(1 - 1e-10, p));

        return _options.CalibrationMethod switch
        {
            ProbabilityCalibrationMethod.PlattScaling =>
                1.0 / (1.0 + Math.Exp(-(_plattA * Math.Log(p / (1 - p)) + _plattB))),

            ProbabilityCalibrationMethod.IsotonicRegression =>
                InterpolateIsotonic(p),

            ProbabilityCalibrationMethod.BetaCalibration =>
                1.0 / (1.0 + Math.Exp(-(_betaA * Math.Log(p) - _betaB * Math.Log(1 - p) + _betaC))),

            ProbabilityCalibrationMethod.TemperatureScaling =>
                1.0 / (1.0 + Math.Exp(-Math.Log(p / (1 - p)) / _temperature)),

            ProbabilityCalibrationMethod.Auto => p, // Auto defaults to no transformation at predict time
            ProbabilityCalibrationMethod.None => p,
            _ => p
        };
    }

    /// <summary>
    /// Interpolates isotonic regression mapping.
    /// </summary>
    private double InterpolateIsotonic(double p)
    {
        if (_isotonicMapping == null || _isotonicMapping.Length == 0)
            return p;

        int low = 0, high = _isotonicMapping.Length - 1;

        // Boundary cases
        if (p <= _isotonicMapping[low].prob)
            return _isotonicMapping[low].calibrated;
        if (p >= _isotonicMapping[high].prob)
            return _isotonicMapping[high].calibrated;

        // Binary search for bracketing points
        while (high - low > 1)
        {
            int mid = (low + high) / 2;
            if (_isotonicMapping[mid].prob <= p)
                low = mid;
            else
                high = mid;
        }

        // Linear interpolation
        double range = _isotonicMapping[high].prob - _isotonicMapping[low].prob;
        if (range < 1e-10)
            return _isotonicMapping[low].calibrated;

        double t = (p - _isotonicMapping[low].prob) / range;
        return _isotonicMapping[low].calibrated * (1 - t) + _isotonicMapping[high].calibrated * t;
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
            clone._isotonicMapping = new (double, double)[_isotonicMapping.Length];
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
            NumOps.FromDouble(_plattA),
            NumOps.FromDouble(_plattB),
            NumOps.FromDouble(_betaA),
            NumOps.FromDouble(_betaB),
            NumOps.FromDouble(_betaC),
            NumOps.FromDouble(_temperature)
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

        _plattA = NumOps.ToDouble(parameters[0]);
        _plattB = NumOps.ToDouble(parameters[1]);
        _betaA = NumOps.ToDouble(parameters[2]);
        _betaB = NumOps.ToDouble(parameters[3]);
        _betaC = NumOps.ToDouble(parameters[4]);
        _temperature = NumOps.ToDouble(parameters[5]);
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
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = base.GetModelMetadata();
        metadata.AdditionalInfo["CalibrationMethod"] = _options.CalibrationMethod.ToString();
        metadata.AdditionalInfo["CrossValidationFolds"] = _options.CrossValidationFolds;
        metadata.AdditionalInfo["PlattA"] = _plattA;
        metadata.AdditionalInfo["PlattB"] = _plattB;
        metadata.AdditionalInfo["BetaA"] = _betaA;
        metadata.AdditionalInfo["BetaB"] = _betaB;
        metadata.AdditionalInfo["BetaC"] = _betaC;
        metadata.AdditionalInfo["Temperature"] = _temperature;
        return metadata;
    }
}
