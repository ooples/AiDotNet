using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Ensemble;

/// <summary>
/// Detects anomalies using XGBOD (Extreme Gradient Boosting Outlier Detection).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> XGBOD is a semi-supervised method that creates new features from
/// unsupervised outlier detection scores, then trains a supervised classifier on these
/// enhanced features. It combines the best of both worlds.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train multiple unsupervised outlier detectors
/// 2. Generate outlier scores as new features (TOS: Transformed Outlier Scores)
/// 3. Combine TOS with original features
/// 4. Train gradient boosting classifier on enhanced features
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When you have some labeled anomaly examples
/// - When feature engineering with outlier scores helps
/// - As a powerful ensemble method
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - N estimators: 10 unsupervised detectors
/// - Boosting rounds: 100
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Zhao, Y., Hryniewicki, M.K. (2018). "XGBOD: Improving Supervised Outlier
/// Detection with Unsupervised Representation Learning." IJCNN.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.DecisionTree)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning", "https://doi.org/10.1109/IJCNN.2018.8489605", Year = 2018, Authors = "Yue Zhao, Maciej K. Hryniewicki")]
public partial class XGBODDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nEstimators;
    private readonly int _boostingRounds;
    private List<IAnomalyDetector<T>>? _baseDetectors;
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T>? _weights; // Simplified boosting weights
    [Buffer]
    private Vector<T>? _featureMean;
    [Buffer]
    private Vector<T>? _featureStd;
    private int _nOriginalFeatures;

    /// <summary>
    /// Gets the number of base estimators.
    /// </summary>
    public int NEstimators => _nEstimators;

    /// <summary>
    /// Gets the number of boosting rounds.
    /// </summary>
    public int BoostingRounds => _boostingRounds;

    /// <summary>
    /// Creates a new XGBOD anomaly detector.
    /// </summary>
    /// <param name="nEstimators">Number of unsupervised base detectors. Default is 10.</param>
    /// <param name="boostingRounds">Number of boosting iterations. Default is 100.</param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public XGBODDetector(int nEstimators = 10, int boostingRounds = 100,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nEstimators < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(nEstimators),
                "NEstimators must be at least 1. Recommended is 10.");
        }

        if (boostingRounds < 1)
        {
            throw new ArgumentOutOfRangeException(nameof(boostingRounds),
                "BoostingRounds must be at least 1. Recommended is 100.");
        }

        _nEstimators = nEstimators;
        _boostingRounds = boostingRounds;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;
        _nOriginalFeatures = X.Columns;

        // Guard against tiny datasets - need at least 3 samples for meaningful detection
        if (n < 3)
        {
            throw new ArgumentException(
                $"XGBOD requires at least 3 samples, but got {n}.",
                nameof(X));
        }

        // Create diverse base detectors
        _baseDetectors = new List<IAnomalyDetector<T>>();
        var tosScores = new List<Vector<T>>(); // Transformed Outlier Scores

        // Base k for neighbor-based detectors, capped at n - 1
        int baseK = Math.Min(10, n - 1);

        for (int e = 0; e < _nEstimators; e++)
        {
            IAnomalyDetector<T> detector;

            // Create diverse detector types
            // Cap neighbor count at n - 1 to avoid exceeding sample count
            int neighborCount = Math.Min(Math.Max(1, baseK + e), n - 1);

            switch (e % 5)
            {
                case 0:
                    detector = new DistanceBased.LocalOutlierFactor<T>(
                        numNeighbors: neighborCount,
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 1:
                    detector = new DistanceBased.KNNDetector<T>(
                        k: neighborCount,
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 2:
                    detector = new TreeBased.IsolationForest<T>(
                        numTrees: 50 + e * 10,
                        maxSamples: Math.Min(256, n),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 3:
                    detector = new Statistical.ZScoreDetector<T>(
                        zThreshold: 2.5 + e * 0.1,
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                default:
                    detector = new Statistical.IQRDetector<T>(
                        multiplier: 1.5 + e * 0.1,
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
            }

            detector.Fit(X);
            _baseDetectors.Add(detector);

            // Get TOS (transformed outlier scores)
            tosScores.Add(detector.ScoreAnomalies(X));
        }

        // Create enhanced feature matrix [original features | TOS]
        int nEnhancedFeatures = _nOriginalFeatures + _nEstimators;
        var engine = Engine;
        var enhanced = BuildEnhancedFeatures(X, tosScores);

        // Normalize enhanced features: population mean and std per feature (a std below eps becomes 1)
        T eps = NumOps.FromDouble(1e-10);
        var (std, mean) = engine.TensorStdMean(enhanced, new[] { 0 }, 0, false);
        std = engine.TensorWhere(engine.TensorLessThan(std, eps), Tensor<T>.CreateOnes(new[] { nEnhancedFeatures }), std);
        _featureMean = mean.ToVector();
        _featureStd = std.ToVector();
        var enhancedData = AnomalyTensorOps.DivideRow(AnomalyTensorOps.SubtractRow(enhanced, mean), std).ToMatrix();

        // Train simple gradient boosting on enhanced features
        _weights = TrainSimplifiedBoosting(enhancedData);

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
    }

    private Vector<T> TrainSimplifiedBoosting(Matrix<T> data)
    {
        int n = data.Rows;
        int d = data.Columns;

        var engine = Engine;
        var dataTensor = AnomalyTensorOps.ToTensor(data);

        // Initialize weights and sample weights (uniform)
        var weights = engine.Fill(d, NumOps.Divide(NumOps.One, NumOps.FromDouble(d))).AsTensor();
        var sampleWeights = engine.Fill(n, NumOps.Divide(NumOps.One, NumOps.FromDouble(n))).AsTensor();

        // Create pseudo-labels based on TOS majority vote: the nAnomaly highest average TOS are anomalies (1)
        int nAnomaly = (int)(n * _contamination);
        var tos = engine.TensorSlice(dataTensor, new[] { 0, _nOriginalFeatures }, new[] { n, _nEstimators });
        var avgTos = engine.TensorDivideScalar(engine.ReduceSum(tos, new[] { 1 }, false), NumOps.FromDouble(_nEstimators));
        var pseudoLabels = Tensor<T>.CreateZeros(new[] { n });
        if (nAnomaly > 0)
        {
            var ranked = AnomalyTensorOps.StableOrder(avgTos, descending: true);
            var anomalies = engine.TensorSlice(ranked, new[] { 0 }, new[] { nAnomaly });
            pseudoLabels = engine.TensorIndexFill(pseudoLabels, 0, anomalies, NumOps.One);
        }

        T half = NumOps.FromDouble(0.5);
        T eps = NumOps.FromDouble(1e-10);

        // The median threshold of every feature, the stump predictions [x > threshold] and where they
        // disagree with the pseudo-labels depend only on the data, so they are computed once.
        var (sortedColumns, _) = engine.TensorSort(dataTensor, 0, false);
        var thresholds = engine.TensorSliceAxis(sortedColumns, 0, n / 2);
        var predicted = engine.TensorGreaterThan(dataTensor, AnomalyTensorOps.BroadcastRow(thresholds, n));
        var mismatch = engine.TensorAbs(engine.TensorSubtract(predicted, AnomalyTensorOps.BroadcastColumn(pseudoLabels, d)));
        var mismatchByFeature = engine.TensorTranspose(mismatch);
        var identity = engine.TensorEye<T>(d);

        // Boosting iterations
        for (int round = 0; round < _boostingRounds; round++)
        {
            // Weighted error of every feature's median stump; the best feature has the largest gain
            // 0.5 - error (the first one on ties)
            var error = engine.Reshape(
                engine.TensorMatMul(mismatchByFeature, engine.Reshape(sampleWeights, new[] { n, 1 })), new[] { d });
            var gain = engine.ScalarMinusTensor(half, error);
            int bestFeature = AnomalyTensorOps.FirstArgMax(gain);
            T bestGain = engine.TensorMaxValue(gain);

            // Update weights
            T numerator = NumOps.Add(NumOps.Subtract(NumOps.One, bestGain), half);
            T denominator = NumOps.Add(NumOps.Add(bestGain, half), eps);
            T alpha = NumOps.Multiply(half, NumOps.Log(NumOps.Divide(numerator, denominator)));
            weights = engine.TensorAdd(weights, engine.TensorMultiplyScalar(engine.TensorSliceAxis(identity, 0, bestFeature), alpha));

            // Update sample weights: misclassified samples are scaled by exp(|alpha|), then normalized
            T expAlpha = NumOps.Exp(NumOps.Abs(alpha));
            var misclassified = engine.TensorSliceAxis(mismatchByFeature, 0, bestFeature);
            sampleWeights = engine.TensorWhere(misclassified, engine.TensorMultiplyScalar(sampleWeights, expAlpha), sampleWeights);
            sampleWeights = engine.TensorDivideScalar(sampleWeights, engine.TensorSum(sampleWeights));
        }

        // Normalize feature weights
        return engine.TensorDivideScalar(weights, engine.TensorSum(weights)).ToVector();
    }

    // [original features | transformed outlier scores of every base detector] as one [n, d + e] tensor.
    private Tensor<T> BuildEnhancedFeatures(Matrix<T> X, List<Vector<T>> tosScores)
    {
        var engine = Engine;
        var tos = engine.TensorStack(tosScores.Select(s => s.AsTensor()).ToArray(), 1);
        return engine.TensorConcatenate(new[] { AnomalyTensorOps.ToTensor(X), tos }, 1);
    }

    /// <inheritdoc/>
    public override Vector<T> ScoreAnomalies(Matrix<T> X)
    {
        EnsureFitted();
        return ScoreAnomaliesInternal(X);
    }

    private Vector<T> ScoreAnomaliesInternal(Matrix<T> X)
    {
        ValidateInput(X);

        if (X.Columns != _nOriginalFeatures)
        {
            throw new ArgumentException(
                $"Input has {X.Columns} features, but model was fitted with {_nOriginalFeatures} features.",
                nameof(X));
        }

        int n = X.Rows;
        int nEnhancedFeatures = _nOriginalFeatures + _nEstimators;

        var featureMean = _featureMean ?? throw new InvalidOperationException("Feature mean not computed.");
        var featureStd = _featureStd ?? throw new InvalidOperationException("Feature std not computed.");
        var weights = _weights ?? throw new InvalidOperationException("Weights not computed.");

        // Get TOS for test data
        var tosScores = new List<Vector<T>>();
        foreach (var detector in _baseDetectors ?? throw new InvalidOperationException("Model not properly fitted."))
        {
            tosScores.Add(detector.ScoreAnomalies(X));
        }

        // Normalized enhanced features, scored by the weighted sum with the boosting weights
        var enhanced = AnomalyTensorOps.DivideRow(
            AnomalyTensorOps.SubtractRow(BuildEnhancedFeatures(X, tosScores), featureMean.AsTensor()), featureStd.AsTensor());
        return Engine.MatrixVectorMultiply(enhanced.ToMatrix(), weights);
    }
}
