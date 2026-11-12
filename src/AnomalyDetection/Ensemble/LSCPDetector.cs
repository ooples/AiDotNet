using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.AnomalyDetection.Ensemble;

/// <summary>
/// Detects anomalies using LSCP (Locally Selective Combination in Parallel Outlier Ensembles).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> LSCP improves ensemble detection by selecting the most competent
/// detectors for each test point locally. Instead of combining all detectors equally,
/// it selects the best performing detectors based on local pseudo ground truth.
/// </para>
/// <para>
/// The algorithm works by:
/// 1. Train multiple diverse base detectors
/// 2. For each test point, find local region
/// 3. Evaluate detector competence in that region
/// 4. Combine scores from most competent detectors
/// </para>
/// <para>
/// <b>When to use:</b>
/// - When different detectors work better in different regions
/// - For heterogeneous anomaly types
/// - When ensemble averaging isn't optimal
/// </para>
/// <para>
/// <b>Industry Standard Defaults:</b>
/// - N estimators: 10
/// - Local region size: 30 neighbors
/// - Contamination: 0.1 (10%)
/// </para>
/// <para>
/// Reference: Zhao, Y., et al. (2019). "LSCP: Locally Selective Combination in Parallel
/// Outlier Ensembles." SDM.
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelTask(ModelTask.AnomalyDetection)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("LSCP: Locally Selective Combination in Parallel Outlier Ensembles", "https://doi.org/10.1137/1.9781611975673.17", Year = 2019, Authors = "Yue Zhao, Zain Nasrullah, Maciej K. Hryniewicki, Zheng Li")]
public partial class LSCPDetector<T> : AnomalyDetectorBase<T>
{
    private readonly int _nEstimators;
    private readonly int _localRegionSize;
    private List<IAnomalyDetector<T>>? _baseDetectors;
    [Buffer]
    private Matrix<T>? _trainingData;
    [Buffer]
    private Vector<T>[]? _detectorScores;

    /// <summary>
    /// Gets the number of estimators.
    /// </summary>
    public int NEstimators => _nEstimators;

    /// <summary>
    /// Gets the local region size.
    /// </summary>
    public int LocalRegionSize => _localRegionSize;

    /// <summary>
    /// Creates a new LSCP anomaly detector.
    /// </summary>
    /// <param name="nEstimators">Number of base detectors. Default is 10.</param>
    /// <param name="localRegionSize">
    /// Number of neighbors for local region. Default is 30.
    /// </param>
    /// <param name="contamination">Expected proportion of anomalies. Default is 0.1 (10%).</param>
    /// <param name="randomSeed">Random seed for reproducibility. Default is 42.</param>
    public LSCPDetector(int nEstimators = 10, int localRegionSize = 30,
        double contamination = 0.1, int randomSeed = 42)
        : base(contamination, randomSeed)
    {
        if (nEstimators < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(nEstimators),
                "NEstimators must be at least 2. Recommended is 10.");
        }

        if (localRegionSize < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(localRegionSize),
                "LocalRegionSize must be at least 2. Recommended is 30.");
        }

        _nEstimators = nEstimators;
        _localRegionSize = localRegionSize;
    }

    /// <inheritdoc/>
    public override void Fit(Matrix<T> X)
    {
        ValidateInput(X);

        int n = X.Rows;

        _trainingData = X;

        // Create diverse base detectors
        _baseDetectors = new List<IAnomalyDetector<T>>();
        _detectorScores = new Vector<T>[_nEstimators];

        int k = Math.Min(10, n - 1);

        for (int e = 0; e < _nEstimators; e++)
        {
            IAnomalyDetector<T> detector;

            // Create diverse detector types
            switch (e % 3)
            {
                case 0:
                    detector = new DistanceBased.LocalOutlierFactor<T>(
                        numNeighbors: Math.Max(1, k + e - 1),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                case 1:
                    detector = new DistanceBased.KNNDetector<T>(
                        k: Math.Max(1, k + e - 1),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
                default:
                    detector = new TreeBased.IsolationForest<T>(
                        numTrees: 50 + e * 10,
                        maxSamples: Math.Min(256, n),
                        contamination: _contamination,
                        randomSeed: _randomSeed + e);
                    break;
            }

            detector.Fit(X);
            _baseDetectors.Add(detector);

            // Store training scores
            _detectorScores[e] = detector.ScoreAnomalies(X);
        }

        // Calculate scores for training data to set threshold
        var trainingScores = ScoreAnomaliesInternal(X);
        SetThresholdFromContamination(trainingScores);

        _isFitted = true;
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

        var trainingData = _trainingData;
        var baseDetectors = _baseDetectors;
        if (trainingData == null || baseDetectors == null)
        {
            throw new InvalidOperationException("Model not properly fitted.");
        }

        int effectiveRegionSize = Math.Min(_localRegionSize, trainingData.Rows);

        // Get scores from all detectors for test data
        var testScores = new Vector<T>[_nEstimators];
        for (int e = 0; e < _nEstimators; e++)
        {
            testScores[e] = baseDetectors[e].ScoreAnomalies(X);
        }

        var detScores = _detectorScores ?? throw new InvalidOperationException("Model not properly fitted.");

        var engine = Engine;
        int m = X.Rows;

        // Local region of every test point: its effectiveRegionSize nearest training points
        var distances = AnomalyTensorOps.PairwiseDistances(
            AnomalyTensorOps.ToTensor(X), AnomalyTensorOps.ToTensor(trainingData));
        var (sortedDistances, order) = AnomalyTensorOps.StableSortRows(distances, descending: false);
        var localDistances = engine.TensorSlice(sortedDistances, new[] { 0, 0 }, new[] { m, effectiveRegionSize });
        var localRegion = engine.Reshape(
            engine.TensorSlice(order, new[] { 0, 0 }, new[] { m, effectiveRegionSize }), new[] { m * effectiveRegionSize });

        // Evaluate detector competence in local region
        // Use Pearson correlation between scores and distance rank as competence
        var competence = new Tensor<T>[_nEstimators];
        for (int e = 0; e < _nEstimators; e++)
        {
            var localScores = engine.Reshape(
                engine.TensorGather(detScores[e].AsTensor(), localRegion, 0), new[] { m, effectiveRegionSize });

            // Higher scores should correlate with higher distances for good detectors
            competence[e] = AnomalyTensorOps.RowPearson(localScores, localDistances);
        }

        // Select top detectors (ties to the lower index) and average their scores
        int nSelect = Math.Max(1, _nEstimators / 2);
        var selected = AnomalyTensorOps.TopCountMask(engine.TensorStack(competence, 1), nSelect);
        var candidateScores = engine.TensorStack(testScores.Select(s => s.AsTensor()).ToArray(), 1);
        var selectedScores = engine.TensorWhere(selected, candidateScores, Tensor<T>.CreateZeros(new[] { m, _nEstimators }));
        return engine.TensorDivideScalar(engine.ReduceSum(selectedScores, new[] { 1 }, false), NumOps.FromDouble(nSelect)).ToVector();
    }
}
