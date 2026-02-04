using AiDotNet.DriftDetection;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.Online;

/// <summary>
/// Implements Adaptive Random Forest (ARF) for online classification with drift detection.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Adaptive Random Forest is an ensemble of Hoeffding trees designed
/// to handle concept drift in data streams. Each tree has an associated drift detector, and when
/// drift is detected, the affected tree is replaced with a new one trained on recent data.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Maintain an ensemble of Hoeffding trees</item>
/// <item>Each tree has a "warning" detector and a "drift" detector</item>
/// <item>When warning is triggered, start training a background tree</item>
/// <item>When drift is confirmed, replace the old tree with the background tree</item>
/// <item>Use majority voting for predictions</item>
/// </list>
/// </para>
///
/// <para><b>Key innovations:</b>
/// <list type="bullet">
/// <item>Per-tree drift detection for local adaptation</item>
/// <item>Resampling with Poisson(λ=6) for diversity</item>
/// <item>Background tree preparation for smooth transitions</item>
/// <item>Weighted voting based on tree accuracy</item>
/// </list>
/// </para>
///
/// <para><b>Advantages:</b>
/// <list type="bullet">
/// <item>Handles both sudden and gradual drift</item>
/// <item>Maintains diversity through random subspaces</item>
/// <item>Provides robust predictions through ensemble voting</item>
/// <item>Adapts locally without resetting entire ensemble</item>
/// </list>
/// </para>
///
/// <para><b>Reference:</b> Gomes et al., "Adaptive Random Forests for Evolving Data Stream Classification" (2017)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AdaptiveRandomForestClassifier<T> : ClassifierBase<T>, IOnlineClassifier<T>
{
    private readonly AdaptiveRandomForestOptions<T> _options;
    private readonly Random _random;
    private readonly List<T> _knownClasses;
    private readonly List<TreeMember> _ensemble;

    /// <summary>
    /// Gets the total number of samples the model has seen.
    /// </summary>
    public long SamplesSeen { get; private set; }

    /// <summary>
    /// Gets whether the model has seen at least one sample.
    /// </summary>
    public bool IsWarm => SamplesSeen > 0;

    /// <summary>
    /// Represents a member tree in the ensemble with its drift detector and statistics.
    /// </summary>
    private class TreeMember
    {
        /// <summary>
        /// The Hoeffding tree classifier.
        /// </summary>
        public HoeffdingTreeClassifier<T>? Tree { get; set; }

        /// <summary>
        /// Drift detector for this tree's errors.
        /// </summary>
        public DDMDriftDetector<T>? DriftDetector { get; set; }

        /// <summary>
        /// Warning detector (triggers background tree training).
        /// </summary>
        public DDMDriftDetector<T>? WarningDetector { get; set; }

        /// <summary>
        /// Background tree being trained when warning is active.
        /// </summary>
        public HoeffdingTreeClassifier<T>? BackgroundTree { get; set; }

        /// <summary>
        /// Features selected for this tree (random subspace).
        /// </summary>
        public int[]? SelectedFeatures { get; set; }

        /// <summary>
        /// Running accuracy estimate for weighted voting.
        /// </summary>
        public double AccuracyEstimate { get; set; } = 1.0;

        /// <summary>
        /// Number of correct predictions.
        /// </summary>
        public long CorrectCount { get; set; }

        /// <summary>
        /// Number of total predictions evaluated.
        /// </summary>
        public long TotalCount { get; set; }

        /// <summary>
        /// Whether this tree is currently in warning state.
        /// </summary>
        public bool InWarning { get; set; }
    }

    /// <summary>
    /// Creates a new Adaptive Random Forest classifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    public AdaptiveRandomForestClassifier(AdaptiveRandomForestOptions<T>? options = null)
        : base(options)
    {
        _options = options ?? new AdaptiveRandomForestOptions<T>();
        _random = _options.RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.RandomSeed.Value)
            : RandomHelper.CreateSecureRandom();
        _knownClasses = new List<T>();
        _ensemble = new List<TreeMember>();
    }

    /// <inheritdoc />
    protected override ModelType GetModelType() => ModelType.AdaptiveRandomForestClassifier;

    /// <summary>
    /// Updates the model with a single training sample.
    /// </summary>
    public void PartialFit(Vector<T> features, T label)
    {
        // Initialize ensemble on first sample
        if (NumFeatures == 0)
        {
            NumFeatures = features.Length;
            InitializeEnsemble();
        }

        // Register new class if needed
        int classIdx = GetOrCreateClassIndex(label);

        SamplesSeen++;

        // Update each tree in the ensemble
        foreach (var member in _ensemble)
        {
            // Validate member state - these must be initialized during ensemble creation
            if (member.Tree is null || member.SelectedFeatures is null ||
                member.DriftDetector is null || member.WarningDetector is null)
            {
                throw new InvalidOperationException("Ensemble member is not properly initialized.");
            }

            // Evaluate prequentially (predict BEFORE training to avoid bias)
            var selectedFeaturesForPred = ExtractSelectedFeatures(features, member.SelectedFeatures);
            var prediction = member.Tree.Predict(ConvertToMatrix(selectedFeaturesForPred));
            bool isCorrect = NumOps.Compare(prediction[0], label) == 0;

            // Update accuracy estimate with exponential decay
            member.TotalCount++;
            if (isCorrect) member.CorrectCount++;
            member.AccuracyEstimate = (double)member.CorrectCount / member.TotalCount;

            // Update drift detectors with error (1 = error, 0 = correct)
            T error = isCorrect ? NumOps.Zero : NumOps.One;

            bool warningTriggered = member.WarningDetector.AddObservation(error);
            bool driftTriggered = member.DriftDetector.AddObservation(error);

            // Handle warning state
            if (member.WarningDetector.IsInWarning && !member.InWarning)
            {
                member.InWarning = true;
                member.BackgroundTree = CreateTree();
            }
            else if (!member.WarningDetector.IsInWarning && member.InWarning && !driftTriggered)
            {
                // Warning cleared, discard background tree
                member.InWarning = false;
                member.BackgroundTree = null;
            }

            // Handle drift detection
            if (driftTriggered)
            {
                if (member.BackgroundTree is not null)
                {
                    // Replace with background tree
                    member.Tree = member.BackgroundTree;
                }
                else
                {
                    // Create new tree
                    member.Tree = CreateTree();
                }

                // Reset state
                member.BackgroundTree = null;
                member.InWarning = false;
                member.DriftDetector.Reset();
                member.WarningDetector.Reset();
                member.CorrectCount = 0;
                member.TotalCount = 0;
                member.AccuracyEstimate = 1.0;
            }

            // Poisson sampling for diversity (train AFTER evaluation)
            int sampleWeight = PoissonSample(_options.LambdaPoisson);
            if (sampleWeight > 0)
            {
                var selectedFeatures = ExtractSelectedFeatures(features, member.SelectedFeatures);
                for (int w = 0; w < sampleWeight; w++)
                    member.Tree.PartialFit(selectedFeatures, label);
                if (member.InWarning && member.BackgroundTree is not null)
                {
                    for (int w = 0; w < sampleWeight; w++)
                        member.BackgroundTree.PartialFit(selectedFeatures, label);
                }
            }
        }
    }

    /// <summary>
    /// Updates the model with a batch of training samples.
    /// </summary>
    public void PartialFit(Matrix<T> features, Vector<T> labels)
    {
        for (int i = 0; i < features.Rows; i++)
        {
            var sample = new Vector<T>(features.Columns);
            for (int j = 0; j < features.Columns; j++)
            {
                sample[j] = features[i, j];
            }
            PartialFit(sample, labels[i]);
        }
    }

    /// <inheritdoc />
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        PartialFit(x, y);
    }

    /// <inheritdoc />
    public override Vector<T> Predict(Matrix<T> input)
    {
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            var features = new Vector<T>(input.Columns);
            for (int j = 0; j < input.Columns; j++)
            {
                features[j] = input[i, j];
            }
            predictions[i] = PredictSingle(features);
        }

        return predictions;
    }

    private T PredictSingle(Vector<T> features)
    {
        if (!IsWarm || _ensemble.Count == 0 || _knownClasses.Count == 0)
        {
            return _knownClasses.Count > 0 ? _knownClasses[0] : default!;
        }

        // Weighted voting
        var voteWeights = new Dictionary<int, double>();

        foreach (var member in _ensemble)
        {
            var selectedFeatures = ExtractSelectedFeatures(features, member.SelectedFeatures!);
            var treePrediction = member.Tree!.Predict(ConvertToMatrix(selectedFeatures));

            int classIdx = GetClassIndex(treePrediction[0]);
            if (classIdx >= 0)
            {
                double weight = member.AccuracyEstimate;
                if (!voteWeights.ContainsKey(classIdx))
                {
                    voteWeights[classIdx] = 0;
                }
                voteWeights[classIdx] += weight;
            }
        }

        if (voteWeights.Count == 0)
        {
            return _knownClasses[0];
        }

        // Find majority vote
        int winningClass = 0;
        double maxWeight = double.MinValue;
        foreach (var kv in voteWeights)
        {
            if (kv.Value > maxWeight)
            {
                maxWeight = kv.Value;
                winningClass = kv.Key;
            }
        }

        return _knownClasses[winningClass];
    }

    private void InitializeEnsemble()
    {
        int numFeaturesToSelect = Math.Max(1, (int)Math.Sqrt(NumFeatures));
        if (_options.NumFeaturesPerTree > 0)
        {
            numFeaturesToSelect = Math.Min(_options.NumFeaturesPerTree, NumFeatures);
        }

        for (int t = 0; t < _options.NumTrees; t++)
        {
            // Select random subset of features
            var allFeatures = Enumerable.Range(0, NumFeatures).ToArray();
            Shuffle(allFeatures);
            var selectedFeatures = allFeatures.Take(numFeaturesToSelect).ToArray();

            var member = new TreeMember
            {
                Tree = CreateTree(),
                DriftDetector = new DDMDriftDetector<T>(_options.WarningThreshold, _options.DriftThreshold),
                WarningDetector = new DDMDriftDetector<T>(_options.WarningThreshold, _options.DriftThreshold),
                SelectedFeatures = selectedFeatures
            };

            _ensemble.Add(member);
        }
    }

    private HoeffdingTreeClassifier<T> CreateTree()
    {
        var treeOptions = new HoeffdingTreeOptions<T>
        {
            GracePeriod = _options.GracePeriod,
            Delta = _options.HoeffdingDelta,
            TieThreshold = _options.TieThreshold,
            MaxDepth = _options.MaxTreeDepth,
            NumBins = _options.NumBins,
            RandomSeed = _random.Next()
        };

        return new HoeffdingTreeClassifier<T>(treeOptions);
    }

    private Vector<T> ExtractSelectedFeatures(Vector<T> allFeatures, int[] selectedIndices)
    {
        var result = new Vector<T>(selectedIndices.Length);
        for (int i = 0; i < selectedIndices.Length; i++)
        {
            result[i] = allFeatures[selectedIndices[i]];
        }
        return result;
    }

    private Matrix<T> ConvertToMatrix(Vector<T> vector)
    {
        var matrix = new Matrix<T>(1, vector.Length);
        for (int j = 0; j < vector.Length; j++)
        {
            matrix[0, j] = vector[j];
        }
        return matrix;
    }

    private int PoissonSample(double lambda)
    {
        // Knuth algorithm for Poisson sampling
        double L = Math.Exp(-lambda);
        int k = 0;
        double p = 1.0;

        do
        {
            k++;
            p *= _random.NextDouble();
        } while (p > L);

        return k - 1;
    }

    private void Shuffle(int[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = _random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    private int GetOrCreateClassIndex(T label)
    {
        for (int i = 0; i < _knownClasses.Count; i++)
        {
            if (NumOps.Compare(_knownClasses[i], label) == 0)
            {
                return i;
            }
        }

        _knownClasses.Add(label);
        NumClasses = _knownClasses.Count;
        ClassLabels = new Vector<T>(_knownClasses.ToArray());
        return _knownClasses.Count - 1;
    }

    private int GetClassIndex(T label)
    {
        for (int i = 0; i < _knownClasses.Count; i++)
        {
            if (NumOps.Compare(_knownClasses[i], label) == 0)
            {
                return i;
            }
        }
        return -1;
    }

    /// <summary>
    /// Gets the number of trees in the ensemble.
    /// </summary>
    public int TreeCount => _ensemble.Count;

    /// <summary>
    /// Gets the number of trees currently in warning state.
    /// </summary>
    public int TreesInWarning => _ensemble.Count(m => m.InWarning);

    /// <summary>
    /// Gets the average accuracy across all trees.
    /// </summary>
    public double AverageTreeAccuracy => _ensemble.Count > 0
        ? _ensemble.Average(m => m.AccuracyEstimate)
        : 0;

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        // Ensemble is structural, return minimal parameters
        return new Vector<T>(1) { [0] = NumOps.FromDouble(_ensemble.Count) };
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        // Ensemble structure cannot be set from flat parameters
    }

    /// <inheritdoc />
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        // Return a cold instance to avoid inconsistent state.
        // Structural parameters only - ensemble cannot be set from flat parameters.
        return new AdaptiveRandomForestClassifier<T>(_options);
    }

    /// <inheritdoc />
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new AdaptiveRandomForestClassifier<T>(_options);
    }

    /// <inheritdoc />
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        // Tree ensemble - no gradients
        return new Vector<T>(0);
    }

    /// <inheritdoc />
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Tree ensemble - no gradient application
    }
}
