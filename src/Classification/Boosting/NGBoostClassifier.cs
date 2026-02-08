using AiDotNet.Classification.Ensemble;
using AiDotNet.Classification.Trees;
using AiDotNet.Distributions;
using AiDotNet.Models.Options;
using AiDotNet.Regression;
using AiDotNet.Scoring;

namespace AiDotNet.Classification.Boosting;

/// <summary>
/// NGBoost (Natural Gradient Boosting) classifier for probabilistic classification.
/// </summary>
/// <remarks>
/// <para>
/// NGBoost is an algorithm for probabilistic prediction that uses natural gradients
/// to efficiently and directly optimize a proper scoring rule. For classification,
/// it predicts class probabilities that are well-calibrated.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional classifiers give you a class prediction like
/// "this email is spam" with maybe a confidence score. But NGBoost gives you
/// properly calibrated probabilities - if it says "70% spam", then about 70% of
/// similar predictions will actually be spam.
///
/// Key benefits:
/// - Well-calibrated probability estimates
/// - Quantifies prediction uncertainty
/// - Uses natural gradients for stable, efficient learning
/// - Works well for both binary and multi-class problems
/// </para>
/// <para>
/// Reference: Duan, T., et al. "NGBoost: Natural Gradient Boosting for Probabilistic
/// Prediction" (2019). https://arxiv.org/abs/1910.03225
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NGBoostClassifier<T> : EnsembleClassifierBase<T>
{
    /// <summary>
    /// Base learners for each class's log-odds.
    /// </summary>
    private readonly List<DecisionTreeRegression<T>[]> _trees;

    /// <summary>
    /// Initial log-odds values for each class.
    /// </summary>
    private Vector<T> _initialLogOdds;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly NGBoostClassifierOptions<T> _options;

    /// <summary>
    /// Number of classes.
    /// </summary>
    private int _numClasses;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random _random;

    /// <summary>
    /// Gets the number of trees in the ensemble.
    /// </summary>
    public int NumberOfTrees => _trees.Count * _numClasses;

    /// <summary>
    /// Initializes a new instance of NGBoostClassifier.
    /// </summary>
    /// <param name="options">Configuration options for the algorithm.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public NGBoostClassifier(NGBoostClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new NGBoostClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
        _options = options ?? new NGBoostClassifierOptions<T>();
        _trees = [];
        _initialLogOdds = new Vector<T>(0);
        _numClasses = 0;
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.NGBoostClassifier;

    /// <summary>
    /// Trains the NGBoost classifier using natural gradient boosting.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        int n = x.Rows;
        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        _numClasses = ClassLabels.Length;
        NumClasses = _numClasses;
        TaskType = InferTaskType(y);

        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Convert labels to class indices
        var yIndices = new int[n];
        for (int i = 0; i < n; i++)
        {
            // Find the class index for this label
            int labelIndex = -1;
            for (int c = 0; c < ClassLabels.Length; c++)
            {
                if (NumOps.Compare(y[i], ClassLabels[c]) == 0)
                {
                    labelIndex = c;
                    break;
                }
            }
            if (labelIndex < 0)
                throw new InvalidOperationException(
                    $"Label value at index {i} does not match any known class label. " +
                    "Ensure all training labels are present in ClassLabels.");
            yIndices[i] = labelIndex;
        }

        // Initialize log-odds based on class frequencies
        _initialLogOdds = new Vector<T>(_numClasses);
        var classCounts = new int[_numClasses];
        for (int i = 0; i < n; i++)
        {
            classCounts[yIndices[i]]++;
        }
        for (int c = 0; c < _numClasses; c++)
        {
            double p = Math.Max(1e-10, Math.Min(1 - 1e-10, (double)classCounts[c] / n));
            _initialLogOdds[c] = NumOps.FromDouble(Math.Log(p));
        }

        // Initialize current log-odds for all samples
        var currentLogOdds = new Vector<T>[_numClasses];
        for (int c = 0; c < _numClasses; c++)
        {
            currentLogOdds[c] = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                currentLogOdds[c][i] = _initialLogOdds[c];
            }
        }

        _trees.Clear();
        Estimators.Clear();

        double bestScore = double.MaxValue;
        int roundsWithoutImprovement = 0;

        for (int iter = 0; iter < _options.NumberOfIterations; iter++)
        {
            // Subsample if needed
            int[] sampleIndices = GetSampleIndices(n);
            int sampleSize = sampleIndices.Length;

            // Compute probabilities from current log-odds (softmax)
            var probs = ComputeProbabilities(currentLogOdds, sampleIndices);

            // Compute gradients for each class
            var gradients = new Vector<T>[_numClasses];
            for (int c = 0; c < _numClasses; c++)
            {
                gradients[c] = new Vector<T>(sampleSize);
            }

            // For cross-entropy loss, gradient = p - y
            for (int i = 0; i < sampleSize; i++)
            {
                int idx = sampleIndices[i];
                int trueClass = yIndices[idx];
                for (int c = 0; c < _numClasses; c++)
                {
                    T target = c == trueClass ? NumOps.One : NumOps.Zero;
                    gradients[c][i] = NumOps.Subtract(probs[c][i], target);
                }
            }

            // Compute natural gradients using Fisher Information approximation
            var naturalGradients = _options.UseNaturalGradient
                ? ComputeNaturalGradients(gradients, probs, sampleSize)
                : gradients;

            // Build trees for each class
            var iterTrees = new DecisionTreeRegression<T>[_numClasses];

            for (int c = 0; c < _numClasses; c++)
            {
                // Create pseudo-residuals (negative natural gradients)
                var residuals = new Vector<T>(sampleSize);
                for (int i = 0; i < sampleSize; i++)
                {
                    residuals[i] = NumOps.Negate(naturalGradients[c][i]);
                }

                // Build subsample matrix
                var xSample = x.GetRows(sampleIndices);

                // Train tree on pseudo-residuals
                var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions
                {
                    MaxDepth = _options.MaxDepth,
                    MinSamplesSplit = _options.MinSamplesSplit,
                    MaxFeatures = _options.MaxFeatures,
                    SplitCriterion = _options.SplitCriterion,
                    Seed = _random.Next()
                });

                tree.Train(xSample, residuals);
                iterTrees[c] = tree;

                // Update log-odds for all samples
                var treePredictions = tree.Predict(x);
                for (int i = 0; i < n; i++)
                {
                    currentLogOdds[c][i] = NumOps.Add(
                        currentLogOdds[c][i],
                        NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), treePredictions[i]));
                }
            }

            _trees.Add(iterTrees);

            // Early stopping check
            if (_options.EarlyStoppingRounds.HasValue)
            {
                double currentScore = ComputeCrossEntropyLoss(currentLogOdds, yIndices);
                if (currentScore < bestScore)
                {
                    bestScore = currentScore;
                    roundsWithoutImprovement = 0;
                }
                else
                {
                    roundsWithoutImprovement++;
                    if (roundsWithoutImprovement >= _options.EarlyStoppingRounds.Value)
                    {
                        break;
                    }
                }
            }

            // Verbose output
            if (_options.Verbose && (iter + 1) % _options.VerboseEval == 0)
            {
                double score = ComputeCrossEntropyLoss(currentLogOdds, yIndices);
                Console.WriteLine($"[{iter + 1}] Cross-Entropy: {score:F6}");
            }
        }

        // Calculate feature importances
        CalculateFeatureImportances(x.Columns);
    }

    /// <summary>
    /// Predicts class labels for the input samples.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            // Find class with highest probability
            int maxClass = 0;
            T maxProb = NumOps.Zero;
            for (int c = 0; c < _numClasses; c++)
            {
                if (NumOps.Compare(probs[i, c], maxProb) > 0)
                {
                    maxProb = probs[i, c];
                    maxClass = c;
                }
            }
            var classLabels = ClassLabels ?? throw new InvalidOperationException("Model must be trained before making predictions.");
            predictions[i] = classLabels[maxClass];
        }

        return predictions;
    }

    /// <summary>
    /// Predicts class probabilities for the input samples.
    /// </summary>
    /// <returns>Matrix of shape [n_samples, n_classes] with probability estimates.</returns>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (_numClasses == 0 || _initialLogOdds.Length == 0)
            throw new InvalidOperationException(
                "Model has not been trained. Call Train() before PredictProbabilities().");

        int n = input.Rows;
        var probs = new Matrix<T>(n, _numClasses);

        // Initialize log-odds
        var currentLogOdds = new Vector<T>[_numClasses];
        for (int c = 0; c < _numClasses; c++)
        {
            currentLogOdds[c] = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                currentLogOdds[c][i] = _initialLogOdds[c];
            }
        }

        // Accumulate tree predictions
        foreach (var iterTrees in _trees)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                var treePreds = iterTrees[c].Predict(input);
                for (int i = 0; i < n; i++)
                {
                    currentLogOdds[c][i] = NumOps.Add(
                        currentLogOdds[c][i],
                        NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), treePreds[i]));
                }
            }
        }

        // Convert log-odds to probabilities using softmax
        for (int i = 0; i < n; i++)
        {
            // Compute softmax with numerical stability (subtract max)
            double maxLogOdd = double.NegativeInfinity;
            for (int c = 0; c < _numClasses; c++)
            {
                maxLogOdd = Math.Max(maxLogOdd, NumOps.ToDouble(currentLogOdds[c][i]));
            }

            double sumExp = 0;
            var expValues = new double[_numClasses];
            for (int c = 0; c < _numClasses; c++)
            {
                expValues[c] = Math.Exp(NumOps.ToDouble(currentLogOdds[c][i]) - maxLogOdd);
                sumExp += expValues[c];
            }

            for (int c = 0; c < _numClasses; c++)
            {
                probs[i, c] = NumOps.FromDouble(expValues[c] / sumExp);
            }
        }

        return probs;
    }

    /// <summary>
    /// Predicts log probabilities for the input samples.
    /// </summary>
    /// <returns>Matrix of shape [n_samples, n_classes] with log probability estimates.</returns>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var logProbs = new Matrix<T>(probs.Rows, probs.Columns);

        for (int i = 0; i < probs.Rows; i++)
        {
            for (int c = 0; c < probs.Columns; c++)
            {
                logProbs[i, c] = NumOps.FromDouble(Math.Log(Math.Max(1e-10, NumOps.ToDouble(probs[i, c]))));
            }
        }

        return logProbs;
    }

    /// <summary>
    /// Computes probabilities from log-odds using softmax.
    /// </summary>
    private Vector<T>[] ComputeProbabilities(Vector<T>[] logOdds, int[] indices)
    {
        int sampleSize = indices.Length;
        var probs = new Vector<T>[_numClasses];
        for (int c = 0; c < _numClasses; c++)
        {
            probs[c] = new Vector<T>(sampleSize);
        }

        for (int i = 0; i < sampleSize; i++)
        {
            int idx = indices[i];

            // Compute softmax with numerical stability
            double maxLogOdd = double.NegativeInfinity;
            for (int c = 0; c < _numClasses; c++)
            {
                maxLogOdd = Math.Max(maxLogOdd, NumOps.ToDouble(logOdds[c][idx]));
            }

            double sumExp = 0;
            var expValues = new double[_numClasses];
            for (int c = 0; c < _numClasses; c++)
            {
                expValues[c] = Math.Exp(NumOps.ToDouble(logOdds[c][idx]) - maxLogOdd);
                sumExp += expValues[c];
            }

            for (int c = 0; c < _numClasses; c++)
            {
                probs[c][i] = NumOps.FromDouble(expValues[c] / sumExp);
            }
        }

        return probs;
    }

    /// <summary>
    /// Computes natural gradients using Fisher Information approximation.
    /// </summary>
    private Vector<T>[] ComputeNaturalGradients(Vector<T>[] gradients, Vector<T>[] probs, int sampleSize)
    {
        var naturalGrads = new Vector<T>[_numClasses];

        // For softmax, Fisher Information diagonal approximation: F_cc = p(1-p)
        // Natural gradient = gradient / F
        for (int c = 0; c < _numClasses; c++)
        {
            naturalGrads[c] = new Vector<T>(sampleSize);
            for (int i = 0; i < sampleSize; i++)
            {
                double p = NumOps.ToDouble(probs[c][i]);
                double fisher = p * (1 - p);
                fisher = Math.Max(fisher, 1e-6); // Prevent division by zero

                naturalGrads[c][i] = NumOps.FromDouble(NumOps.ToDouble(gradients[c][i]) / fisher);
            }
        }

        return naturalGrads;
    }

    /// <summary>
    /// Computes cross-entropy loss for current predictions.
    /// </summary>
    private double ComputeCrossEntropyLoss(Vector<T>[] logOdds, int[] yIndices)
    {
        int n = yIndices.Length;
        double totalLoss = 0;

        for (int i = 0; i < n; i++)
        {
            // Compute log-softmax for numerical stability
            double maxLogOdd = double.NegativeInfinity;
            for (int c = 0; c < _numClasses; c++)
            {
                maxLogOdd = Math.Max(maxLogOdd, NumOps.ToDouble(logOdds[c][i]));
            }

            double sumExp = 0;
            for (int c = 0; c < _numClasses; c++)
            {
                sumExp += Math.Exp(NumOps.ToDouble(logOdds[c][i]) - maxLogOdd);
            }

            double logProb = NumOps.ToDouble(logOdds[yIndices[i]][i]) - maxLogOdd - Math.Log(sumExp);
            totalLoss -= logProb;
        }

        return totalLoss / n;
    }

    /// <summary>
    /// Gets sample indices for subsampling.
    /// </summary>
    private int[] GetSampleIndices(int n)
    {
        if (_options.SubsampleRatio >= 1.0)
        {
            return Enumerable.Range(0, n).ToArray();
        }

        int sampleSize = (int)(n * _options.SubsampleRatio);
        return SamplingHelper.SampleWithoutReplacement(n, sampleSize);
    }

    /// <summary>
    /// Calculates feature importances from all trees.
    /// </summary>
    private void CalculateFeatureImportances(int featureCount)
    {
        var importances = new Vector<T>(featureCount);

        // Aggregate importances from all trees
        foreach (var iterTrees in _trees)
        {
            foreach (var tree in iterTrees)
            {
                var fi = tree.FeatureImportances;
                int copyCount = Math.Min(featureCount, fi.Length);
                for (int i = 0; i < copyCount; i++)
                {
                    importances[i] = NumOps.Add(importances[i], fi[i]);
                }
            }
        }

        // Normalize
        T sum = NumOps.Zero;
        for (int i = 0; i < featureCount; i++)
        {
            sum = NumOps.Add(sum, importances[i]);
        }
        if (NumOps.ToDouble(sum) > 0)
        {
            for (int i = 0; i < featureCount; i++)
            {
                importances[i] = NumOps.Divide(importances[i], sum);
            }
        }

        FeatureImportances = importances;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NGBoostClassifier,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfIterations", _trees.Count },
                { "NumberOfClasses", _numClasses },
                { "LearningRate", _options.LearningRate },
                { "MaxDepth", _options.MaxDepth },
                { "UseNaturalGradient", _options.UseNaturalGradient }
            }
        };
    }

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Options
        writer.Write(_options.NumberOfIterations);
        writer.Write(_options.LearningRate);
        writer.Write(_options.SubsampleRatio);
        writer.Write(_options.UseNaturalGradient);
        writer.Write(_options.MaxDepth);
        writer.Write(_options.MinSamplesSplit);
        writer.Write(_options.MaxFeatures);
        writer.Write((int)_options.SplitCriterion);
        writer.Write(_options.EarlyStoppingRounds.HasValue);
        if (_options.EarlyStoppingRounds.HasValue)
            writer.Write(_options.EarlyStoppingRounds.Value);
        writer.Write(_options.Verbose);
        writer.Write(_options.VerboseEval);
        writer.Write(_options.Seed.HasValue);
        if (_options.Seed.HasValue)
            writer.Write(_options.Seed.Value);

        // Class info
        writer.Write(_numClasses);
        for (int c = 0; c < _numClasses; c++)
        {
            writer.Write(NumOps.ToDouble(_initialLogOdds[c]));
        }

        // Trees
        writer.Write(_trees.Count);
        foreach (var iterTrees in _trees)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                byte[] treeData = iterTrees[c].Serialize();
                writer.Write(treeData.Length);
                writer.Write(treeData);
            }
        }

        return ms.ToArray();
    }

    /// <inheritdoc/>
    public override void Deserialize(byte[] modelData)
    {
        using var ms = new MemoryStream(modelData);
        using var reader = new BinaryReader(ms);

        int baseLen = reader.ReadInt32();
        byte[] baseData = reader.ReadBytes(baseLen);
        base.Deserialize(baseData);

        // Options
        _options.NumberOfIterations = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.SubsampleRatio = reader.ReadDouble();
        _options.UseNaturalGradient = reader.ReadBoolean();
        _options.MaxDepth = reader.ReadInt32();
        _options.MinSamplesSplit = reader.ReadInt32();
        _options.MaxFeatures = reader.ReadDouble();
        _options.SplitCriterion = (Enums.SplitCriterion)reader.ReadInt32();
        if (reader.ReadBoolean())
            _options.EarlyStoppingRounds = reader.ReadInt32();
        _options.Verbose = reader.ReadBoolean();
        _options.VerboseEval = reader.ReadInt32();
        if (reader.ReadBoolean())
            _options.Seed = reader.ReadInt32();

        // Class info
        _numClasses = reader.ReadInt32();
        _initialLogOdds = new Vector<T>(_numClasses);
        for (int c = 0; c < _numClasses; c++)
        {
            _initialLogOdds[c] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Trees
        int numIter = reader.ReadInt32();
        _trees.Clear();
        for (int iter = 0; iter < numIter; iter++)
        {
            var iterTrees = new DecisionTreeRegression<T>[_numClasses];
            for (int c = 0; c < _numClasses; c++)
            {
                int treeLen = reader.ReadInt32();
                byte[] treeData = reader.ReadBytes(treeLen);
                iterTrees[c] = new DecisionTreeRegression<T>(new DecisionTreeOptions());
                iterTrees[c].Deserialize(treeData);
            }
            _trees.Add(iterTrees);
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new NGBoostClassifier<T>(_options, Regularization);
    }
}
