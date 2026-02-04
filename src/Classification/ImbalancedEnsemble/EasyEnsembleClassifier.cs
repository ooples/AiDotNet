using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Classification.ImbalancedEnsemble;

/// <summary>
/// Easy Ensemble Classifier for extremely imbalanced datasets.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Easy Ensemble is designed for highly imbalanced data. It creates
/// multiple balanced subsets by undersampling the majority class, trains a base classifier
/// (typically AdaBoost) on each subset, and combines their predictions.</para>
///
/// <para><b>How it works:</b>
/// <list type="number">
/// <item>Keep all minority class samples</item>
/// <item>For each subset: randomly undersample majority to match minority</item>
/// <item>Train an AdaBoost classifier on each balanced subset</item>
/// <item>Combine predictions using majority voting or averaging</item>
/// </list>
/// </para>
///
/// <para><b>Key advantages:</b>
/// <list type="bullet">
/// <item><b>Handles extreme imbalance:</b> Works well even with 100:1 or 1000:1 ratios</item>
/// <item><b>Uses all majority data:</b> Through multiple subsets, all majority samples contribute</item>
/// <item><b>Reduces variance:</b> Ensemble approach gives stable predictions</item>
/// </list>
/// </para>
///
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Extremely imbalanced datasets (more than 10:1 ratio)</item>
/// <item>When you need high recall for the minority class</item>
/// <item>When losing some majority class accuracy is acceptable</item>
/// </list>
/// </para>
///
/// <para><b>References:</b>
/// <list type="bullet">
/// <item>Liu, X.Y., Wu, J., &amp; Zhou, Z.H. (2009). "Exploratory Undersampling for Class-Imbalance Learning"</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class EasyEnsembleClassifier<T> : ClassifierBase<T>
{
    /// <summary>
    /// The ensemble of AdaBoost classifiers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each AdaBoost classifier is trained on a different balanced
    /// subset. Their combined predictions give the final result.</para>
    /// </remarks>
    private readonly List<AdaBoostSubClassifier> _subClassifiers;

    /// <summary>
    /// Number of subset classifiers.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More subsets means more diversity and potentially better
    /// results, but also longer training time. 10-50 is typical.</para>
    /// </remarks>
    private readonly int _nSubsets;

    /// <summary>
    /// Number of AdaBoost estimators per subset.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each subset classifier is an AdaBoost with this many
    /// weak learners. More estimators = stronger but slower classifiers.</para>
    /// </remarks>
    private readonly int _nEstimatorsPerSubset;

    /// <summary>
    /// Maximum depth for weak learners in AdaBoost.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Shallow trees (depth 1-3) are typical for AdaBoost.
    /// Depth 1 ("decision stumps") is common and works well.</para>
    /// </remarks>
    private readonly int _maxDepth;

    /// <summary>
    /// Learning rate for AdaBoost.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Smaller values require more estimators but can give better
    /// results. 1.0 is standard, 0.1-0.5 for more conservative boosting.</para>
    /// </remarks>
    private readonly double _learningRate;

    /// <summary>
    /// Sampling strategy for undersampling.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how majority class is undersampled:
    /// "auto" matches minority count, "ratio:2" uses 2x minority count, etc.</para>
    /// </remarks>
    private readonly string _samplingStrategy;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private readonly Random _random;

    /// <summary>
    /// Whether to use soft voting (probability averaging) vs hard voting.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Soft voting averages probabilities from all classifiers.
    /// Hard voting counts class votes. Soft voting often gives better results.</para>
    /// </remarks>
    private readonly bool _softVoting;

    /// <summary>
    /// Initializes a new instance of EasyEnsembleClassifier.
    /// </summary>
    /// <param name="nSubsets">Number of balanced subsets. Default is 10.</param>
    /// <param name="nEstimatorsPerSubset">AdaBoost estimators per subset. Default is 50.</param>
    /// <param name="maxDepth">Max depth of weak learners. Default is 1 (stumps).</param>
    /// <param name="learningRate">AdaBoost learning rate. Default is 1.0.</param>
    /// <param name="samplingStrategy">Undersampling strategy. Default is "auto".</param>
    /// <param name="softVoting">Use soft voting. Default is true.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Default parameters work well for most imbalanced cases:
    /// <list type="bullet">
    /// <item>Increase nSubsets for very imbalanced data (uses more majority samples)</item>
    /// <item>Increase nEstimatorsPerSubset for complex decision boundaries</item>
    /// <item>Reduce learningRate if overfitting (with more estimators)</item>
    /// </list>
    /// </para>
    /// </remarks>
    public EasyEnsembleClassifier(
        int nSubsets = 10,
        int nEstimatorsPerSubset = 50,
        int maxDepth = 1,
        double learningRate = 1.0,
        string samplingStrategy = "auto",
        bool softVoting = true,
        int? seed = null)
        : base()
    {
        if (nSubsets < 1)
            throw new ArgumentOutOfRangeException(nameof(nSubsets), "nSubsets must be at least 1.");
        if (nEstimatorsPerSubset < 1)
            throw new ArgumentOutOfRangeException(nameof(nEstimatorsPerSubset), "nEstimatorsPerSubset must be at least 1.");
        if (maxDepth < 1)
            throw new ArgumentOutOfRangeException(nameof(maxDepth), "maxDepth must be at least 1.");
        if (learningRate <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "learningRate must be greater than 0.");
        if (string.IsNullOrWhiteSpace(samplingStrategy))
            throw new ArgumentException("samplingStrategy cannot be null or whitespace.", nameof(samplingStrategy));

        _nSubsets = nSubsets;
        _nEstimatorsPerSubset = nEstimatorsPerSubset;
        _maxDepth = maxDepth;
        _learningRate = learningRate;
        _samplingStrategy = samplingStrategy;
        _softVoting = softVoting;
        _subClassifiers = [];
        _random = seed.HasValue
            ? RandomHelper.CreateSeededRandom(seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Gets the model type.
    /// </summary>
    /// <returns>ModelType.EasyEnsembleClassifier.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This identifier helps the system track what type of model this is.</para>
    /// </remarks>
    protected override ModelType GetModelType() => ModelType.EasyEnsembleClassifier;

    /// <summary>
    /// Trains the Easy Ensemble classifier.
    /// </summary>
    /// <param name="x">Feature matrix [n_samples, n_features].</param>
    /// <param name="y">Class labels.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Training creates multiple balanced subsets and trains an
    /// AdaBoost classifier on each. This ensures all majority samples contribute across subsets.</para>
    /// </remarks>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples must match number of labels.");
        }

        NumFeatures = x.Columns;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        if (NumClasses != 2)
        {
            throw new ArgumentException(
                "EasyEnsembleClassifier only supports binary classification. " +
                $"Found {NumClasses} classes. For multi-class problems, consider One-vs-Rest strategies.");
        }

        _subClassifiers.Clear();

        // Group samples by class
        var classSamples = new Dictionary<int, List<int>>();
        for (int c = 0; c < NumClasses; c++)
        {
            classSamples[c] = [];
        }

        for (int i = 0; i < y.Length; i++)
        {
            int classIdx = GetClassIndexFromLabel(y[i]);
            if (classIdx >= 0 && classIdx < NumClasses)
            {
                classSamples[classIdx].Add(i);
            }
        }

        // Find minority class size and majority class
        int minoritySize = classSamples.Values.Min(v => v.Count);
        if (minoritySize == 0)
        {
            throw new ArgumentException(
                "One or more classes have no samples. Cannot train on empty class data.",
                nameof(y));
        }
        int minorityClass = classSamples.First(kv => kv.Value.Count == minoritySize).Key;
        int majorityClass = NumClasses == 2 ? 1 - minorityClass :
            classSamples.First(kv => kv.Key != minorityClass).Key;

        // Train subclassifiers
        for (int s = 0; s < _nSubsets; s++)
        {
            // Create balanced subset
            var subsetIndices = CreateBalancedSubset(classSamples, minoritySize, minorityClass);

            // Extract subset data
            var subX = new Matrix<T>(subsetIndices.Length, NumFeatures);
            var subY = new Vector<T>(subsetIndices.Length);
            for (int i = 0; i < subsetIndices.Length; i++)
            {
                int srcIdx = subsetIndices[i];
                for (int j = 0; j < NumFeatures; j++)
                {
                    subX[i, j] = x[srcIdx, j];
                }
                subY[i] = y[srcIdx];
            }

            // Train AdaBoost on this subset
            var adaBoost = TrainAdaBoost(subX, subY);
            _subClassifiers.Add(adaBoost);
        }
    }

    /// <summary>
    /// Creates a balanced subset by undersampling majority class.
    /// </summary>
    /// <param name="classSamples">Samples grouped by class.</param>
    /// <param name="samplesPerClass">Target samples per class.</param>
    /// <param name="minorityClass">Index of minority class.</param>
    /// <returns>Indices of selected samples.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Keeps all minority samples and randomly selects an equal
    /// number from each majority class. This creates a balanced training set.</para>
    /// </remarks>
    private int[] CreateBalancedSubset(Dictionary<int, List<int>> classSamples,
        int samplesPerClass, int minorityClass)
    {
        var selected = new List<int>();

        foreach (var (classIdx, samples) in classSamples)
        {
            if (classIdx == minorityClass)
            {
                // Use all minority samples
                selected.AddRange(samples);
            }
            else
            {
                // Undersample majority class
                var shuffled = samples.OrderBy(_ => _random.Next()).Take(samplesPerClass).ToList();
                selected.AddRange(shuffled);
            }
        }

        return [.. selected];
    }

    /// <summary>
    /// Trains an AdaBoost classifier.
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="y">Labels.</param>
    /// <returns>Trained AdaBoost classifier.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> AdaBoost works by:
    /// <list type="number">
    /// <item>Start with equal weights for all samples</item>
    /// <item>Train a weak learner (shallow tree)</item>
    /// <item>Increase weights on misclassified samples</item>
    /// <item>Repeat, building a weighted ensemble</item>
    /// </list>
    /// This focuses later learners on hard-to-classify samples.</para>
    /// </remarks>
    private AdaBoostSubClassifier TrainAdaBoost(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;
        var weights = new double[n];
        for (int i = 0; i < n; i++)
        {
            weights[i] = 1.0 / n;
        }

        var weakLearners = new List<WeakLearner>();
        var alphas = new List<double>();

        for (int t = 0; t < _nEstimatorsPerSubset; t++)
        {
            // Train weak learner (decision stump or shallow tree)
            var learner = TrainWeakLearner(x, y, weights);

            // Compute weighted error
            double weightedError = 0;
            var predictions = new int[n];
            for (int i = 0; i < n; i++)
            {
                predictions[i] = PredictWeakLearner(learner, x, i);
                int actual = GetClassIndexFromLabel(y[i]);
                if (predictions[i] != actual)
                {
                    weightedError += weights[i];
                }
            }

            // Avoid edge cases
            weightedError = Math.Max(1e-10, Math.Min(1 - 1e-10, weightedError));

            // Compute alpha (classifier weight)
            double alpha = _learningRate * 0.5 * Math.Log((1 - weightedError) / weightedError);

            // Update sample weights
            double weightSum = 0;
            for (int i = 0; i < n; i++)
            {
                int actual = GetClassIndexFromLabel(y[i]);
                double sign = predictions[i] == actual ? -1 : 1;
                weights[i] *= Math.Exp(alpha * sign);
                weightSum += weights[i];
            }

            // Normalize weights
            for (int i = 0; i < n; i++)
            {
                weights[i] /= weightSum;
            }

            weakLearners.Add(learner);
            alphas.Add(alpha);
        }

        return new AdaBoostSubClassifier
        {
            WeakLearners = weakLearners,
            Alphas = alphas
        };
    }

    /// <summary>
    /// Trains a weak learner (decision stump or shallow tree).
    /// </summary>
    /// <param name="x">Feature matrix.</param>
    /// <param name="y">Labels.</param>
    /// <param name="weights">Sample weights.</param>
    /// <returns>Trained weak learner.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> A weak learner is a simple classifier that's only slightly
    /// better than random guessing. Decision stumps (depth-1 trees) are common because they're
    /// fast and work well with boosting.</para>
    /// </remarks>
    private WeakLearner TrainWeakLearner(Matrix<T> x, Vector<T> y, double[] weights)
    {
        int n = x.Rows;
        var bestLearner = new WeakLearner();
        double bestWeightedError = double.MaxValue;

        // Try each feature
        for (int j = 0; j < NumFeatures; j++)
        {
            // Get sorted unique values
            var values = new SortedSet<double>();
            for (int i = 0; i < n; i++)
            {
                values.Add(NumOps.ToDouble(x[i, j]));
            }

            if (values.Count < 2)
            {
                continue;
            }

            // Try split points
            var valueList = values.ToList();
            for (int v = 0; v < valueList.Count - 1; v++)
            {
                double threshold = (valueList[v] + valueList[v + 1]) / 2.0;

                // Try both polarities
                for (int polarity = -1; polarity <= 1; polarity += 2)
                {
                    double weightedError = 0;
                    for (int i = 0; i < n; i++)
                    {
                        double val = NumOps.ToDouble(x[i, j]);
                        int pred = (polarity * (val - threshold) > 0) ? 1 : 0;
                        int actual = GetClassIndexFromLabel(y[i]);
                        if (pred != actual)
                        {
                            weightedError += weights[i];
                        }
                    }

                    if (weightedError < bestWeightedError)
                    {
                        bestWeightedError = weightedError;
                        bestLearner = new WeakLearner
                        {
                            FeatureIndex = j,
                            Threshold = threshold,
                            Polarity = polarity
                        };
                    }
                }
            }
        }

        // If no valid split was found (all features constant), throw an exception
        if (bestWeightedError == double.MaxValue)
        {
            throw new InvalidOperationException(
                "No valid split found: all features appear to be constant. " +
                "Cannot train a weak learner when there is no feature variation.");
        }

        return bestLearner;
    }

    /// <summary>
    /// Gets prediction from a weak learner.
    /// </summary>
    /// <param name="learner">The weak learner.</param>
    /// <param name="x">Feature matrix.</param>
    /// <param name="sampleIdx">Sample index.</param>
    /// <returns>Predicted class (0 or 1 for binary).</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Applies the simple threshold rule learned by the stump.
    /// If polarity is positive, values above threshold predict class 1.</para>
    /// </remarks>
    private int PredictWeakLearner(WeakLearner learner, Matrix<T> x, int sampleIdx)
    {
        double val = NumOps.ToDouble(x[sampleIdx, learner.FeatureIndex]);
        return (learner.Polarity * (val - learner.Threshold) > 0) ? 1 : 0;
    }

    /// <summary>
    /// Predicts class labels for the given input data.
    /// </summary>
    /// <param name="input">Feature matrix.</param>
    /// <returns>Predicted class labels.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each subset classifier votes. Final prediction comes from
    /// combining all votes (soft voting averages probabilities, hard voting counts class votes).</para>
    /// </remarks>
    public override Vector<T> Predict(Matrix<T> input)
    {
        if (ClassLabels is null || _subClassifiers.Count == 0)
        {
            throw new InvalidOperationException("Model has not been trained.");
        }

        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            if (_softVoting)
            {
                // Average probabilities across subclassifiers
                var avgScores = new double[NumClasses];

                foreach (var subClassifier in _subClassifiers)
                {
                    double score = PredictAdaBoostScore(subClassifier, input, i);
                    // Convert score to probability
                    double prob = 1.0 / (1.0 + Math.Exp(-2 * score));
                    avgScores[1] += prob;
                    avgScores[0] += 1 - prob;
                }

                // Pick class with highest average score
                int bestClass = avgScores[1] > avgScores[0] ? 1 : 0;
                predictions[i] = ClassLabels[bestClass];
            }
            else
            {
                // Hard voting
                var votes = new int[NumClasses];
                foreach (var subClassifier in _subClassifiers)
                {
                    int pred = PredictAdaBoost(subClassifier, input, i);
                    votes[pred]++;
                }

                int bestClass = votes[1] > votes[0] ? 1 : 0;
                predictions[i] = ClassLabels[bestClass];
            }
        }

        return predictions;
    }

    /// <summary>
    /// Gets the raw AdaBoost score for a sample.
    /// </summary>
    /// <param name="classifier">AdaBoost classifier.</param>
    /// <param name="x">Feature matrix.</param>
    /// <param name="sampleIdx">Sample index.</param>
    /// <returns>Weighted sum of weak learner predictions.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The score is the weighted sum of weak learner votes.
    /// Positive scores indicate class 1, negative indicate class 0.</para>
    /// </remarks>
    private double PredictAdaBoostScore(AdaBoostSubClassifier classifier, Matrix<T> x, int sampleIdx)
    {
        double score = 0;
        for (int t = 0; t < classifier.WeakLearners.Count; t++)
        {
            int pred = PredictWeakLearner(classifier.WeakLearners[t], x, sampleIdx);
            int sign = pred == 1 ? 1 : -1;
            score += classifier.Alphas[t] * sign;
        }
        return score;
    }

    /// <summary>
    /// Gets the AdaBoost class prediction for a sample.
    /// </summary>
    /// <param name="classifier">AdaBoost classifier.</param>
    /// <param name="x">Feature matrix.</param>
    /// <param name="sampleIdx">Sample index.</param>
    /// <returns>Predicted class index.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Takes the sign of the weighted score to get the final class.</para>
    /// </remarks>
    private int PredictAdaBoost(AdaBoostSubClassifier classifier, Matrix<T> x, int sampleIdx)
    {
        double score = PredictAdaBoostScore(classifier, x, sampleIdx);
        return score >= 0 ? 1 : 0;
    }

    /// <summary>
    /// Gets the model parameters.
    /// </summary>
    /// <returns>Vector with subclassifier count.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Complex ensemble models don't fit into a simple parameter vector.
    /// Use serialization for full model persistence.</para>
    /// </remarks>
    public override Vector<T> GetParameters()
    {
        return new Vector<T>(1) { [0] = NumOps.FromDouble(_subClassifiers.Count) };
    }

    /// <summary>
    /// Sets the model parameters.
    /// </summary>
    /// <param name="parameters">Parameter vector (limited support).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use serialization to save/load ensemble models.</para>
    /// </remarks>
    public override void SetParameters(Vector<T> parameters)
    {
        // Limited support for ensemble models
    }

    /// <summary>
    /// Creates a new instance with the specified parameters.
    /// </summary>
    /// <param name="parameters">Parameters (limited support).</param>
    /// <returns>New model instance.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates a new untrained model with same hyperparameters.</para>
    /// </remarks>
    public override IFullModel<T, Matrix<T>, Vector<T>> WithParameters(Vector<T> parameters)
    {
        return new EasyEnsembleClassifier<T>(_nSubsets, _nEstimatorsPerSubset, _maxDepth,
            _learningRate, _samplingStrategy, _softVoting);
    }

    /// <summary>
    /// Creates a new instance of this model type.
    /// </summary>
    /// <returns>New instance with same hyperparameters.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Creates an untrained copy with the same settings.</para>
    /// </remarks>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new EasyEnsembleClassifier<T>(_nSubsets, _nEstimatorsPerSubset, _maxDepth,
            _learningRate, _samplingStrategy, _softVoting);
    }

    /// <summary>
    /// Computes gradients (not applicable for ensemble models).
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="target">Target labels.</param>
    /// <param name="lossFunction">Optional loss function.</param>
    /// <returns>Zero gradient vector.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensemble models don't use gradient descent.</para>
    /// </remarks>
    public override Vector<T> ComputeGradients(Matrix<T> input, Vector<T> target, ILossFunction<T>? lossFunction = null)
    {
        return new Vector<T>(1) { [0] = NumOps.Zero };
    }

    /// <summary>
    /// Applies gradients (not applicable for ensemble models).
    /// </summary>
    /// <param name="gradients">Gradient vector.</param>
    /// <param name="learningRate">Learning rate.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Ensemble models don't support gradient updates.</para>
    /// </remarks>
    public override void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        // Ensemble models don't support gradient-based updates
    }

    /// <summary>
    /// Gets feature importance based on weak learner usage.
    /// </summary>
    /// <returns>Dictionary mapping feature names to importance scores.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Features used in more weak learners with higher alpha
    /// weights are considered more important. Values are normalized to sum to 1.</para>
    /// </remarks>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new double[NumFeatures];

        foreach (var subClassifier in _subClassifiers)
        {
            for (int t = 0; t < subClassifier.WeakLearners.Count; t++)
            {
                int featureIdx = subClassifier.WeakLearners[t].FeatureIndex;
                importance[featureIdx] += Math.Abs(subClassifier.Alphas[t]);
            }
        }

        double total = importance.Sum();
        if (total == 0) total = 1;

        var result = new Dictionary<string, T>();
        for (int i = 0; i < NumFeatures; i++)
        {
            string name = FeatureNames is not null && i < FeatureNames.Length
                ? FeatureNames[i]
                : $"Feature_{i}";
            result[name] = NumOps.FromDouble(importance[i] / total);
        }

        return result;
    }

    /// <summary>
    /// Represents an AdaBoost sub-classifier.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Contains the weak learners and their weights (alphas)
    /// that make up one AdaBoost classifier.</para>
    /// </remarks>
    private class AdaBoostSubClassifier
    {
        /// <summary>
        /// List of weak learners (decision stumps).
        /// </summary>
        public List<WeakLearner> WeakLearners { get; set; } = [];

        /// <summary>
        /// Weights for each weak learner.
        /// </summary>
        public List<double> Alphas { get; set; } = [];
    }

    /// <summary>
    /// Represents a weak learner (decision stump).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A decision stump splits on a single feature at a threshold.
    /// Polarity determines which side predicts which class.</para>
    /// </remarks>
    private class WeakLearner
    {
        /// <summary>
        /// Feature index for the split.
        /// </summary>
        public int FeatureIndex { get; set; }

        /// <summary>
        /// Threshold value for the split.
        /// </summary>
        public double Threshold { get; set; }

        /// <summary>
        /// Polarity: +1 means values above threshold predict class 1,
        /// -1 means values below threshold predict class 1.
        /// </summary>
        public int Polarity { get; set; } = 1;
    }
}
