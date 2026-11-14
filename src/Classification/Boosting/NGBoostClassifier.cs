using AiDotNet.Attributes;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Classification.Trees;
using AiDotNet.Distributions;
using AiDotNet.Enums;
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
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
/// // Create NGBoost classifier for well-calibrated probabilistic predictions
/// var options = new NGBoostClassifierOptions&lt;double&gt;();
///
/// // Prepare training data: 6 samples with 2 features
/// var features = new Matrix&lt;double&gt;(6, 2);
/// features[0, 0] = 1.0; features[0, 1] = 1.1;
/// features[1, 0] = 1.2; features[1, 1] = 0.9;
/// features[2, 0] = 0.8; features[2, 1] = 1.0;
/// features[3, 0] = 5.0; features[3, 1] = 5.1;
/// features[4, 0] = 5.2; features[4, 1] = 4.9;
/// features[5, 0] = 4.8; features[5, 1] = 5.0;
/// var labels = new Vector&lt;double&gt;(new double[] { 0, 0, 0, 1, 1, 1 });
///
/// // Train using natural gradient boosting for calibrated probabilities
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new NGBoostClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict with calibrated probability estimates
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("NGBoost: Natural Gradient Boosting for Probabilistic Prediction", "https://arxiv.org/abs/1910.03225", Year = 2020, Authors = "Tony Duan, Avati Anand, Daisy Yi Ding, Khanh K. Thai, Sanjay Basu, Andrew Y. Ng, Alejandro Schuler")]
public partial class NGBoostClassifier<T> : EnsembleClassifierBase<T>
{
    /// <summary>
    /// Base learners for each class's log-odds.
    /// </summary>
    private readonly List<DecisionTreeRegression<T>[]> _trees;

    /// <summary>
    /// Initial log-odds values for each class.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
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
        : base(options ??= new NGBoostClassifierOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options;
        _trees = [];
        _initialLogOdds = new Vector<T>(0);
        _numClasses = 0;
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Returns the model type identifier for this classifier.
    /// </summary>

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

        var engine = Engine;

        // Convert labels to class indices
        var yIndexTensor = GetClassIndexTensor(y);
        var yIndices = yIndexTensor.ToArray();
        int unknownAt = Array.IndexOf(yIndices, -1);
        if (unknownAt >= 0)
            throw new InvalidOperationException(
                $"Label value at index {unknownAt} does not match any known class label. " +
                "Ensure all training labels are present in ClassLabels.");

        // Initialize log-odds based on class frequencies: log(clamp(count_c / n, 1e-10, 1 - 1e-10)); the counts
        // are taken in int (a sum of ones in T stops at 2048 for Half)
        var classCounts = engine.TensorBinCount(yIndexTensor, _numClasses).Slice(0, 0, _numClasses);
        var priors = engine.TensorClamp(CountFractions(classCounts, n),
            Tolerance(1e-10), NumOps.FromDouble(1 - 1e-10));
        _initialLogOdds = engine.TensorLog(priors).ToVector();

        // Current log-odds as a [n, classes] matrix, every row starting at the prior
        var currentLogOdds = InitialLogOddsMatrix(n);

        _trees.Clear();
        Estimators.Clear();

        T bestScore = NumOps.MaxValue;
        int roundsWithoutImprovement = 0;
        T lr = NumOps.FromDouble(_options.LearningRate);

        for (int iter = 0; iter < _options.NumberOfIterations; iter++)
        {
            // Subsample if needed
            int[] sampleIndices = GetSampleIndices(n);
            var sampleIndexTensor = new Tensor<int>(sampleIndices, new[] { sampleIndices.Length });

            // Compute probabilities from current log-odds (softmax over classes)
            var probs = engine.Softmax(engine.TensorIndexSelect(currentLogOdds, sampleIndexTensor, 0), 1);

            // For cross-entropy loss, gradient = p - onehot(y)
            var targets = engine.TensorOneHot<T>(engine.TensorGather(yIndexTensor, sampleIndexTensor, 0), _numClasses);
            var gradients = engine.TensorSubtract(probs, targets);

            // Compute natural gradients using Fisher Information approximation
            var naturalGradients = (_options.UseNaturalGradient
                ? ComputeNaturalGradients(gradients, probs)
                : gradients).ToMatrix();

            // Build trees for each class
            var iterTrees = new DecisionTreeRegression<T>[_numClasses];
            var classUpdates = new Vector<T>[_numClasses];

            var xSample = x.GetRows(sampleIndices);

            for (int c = 0; c < _numClasses; c++)
            {
                var residuals = engine.Negate(naturalGradients.GetColumn(c));

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

                classUpdates[c] = engine.Multiply(tree.Predict(x), lr);
            }

            currentLogOdds = engine.TensorAdd(currentLogOdds, Tensor<T>.FromMatrix(StackColumns(classUpdates)));
            _trees.Add(iterTrees);

            // Early stopping check
            if (_options.EarlyStoppingRounds.HasValue
                && ShouldStopEarly(ComputeCrossEntropyLoss(currentLogOdds, yIndexTensor), ref bestScore,
                    ref roundsWithoutImprovement))
            {
                break;
            }

            // Verbose output
            if (_options.Verbose && (iter + 1) % _options.VerboseEval == 0)
            {
                double score = NumOps.ToDouble(ComputeCrossEntropyLoss(currentLogOdds, yIndexTensor));
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
        if (input.Rows == 0)
        {
            return new Vector<T>(0);
        }

        if (ClassLabels is null)
        {
            throw new InvalidOperationException("Model must be trained before making predictions.");
        }

        // Class with highest probability
        return ClassIndicesToLabels(ArgMaxRows(probs));
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
        if (n == 0)
        {
            return new Matrix<T>(0, _numClasses);
        }

        var engine = Engine;
        var currentLogOdds = InitialLogOddsMatrix(n);

        T lr = NumOps.FromDouble(_options.LearningRate);
        var classUpdates = new Vector<T>[_numClasses];
        foreach (var iterTrees in _trees)
        {
            for (int c = 0; c < _numClasses; c++)
            {
                classUpdates[c] = engine.Multiply(iterTrees[c].Predict(input), lr);
            }

            currentLogOdds = engine.TensorAdd(currentLogOdds, Tensor<T>.FromMatrix(StackColumns(classUpdates)));
        }

        // Convert log-odds to probabilities using softmax (numerically stable)
        return engine.Softmax(currentLogOdds, 1).ToMatrix();
    }

    /// <summary>
    /// Predicts log probabilities for the input samples.
    /// </summary>
    /// <returns>Matrix of shape [n_samples, n_classes] with log probability estimates.</returns>
    public override Matrix<T> PredictLogProbabilities(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        if (probs.Rows == 0 || probs.Columns == 0)
        {
            return new Matrix<T>(probs.Rows, probs.Columns);
        }

        var engine = Engine;
        return engine.TensorLog(engine.TensorClampMin(Tensor<T>.FromMatrix(probs), Tolerance(1e-10))).ToMatrix();
    }

    /// <summary>
    /// Builds the [n, classes] log-odds matrix with every row equal to the class priors.
    /// </summary>
    private Tensor<T> InitialLogOddsMatrix(int n)
    {
        var row = _initialLogOdds.AsTensor().Reshape(1, _numClasses);
        return Engine.TensorTile(row, new[] { n, 1 });
    }

    private bool ShouldStopEarly(T score, ref T bestScore, ref int roundsWithoutImprovement)
    {
        if (NumOps.LessThan(score, bestScore))
        {
            bestScore = score;
            roundsWithoutImprovement = 0;
            return false;
        }

        roundsWithoutImprovement++;
        return roundsWithoutImprovement >= (_options.EarlyStoppingRounds ?? int.MaxValue);
    }

    /// <summary>
    /// Computes natural gradients using Fisher Information approximation.
    /// </summary>
    private Tensor<T> ComputeNaturalGradients(Tensor<T> gradients, Tensor<T> probs)
    {
        // For softmax, Fisher Information diagonal approximation: F_cc = p(1-p)
        // Natural gradient = gradient / max(F, 1e-6)
        var engine = Engine;
        var fisher = engine.TensorMultiply(probs, engine.ScalarMinusTensor(NumOps.One, probs));
        return engine.TensorDivide(gradients, engine.TensorClampMin(fisher, NumOps.FromDouble(1e-6)));
    }

    /// <summary>
    /// Computes cross-entropy loss for current predictions.
    /// </summary>
    private T ComputeCrossEntropyLoss(Tensor<T> logOdds, Tensor<int> yIndices)
    {
        int n = yIndices.Length;
        if (n == 0)
        {
            return NumOps.Zero;
        }

        // Mean negative log-softmax probability of the true class
        var engine = Engine;
        var logProbs = engine.TensorLogSoftmax(logOdds, 1);
        var trueClass = engine.TensorOneHot<T>(yIndices, _numClasses);
        T total = engine.TensorSum(engine.TensorMultiply(logProbs, trueClass));
        return NumOps.Negate(NumOps.Divide(total, NumOps.FromDouble(n)));
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

        int sampleSize = Math.Max(1, Math.Min(n, (int)(n * _options.SubsampleRatio)));
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
                importances = Engine.Add(importances, FitToLength(tree.FeatureImportances, featureCount));
            }
        }

        T sum = importances.Sum();
        if (NumOps.GreaterThan(sum, NumOps.Zero))
            importances.MultiplyInPlace(NumOps.Divide(NumOps.One, sum));

        FeatureImportances = importances;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
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
}
