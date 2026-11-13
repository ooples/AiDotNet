using AiDotNet.Attributes;
using AiDotNet.Distributions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.Scoring;

namespace AiDotNet.Regression;

/// <summary>
/// NGBoost (Natural Gradient Boosting) for probabilistic regression.
/// </summary>
/// <remarks>
/// <para>
/// NGBoost is an algorithm for probabilistic prediction that uses natural gradients
/// to efficiently and directly optimize a proper scoring rule. Instead of predicting
/// a single value, NGBoost predicts a full probability distribution.
/// </para>
/// <para>
/// <b>For Beginners:</b> Traditional regression gives you a point prediction like
/// "the house price is $300,000." But NGBoost tells you "the house price follows a
/// normal distribution with mean $300,000 and standard deviation $50,000."
///
/// This uncertainty information is valuable because it tells you how confident the
/// model is. A prediction with small uncertainty means the model is confident.
/// A prediction with large uncertainty means you should be more cautious.
///
/// Key benefits:
/// - Quantifies prediction uncertainty
/// - Can use different distributions for different types of data
/// - Uses natural gradients for stable, efficient learning
/// </para>
/// <para>
/// Reference: Duan, T., et al. "NGBoost: Natural Gradient Boosting for Probabilistic
/// Prediction" (2019). https://arxiv.org/abs/1910.03225
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an NGBoost model for probabilistic regression
/// var options = new NGBoostRegressionOptions&lt;double&gt;();
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train to predict full probability distributions
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new NGBoostRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict distribution parameters (mean and uncertainty)
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("NGBoost: Natural Gradient Boosting for Probabilistic Prediction", "https://arxiv.org/abs/1910.03225", Year = 2019, Authors = "Tony Duan, Anand Avati, Daisy Yi Ding, Khanh K. Thai, Sanjay Basu, Andrew Y. Ng, Alejandro Schuler")]
public partial class NGBoostRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    private const double MinVariance = 1e-6;
    private const double MaxStdDevMultiplier = 4.0;
    private const double MatrixRegularization = 1e-4;
    private const double DeterminantThreshold = 1e-10;
    private const double PivotThreshold = 1e-10;

    /// <summary>
    /// Base learners for each distribution parameter.
    /// </summary>
    private List<DecisionTreeRegression<T>[]> _trees;

    /// <summary>
    /// Initial parameter values (e.g., mean of y for location, initial scale).
    /// </summary>
    [AiDotNet.Attributes.FittedParameter]
    private Vector<T> _initialParameters;

    /// <summary>
    /// The scoring rule used for optimization.
    /// </summary>
    private readonly IScoringRule<T> _scoringRule;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly NGBoostRegressionOptions<T> _options;

    /// <summary>
    /// Number of parameters in the distribution.
    /// </summary>
    private int _numParams;

    /// <summary>
    /// Y standardization parameters for scale-invariant training.
    /// </summary>

    private T _yMean;


    private T _yStd;


    /// <inheritdoc/>
    public override int NumberOfTrees => _trees.Count * _numParams;

    /// <summary>
    /// Gets the distribution type used by this model.
    /// </summary>
    public NGBoostDistributionType DistributionType => _options.DistributionType;

    /// <summary>
    /// Initializes a new instance of NGBoostRegression.
    /// </summary>
    /// <param name="options">Configuration options for the algorithm.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public NGBoostRegression(NGBoostRegressionOptions<T>? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _yMean = NumOps.Zero;
        _yStd = NumOps.Zero;
        _options = options ?? new NGBoostRegressionOptions<T>();
        _trees = [];
        _initialParameters = new Vector<T>(0);
        _numParams = 0;

        // Null means the standard rule. NGBoost optimizes the log score by default in the reference
        // implementation, so that is the default here.
        _scoringRule = _options.ScoringRule ?? new LogScore<T>();
    }

    /// <summary>
    /// Trains the NGBoost model using natural gradient boosting.
    /// </summary>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;

        // Standardize y for scale-invariant training (SIMD-accelerated via Engine)
        T yMeanT = Engine.Mean(y);
        var centered = new Vector<T>(y);
        centered.SubtractInPlace(Vector<T>.CreateDefault(n, yMeanT));
        T yVarT = Engine.DotProduct(centered, centered);
        T yStdT = NumOps.Sqrt(NumOps.Divide(yVarT, NumOps.FromDouble(n)));
        T eps = NumOps.FromDouble(1e-10);
        if (NumOps.LessThan(yStdT, eps)) yStdT = NumOps.One;
        _yMean = yMeanT;
        _yStd = yStdT;

        var yStandardized = new Vector<T>(centered);
        yStandardized.MultiplyInPlace(NumOps.Divide(NumOps.One, yStdT));
        y = yStandardized;

        // Initialize distribution and get number of parameters
        var initialDist = CreateDistribution(y);
        _numParams = initialDist.NumParameters;
        _initialParameters = initialDist.Parameters;

        var currentParams = _initialParameters.Take(_numParams).Select(value => Engine.Fill(n, value)).ToArray();

        _trees = [];
        FeatureImportances = new Vector<T>(x.Columns);

        T bestScore = NumOps.MaxValue;
        int roundsWithoutImprovement = 0;
        T lr = NumOps.FromDouble(_options.LearningRate);

        for (int iter = 0; iter < _options.NumberOfIterations; iter++)
        {
            int[] sampleIndices = GetSampleIndices(n);
            int sampleSize = sampleIndices.Length;

            // Accumulate Fisher Information Matrix
            var fisherSum = new Matrix<T>(_numParams, _numParams);

            // Score gradient of every sampled row's distribution, one row per sample
            var sampleParameters = StackParameters(currentParams).GetRows(sampleIndices);
            var sampleTargets = RegressionEngineOps.Gather(y, sampleIndices);
            var scoreGradients = new List<Vector<T>>(sampleSize);
            foreach (var (parameters, target) in sampleParameters.GetRows().Zip(sampleTargets, (row, value) => (row, value)))
            {
                // Create distribution with current parameters
                var dist = CreateDistributionWithParams(parameters);

                // Compute score gradient
                scoreGradients.Add(_scoringRule.ScoreGradient(dist, target).GetSubVector(0, _numParams));

                // Accumulate Fisher Information if using natural gradients
                if (_options.UseNaturalGradient)
                {
                    fisherSum = Engine.MatrixAdd(fisherSum, LeadingBlock(dist.FisherInformation()));
                }
            }

            // gradients[p] = the p-th score-gradient component over the sample
            var gradients = UnstackParameters(scoreGradients, sampleSize);

            // Compute natural gradients by preconditioning with Fisher
            var naturalGradients = gradients;
            if (_options.UseNaturalGradient)
            {
                naturalGradients = ComputeNaturalGradients(gradients, fisherSum, sampleSize);
            }

            // Build trees for each parameter
            var iterTrees = new DecisionTreeRegression<T>[_numParams];

            var xSample = x.GetRows(sampleIndices);

            for (int p = 0; p < _numParams; p++)
            {
                var residuals = new Vector<T>(naturalGradients[p]);
                residuals.MultiplyInPlace(NumOps.Negate(NumOps.One));

                var tree = new DecisionTreeRegression<T>(new DecisionTreeOptions
                {
                    MaxDepth = _options.MaxDepth,
                    MinSamplesSplit = _options.MinSamplesSplit,
                    MaxFeatures = _options.MaxFeatures,
                    SplitCriterion = _options.SplitCriterion,
                    Seed = Random.Next()
                });

                tree.Train(xSample, residuals);
                iterTrees[p] = tree;

                // Copy before mutating — tree may cache prediction vector
                var treePredictions = tree.Predict(x);
                var scaled = new Vector<T>(treePredictions);
                scaled.MultiplyInPlace(lr);
                currentParams[p].AddInPlace(scaled);
            }

            _trees.Add(iterTrees);

            // Early stopping check
            if (_options.EarlyStoppingRounds.HasValue)
            {
                T currentScore = ComputeMeanScore(currentParams, y);
                if (Improves(currentScore, bestScore))
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

        }

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <summary>
    /// Predicts the mean of the distribution for each input sample.
    /// </summary>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        var distributions = await PredictDistributionsAsync(input);
        if (distributions.Length == 0)
        {
            return new Vector<T>(0);
        }

        // Denormalize: prediction = standardized_mean * yStd + yMean
        var means = new Vector<T>(distributions.Select(distribution => distribution.Mean));
        return Engine.Add(Engine.Multiply(means, _yStd), Engine.Fill(means.Length, _yMean));
    }

    /// <summary>
    /// Predicts full probability distributions for each input sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <returns>Array of predicted distributions.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the full predictive distribution for each sample,
    /// allowing you to obtain prediction intervals, quantiles, and other
    /// probabilistic information.
    /// </para>
    /// <para><b>For Beginners:</b> This gives you the full probability distribution
    /// for each prediction. You can then:
    /// - Get the mean (point prediction)
    /// - Get prediction intervals (e.g., 95% confidence interval)
    /// - Sample possible outcomes
    /// - Compute the probability of outcomes in any range
    /// </para>
    /// </remarks>
    public async Task<IParametricDistribution<T>[]> PredictDistributionsAsync(Matrix<T> input)
    {
        int n = input.Rows;
        var distributions = new IParametricDistribution<T>[n];

        var currentParams = _initialParameters.Take(_numParams).Select(value => Engine.Fill(n, value)).ToArray();

        var treePredictionTasks = _trees.Select(iterTrees =>
            Task.Run(() =>
            {
                var treePreds = new Vector<T>[_numParams];
                for (int p = 0; p < _numParams; p++)
                    treePreds[p] = iterTrees[p].Predict(input);
                return treePreds;
            }));

        var allTreePredictions = await ParallelProcessingHelper.ProcessTasksInParallel(treePredictionTasks);

        T lr = NumOps.FromDouble(_options.LearningRate);
        foreach (var treePreds in allTreePredictions)
        {
            for (int p = 0; p < _numParams; p++)
            {
                var scaled = new Vector<T>(treePreds[p]);
                scaled.MultiplyInPlace(lr);
                currentParams[p].AddInPlace(scaled);
            }
        }

        // Create distributions
        if (n > 0)
        {
            distributions = StackParameters(currentParams, n).GetRows().Select(CreateDistributionWithParams).ToArray();
        }

        return distributions;
    }

    /// <summary>
    /// Predicts quantiles for each input sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="quantiles">Array of quantile levels (e.g., [0.1, 0.5, 0.9]).</param>
    /// <returns>Matrix where each row is a sample and each column is a quantile.</returns>
    public async Task<Matrix<T>> PredictQuantilesAsync(Matrix<T> input, Vector<double> quantiles)
    {
        var distributions = await PredictDistributionsAsync(input);
        var result = new Matrix<T>(input.Rows, quantiles.Length);

        for (int i = 0; i < input.Rows; i++)
        {
            for (int q = 0; q < quantiles.Length; q++)
            {
                result[i, q] = distributions[i].InverseCdf(NumOps.FromDouble(quantiles[q]));
            }
        }

        return result;
    }

    /// <summary>
    /// Gets prediction intervals for each input sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="confidenceLevel">Confidence level (e.g., 0.95 for 95% CI).</param>
    /// <returns>Tuple of (lower bounds, upper bounds) vectors.</returns>
    public async Task<(Vector<T> Lower, Vector<T> Upper)> PredictIntervalAsync(Matrix<T> input, double confidenceLevel = 0.95)
    {
        double alpha = 1 - confidenceLevel;
        var quantiles = new Vector<double>(new[] { alpha / 2, 1 - alpha / 2 });

        var quantileMatrix = await PredictQuantilesAsync(input, quantiles);

        if (input.Rows == 0)
        {
            return (new Vector<T>(0), new Vector<T>(0));
        }

        return (quantileMatrix.GetColumn(0), quantileMatrix.GetColumn(1));
    }

    /// <summary>
    /// Creates a distribution initialized from the target values.
    /// </summary>
    private IParametricDistribution<T> CreateDistribution(Vector<T> y)
    {
        T mean = Engine.Mean(y);

        var centered = new Vector<T>(y);
        centered.SubtractInPlace(Vector<T>.CreateDefault(y.Length, mean));
        T variance = NumOps.Divide(Engine.DotProduct(centered, centered), NumOps.FromDouble(y.Length));
        T minVariance = NumOps.FromDouble(MinVariance);
        if (NumOps.LessThan(variance, minVariance))
        {
            variance = minVariance;
        }

        // Distribution construction using NumOps for all math
        T minVar = NumOps.FromDouble(MinVariance);

        return _options.DistributionType switch
        {
            NGBoostDistributionType.Normal => new NormalDistribution<T>(mean, variance),
            NGBoostDistributionType.Laplace => new LaplaceDistribution<T>(
                mean, NumOps.Sqrt(NumOps.Divide(variance, NumOps.FromDouble(2)))),
            NGBoostDistributionType.StudentT => new StudentTDistribution<T>(
                mean, NumOps.Sqrt(variance), NumOps.FromDouble(MaxStdDevMultiplier)),
            NGBoostDistributionType.LogNormal => new LogNormalDistribution<T>(
                NumOps.Log(NumOps.GreaterThan(mean, minVar) ? mean : minVar),
                NumOps.Sqrt(NumOps.Log(NumOps.Add(NumOps.One,
                    NumOps.Divide(variance, NumOps.Add(NumOps.Multiply(mean, mean), minVar)))))),
            NGBoostDistributionType.Exponential => new ExponentialDistribution<T>(
                NumOps.Divide(NumOps.One, EnsurePositive(mean))),
            NGBoostDistributionType.Gamma => new GammaDistribution<T>(
                NumOps.Divide(NumOps.Multiply(mean, mean), variance),
                NumOps.Divide(mean, variance)),
            NGBoostDistributionType.Poisson => new PoissonDistribution<T>(EnsurePositive(mean)),
            _ => new NormalDistribution<T>(mean, variance)
        };
    }

    /// <summary>
    /// Stacks the per-parameter vectors as the columns of a [samples, numParameters] matrix.
    /// </summary>
    private Matrix<T> StackParameters(Vector<T>[] currentParams, int rowsWhenEmpty = 0)
    {
        // With no parameters (an untrained model) keep one empty row per sample, as before.
        int rows = currentParams.Length == 0 ? rowsWhenEmpty : currentParams[0].Length;
        if (currentParams.Length == 0)
        {
            return new Matrix<T>(rows, 0);
        }

        var columns = currentParams.Select(column => new Tensor<T>(new[] { rows, 1 }, column)).ToArray();
        return Engine.TensorConcatenate(columns, 1).ToMatrix();
    }

    /// <summary>
    /// Splits per-sample [numParams] vectors into numParams per-parameter vectors over the samples.
    /// </summary>
    private Vector<T>[] UnstackParameters(List<Vector<T>> perSample, int sampleCount)
    {
        if (sampleCount == 0 || _numParams == 0)
        {
            return Enumerable.Range(0, _numParams).Select(_ => new Vector<T>(sampleCount)).ToArray();
        }

        var stacked = new Tensor<T>(new[] { sampleCount, _numParams }, Vector<T>.Concatenate(perSample));
        return Engine.MatrixTranspose(stacked.ToMatrix()).GetRows().ToArray();
    }

    /// <summary>
    /// The leading [numParams, numParams] block of a Fisher information matrix.
    /// </summary>
    private Matrix<T> LeadingBlock(Matrix<T> fisher)
    {
        return fisher.Rows == _numParams && fisher.Columns == _numParams
            ? fisher
            : fisher.GetSubMatrix(0, 0, _numParams, _numParams);
    }

    /// <summary>
    /// Whether <paramref name="score"/> is strictly lower (better) than <paramref name="best"/>.
    /// </summary>
    private bool Improves(T score, T best)
    {
        return NumOps.LessThan(score, best);
    }

    /// <summary>
    /// Creates a distribution with the specified parameters.
    /// </summary>
    private IParametricDistribution<T> CreateDistributionWithParams(Vector<T> parameters)
    {
        return _options.DistributionType switch
        {
            NGBoostDistributionType.Normal => new NormalDistribution<T>(parameters[0], EnsurePositive(parameters[1])),
            NGBoostDistributionType.Laplace => new LaplaceDistribution<T>(parameters[0], EnsurePositive(parameters[1])),
            NGBoostDistributionType.StudentT => new StudentTDistribution<T>(parameters[0], EnsurePositive(parameters[1]), EnsurePositive(parameters[2])),
            NGBoostDistributionType.LogNormal => new LogNormalDistribution<T>(parameters[0], EnsurePositive(parameters[1])),
            NGBoostDistributionType.Exponential => new ExponentialDistribution<T>(EnsurePositive(parameters[0])),
            NGBoostDistributionType.Gamma => new GammaDistribution<T>(EnsurePositive(parameters[0]), EnsurePositive(parameters[1])),
            NGBoostDistributionType.Poisson => new PoissonDistribution<T>(EnsurePositive(parameters[0])),
            _ => new NormalDistribution<T>(parameters[0], EnsurePositive(parameters[1]))
        };
    }

    /// <summary>
    /// Ensures a parameter value is positive.
    /// </summary>
    private T EnsurePositive(T value)
    {
        T minVal = NumOps.FromDouble(MinVariance);
        return NumOps.LessThan(value, minVal) ? minVal : value;
    }

    /// <summary>
    /// Computes natural gradients by preconditioning with Fisher Information.
    /// </summary>
    private Vector<T>[] ComputeNaturalGradients(Vector<T>[] gradients, Matrix<T> fisherSum, int sampleSize)
    {
        var naturalGrads = new Vector<T>[_numParams];

        // Average Fisher Information Matrix
        T sampleSizeT = NumOps.FromDouble(sampleSize);
        var fisherAvg = Engine.TensorDivideScalar(Tensor<T>.FromMatrix(fisherSum), sampleSizeT).ToMatrix();

        // Invert Fisher Information Matrix (with regularization for stability)
        var fisherInv = InvertMatrix(fisherAvg);

        // Apply Fisher inverse to gradients: natural[p] = sum_q fisherInv[p, q] * gradients[q]
        int sampleCount = gradients[0].Length;
        if (sampleCount == 0)
        {
            for (int p = 0; p < _numParams; p++)
            {
                naturalGrads[p] = new Vector<T>(0);
            }

            return naturalGrads;
        }

        var gradientRows = Engine.TensorConcatenate(
            gradients.Select(g => new Tensor<T>(new[] { 1, sampleCount }, g)).ToArray(), 0).ToMatrix();
        return Engine.MatrixMultiply(fisherInv, gradientRows).GetRows().ToArray();
    }

    /// <summary>
    /// Inverts a small matrix with regularization.
    /// </summary>
    private Matrix<T> InvertMatrix(Matrix<T> matrix)
    {
        int n = matrix.Rows;
        T reg = NumOps.FromDouble(MatrixRegularization);
        T detThreshold = NumOps.FromDouble(DeterminantThreshold);

        // Add small regularization for numerical stability
        matrix = Engine.MatrixAdd(matrix, Matrix<T>.CreateDiagonal(Engine.Fill(n, reg)));

        // For small matrices (2x2 or 3x3), use analytical formulas
        if (n == 1)
        {
            var result1 = new Matrix<T>(1, 1);
            result1[0, 0] = NumOps.Divide(NumOps.One, matrix[0, 0]);
            return result1;
        }
        else if (n == 2)
        {
            T det = NumOps.Subtract(
                NumOps.Multiply(matrix[0, 0], matrix[1, 1]),
                NumOps.Multiply(matrix[0, 1], matrix[1, 0]));
            if (NumOps.LessThan(NumOps.Abs(det), detThreshold))
            {
                det = detThreshold;
            }
            var result2 = new Matrix<T>(2, 2);
            result2[0, 0] = NumOps.Divide(matrix[1, 1], det);
            result2[0, 1] = NumOps.Divide(NumOps.Negate(matrix[0, 1]), det);
            result2[1, 0] = NumOps.Divide(NumOps.Negate(matrix[1, 0]), det);
            result2[1, 1] = NumOps.Divide(matrix[0, 0], det);
            return result2;
        }

        // For larger matrices, use Gaussian elimination
        return GaussianElimination(matrix, n);
    }

    /// <summary>
    /// Matrix inversion using Gaussian elimination.
    /// </summary>
    private Matrix<T> GaussianElimination(Matrix<T> matrix, int n)
    {
        T pivotThreshold = NumOps.FromDouble(PivotThreshold);

        // Create augmented matrix [A | I]
        var augmented = Engine.TensorConcatenate(
            new[]
            {
                Tensor<T>.FromMatrix(matrix.GetSubMatrix(0, 0, n, n)),
                Tensor<T>.FromMatrix(Matrix<T>.CreateDiagonal(Engine.Fill(n, NumOps.One)))
            }, 1).ToMatrix();

        // Forward elimination, one pivot column at a time
        for (int col = 0; col < n; col++)
        {
            augmented = RegressionEngineOps.GaussJordanStep(augmented, col, n, pivotThreshold);
        }

        // Extract inverse
        return augmented.GetSubMatrix(0, n, n, n);
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

        int sampleSize = Math.Max(1, (int)(n * _options.SubsampleRatio));
        return SamplingHelper.SampleWithoutReplacement(n, sampleSize);
    }

    /// <summary>
    /// Computes the mean score for the current parameter values.
    /// </summary>
    private T ComputeMeanScore(Vector<T>[] currentParams, Vector<T> y)
    {
        T sum = NumOps.Zero;
        if (y.Length > 0)
        {
            var scores = StackParameters(currentParams).GetRows()
                .Zip(y, (parameters, target) => _scoringRule.Score(CreateDistributionWithParams(parameters), target))
                .ToList();
            sum = Engine.Sum(new Vector<T>(scores));
        }

        return NumOps.Divide(sum, NumOps.FromDouble(y.Length));
    }

    /// <inheritdoc/>
    protected override async Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(featureCount);

        // Aggregate importances from all trees
        var importanceTasks = _trees.SelectMany(iterTrees =>
            iterTrees.Select(tree => Task.Run(() =>
            {
                var fi = tree.FeatureImportances;
                int copyCount = Math.Min(featureCount, fi.Length);
                if (copyCount == featureCount)
                {
                    return fi.GetSubVector(0, featureCount);
                }

                // Pad a shorter importance vector with zeros
                return Vector<T>.Concatenate(new[] { fi.GetSubVector(0, copyCount), new Vector<T>(featureCount - copyCount) });
            })));

        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(importanceTasks);

        foreach (var ti in allImportances)
        {
            importances = Engine.Add(importances, ti);
        }

        // Normalize
        T sum = featureCount == 0 ? NumOps.Zero : Engine.Sum(importances);
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            importances = Engine.Divide(importances, sum);
        }

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
                { "NumberOfParameters", _numParams },
                { "DistributionType", _options.DistributionType.ToString() },
                { "ScoringRule", _scoringRule.Name },
                { "LearningRate", _options.LearningRate },
                { "MaxDepth", _options.MaxDepth },
                { "UseNaturalGradient", _options.UseNaturalGradient }
            }
        };
    }

}
