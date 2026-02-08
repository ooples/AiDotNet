using AiDotNet.Distributions;
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
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class NGBoostRegression<T> : AsyncDecisionTreeRegressionBase<T>
{
    /// <summary>
    /// Base learners for each distribution parameter.
    /// </summary>
    private List<DecisionTreeRegression<T>[]> _trees;

    /// <summary>
    /// Initial parameter values (e.g., mean of y for location, initial scale).
    /// </summary>
    private Vector<T> _initialParameters;

    /// <summary>
    /// The scoring rule used for optimization.
    /// </summary>
    private readonly IScoringRule<T> _scoringRule;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly NGBoostRegressionOptions _options;

    /// <summary>
    /// Number of parameters in the distribution.
    /// </summary>
    private int _numParams;

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
    public NGBoostRegression(NGBoostRegressionOptions? options = null, IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new NGBoostRegressionOptions();
        _trees = [];
        _initialParameters = new Vector<T>(0);
        _numParams = 0;

        // Initialize scoring rule
        _scoringRule = _options.ScoringRule switch
        {
            NGBoostScoringRuleType.LogScore => new LogScore<T>(),
            NGBoostScoringRuleType.CRPS => new CRPSScore<T>(),
            _ => new LogScore<T>()
        };
    }

    /// <summary>
    /// Trains the NGBoost model using natural gradient boosting.
    /// </summary>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        int n = x.Rows;

        // Initialize distribution and get number of parameters
        var initialDist = CreateDistribution(y);
        _numParams = initialDist.NumParameters;
        _initialParameters = initialDist.Parameters;

        // Initialize current parameter predictions for all samples
        var currentParams = new Vector<T>[_numParams];
        for (int p = 0; p < _numParams; p++)
        {
            currentParams[p] = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                currentParams[p][i] = _initialParameters[p];
            }
        }

        _trees = [];
        FeatureImportances = new Vector<T>(x.Columns);

        double bestScore = double.MaxValue;
        int roundsWithoutImprovement = 0;

        for (int iter = 0; iter < _options.NumberOfIterations; iter++)
        {
            // Subsample if needed
            int[] sampleIndices = GetSampleIndices(n);
            int sampleSize = sampleIndices.Length;

            // Compute scores and gradients for the sample
            var scores = new Vector<T>(sampleSize);
            var gradients = new Vector<T>[_numParams];
            for (int p = 0; p < _numParams; p++)
            {
                gradients[p] = new Vector<T>(sampleSize);
            }

            // Accumulate Fisher Information Matrix
            var fisherSum = new Matrix<T>(_numParams, _numParams);

            for (int i = 0; i < sampleSize; i++)
            {
                int idx = sampleIndices[i];

                // Create distribution with current parameters
                var dist = CreateDistributionFromParams(currentParams, idx);

                // Compute score gradient
                var scoreGrad = _scoringRule.ScoreGradient(dist, y[idx]);

                for (int p = 0; p < _numParams; p++)
                {
                    gradients[p][i] = scoreGrad[p];
                }

                // Accumulate Fisher Information if using natural gradients
                if (_options.UseNaturalGradient)
                {
                    var fisher = dist.FisherInformation();
                    for (int p1 = 0; p1 < _numParams; p1++)
                    {
                        for (int p2 = 0; p2 < _numParams; p2++)
                        {
                            fisherSum[p1, p2] = NumOps.Add(fisherSum[p1, p2], fisher[p1, p2]);
                        }
                    }
                }
            }

            // Compute natural gradients by preconditioning with Fisher
            var naturalGradients = gradients;
            if (_options.UseNaturalGradient)
            {
                naturalGradients = ComputeNaturalGradients(gradients, fisherSum, sampleSize);
            }

            // Build trees for each parameter
            var iterTrees = new DecisionTreeRegression<T>[_numParams];

            for (int p = 0; p < _numParams; p++)
            {
                // Create pseudo-residuals (natural gradients)
                var residuals = new Vector<T>(sampleSize);
                for (int i = 0; i < sampleSize; i++)
                {
                    residuals[i] = NumOps.Negate(naturalGradients[p][i]);
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
                    Seed = Random.Next()
                });

                tree.Train(xSample, residuals);
                iterTrees[p] = tree;

                // Update parameters for all samples (not just the subsample)
                var treePredictions = tree.Predict(x);
                for (int i = 0; i < n; i++)
                {
                    currentParams[p][i] = NumOps.Add(
                        currentParams[p][i],
                        NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), treePredictions[i]));
                }
            }

            _trees.Add(iterTrees);

            // Early stopping check
            if (_options.EarlyStoppingRounds.HasValue)
            {
                double currentScore = ComputeMeanScore(currentParams, y);
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
                double score = ComputeMeanScore(currentParams, y);
                Console.WriteLine($"[{iter + 1}] {_scoringRule.Name}: {score:F6}");
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
        var predictions = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = distributions[i].Mean;
        }

        return predictions;
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

        // Initialize parameters
        var currentParams = new Vector<T>[_numParams];
        for (int p = 0; p < _numParams; p++)
        {
            currentParams[p] = new Vector<T>(n);
            for (int i = 0; i < n; i++)
            {
                currentParams[p][i] = _initialParameters[p];
            }
        }

        // Accumulate tree predictions
        var treePredictionTasks = _trees.Select(iterTrees =>
            Task.Run(() =>
            {
                var treePreds = new Vector<T>[_numParams];
                for (int p = 0; p < _numParams; p++)
                {
                    treePreds[p] = iterTrees[p].Predict(input);
                }
                return treePreds;
            }));

        var allTreePredictions = await ParallelProcessingHelper.ProcessTasksInParallel(treePredictionTasks);

        foreach (var treePreds in allTreePredictions)
        {
            for (int p = 0; p < _numParams; p++)
            {
                for (int i = 0; i < n; i++)
                {
                    currentParams[p][i] = NumOps.Add(
                        currentParams[p][i],
                        NumOps.Multiply(NumOps.FromDouble(_options.LearningRate), treePreds[p][i]));
                }
            }
        }

        // Create distributions
        for (int i = 0; i < n; i++)
        {
            distributions[i] = CreateDistributionFromParams(currentParams, i);
        }

        return distributions;
    }

    /// <summary>
    /// Predicts quantiles for each input sample.
    /// </summary>
    /// <param name="input">Input feature matrix.</param>
    /// <param name="quantiles">Array of quantile levels (e.g., [0.1, 0.5, 0.9]).</param>
    /// <returns>Matrix where each row is a sample and each column is a quantile.</returns>
    public async Task<Matrix<T>> PredictQuantilesAsync(Matrix<T> input, double[] quantiles)
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
        double[] quantiles = [alpha / 2, 1 - alpha / 2];

        var quantileMatrix = await PredictQuantilesAsync(input, quantiles);

        var lower = new Vector<T>(input.Rows);
        var upper = new Vector<T>(input.Rows);

        for (int i = 0; i < input.Rows; i++)
        {
            lower[i] = quantileMatrix[i, 0];
            upper[i] = quantileMatrix[i, 1];
        }

        return (lower, upper);
    }

    /// <summary>
    /// Creates a distribution initialized from the target values.
    /// </summary>
    private IParametricDistribution<T> CreateDistribution(Vector<T> y)
    {
        double mean = 0, variance = 0;
        for (int i = 0; i < y.Length; i++)
        {
            mean += NumOps.ToDouble(y[i]);
        }
        mean /= y.Length;

        for (int i = 0; i < y.Length; i++)
        {
            double diff = NumOps.ToDouble(y[i]) - mean;
            variance += diff * diff;
        }
        variance /= y.Length;
        variance = Math.Max(variance, 1e-6); // Prevent zero variance

        return _options.DistributionType switch
        {
            NGBoostDistributionType.Normal => new NormalDistribution<T>(
                NumOps.FromDouble(mean),
                NumOps.FromDouble(variance)),
            NGBoostDistributionType.Laplace => new LaplaceDistribution<T>(
                NumOps.FromDouble(mean),
                NumOps.FromDouble(Math.Sqrt(variance / 2))),
            NGBoostDistributionType.StudentT => new StudentTDistribution<T>(
                NumOps.FromDouble(mean),
                NumOps.FromDouble(Math.Sqrt(variance)),
                NumOps.FromDouble(4.0)),  // 4 degrees of freedom
            NGBoostDistributionType.LogNormal => new LogNormalDistribution<T>(
                NumOps.FromDouble(Math.Log(Math.Max(mean, 1e-6))),
                NumOps.FromDouble(Math.Sqrt(Math.Log(1 + variance / (mean * mean + 1e-6))))),
            NGBoostDistributionType.Exponential => new ExponentialDistribution<T>(
                NumOps.FromDouble(1.0 / Math.Max(mean, 1e-6))),
            NGBoostDistributionType.Gamma => new GammaDistribution<T>(
                NumOps.FromDouble(mean * mean / variance),
                NumOps.FromDouble(mean / variance)),
            NGBoostDistributionType.Poisson => new PoissonDistribution<T>(
                NumOps.FromDouble(Math.Max(mean, 1e-6))),
            _ => new NormalDistribution<T>(NumOps.FromDouble(mean), NumOps.FromDouble(variance))
        };
    }

    /// <summary>
    /// Creates a distribution from the current parameter values for a specific sample.
    /// </summary>
    private IParametricDistribution<T> CreateDistributionFromParams(Vector<T>[] currentParams, int sampleIndex)
    {
        var params_ = new Vector<T>(_numParams);
        for (int p = 0; p < _numParams; p++)
        {
            params_[p] = currentParams[p][sampleIndex];
        }

        return CreateDistributionWithParams(params_);
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
        double v = NumOps.ToDouble(value);
        return NumOps.FromDouble(Math.Max(v, 1e-6));
    }

    /// <summary>
    /// Computes natural gradients by preconditioning with Fisher Information.
    /// </summary>
    private Vector<T>[] ComputeNaturalGradients(Vector<T>[] gradients, Matrix<T> fisherSum, int sampleSize)
    {
        var naturalGrads = new Vector<T>[_numParams];

        // Average Fisher Information Matrix
        var fisherAvg = new double[_numParams, _numParams];
        for (int i = 0; i < _numParams; i++)
        {
            for (int j = 0; j < _numParams; j++)
            {
                fisherAvg[i, j] = NumOps.ToDouble(fisherSum[i, j]) / sampleSize;
            }
        }

        // Invert Fisher Information Matrix (with regularization for stability)
        var fisherInv = InvertMatrix(fisherAvg);

        // Apply Fisher inverse to gradients
        for (int p = 0; p < _numParams; p++)
        {
            naturalGrads[p] = new Vector<T>(gradients[0].Length);
        }

        for (int i = 0; i < gradients[0].Length; i++)
        {
            for (int p = 0; p < _numParams; p++)
            {
                double sum = 0;
                for (int q = 0; q < _numParams; q++)
                {
                    sum += fisherInv[p, q] * NumOps.ToDouble(gradients[q][i]);
                }
                naturalGrads[p][i] = NumOps.FromDouble(sum);
            }
        }

        return naturalGrads;
    }

    /// <summary>
    /// Inverts a small matrix with regularization.
    /// </summary>
    private double[,] InvertMatrix(double[,] matrix)
    {
        int n = matrix.GetLength(0);

        // Add small regularization for numerical stability
        for (int i = 0; i < n; i++)
        {
            matrix[i, i] += 1e-4;
        }

        // For small matrices (2x2 or 3x3), use analytical formulas
        if (n == 1)
        {
            return new double[,] { { 1.0 / matrix[0, 0] } };
        }
        else if (n == 2)
        {
            double det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0];
            if (Math.Abs(det) < 1e-10) det = 1e-10;
            return new double[,]
            {
                { matrix[1, 1] / det, -matrix[0, 1] / det },
                { -matrix[1, 0] / det, matrix[0, 0] / det }
            };
        }

        // For larger matrices, use Gaussian elimination
        return GaussianElimination(matrix, n);
    }

    /// <summary>
    /// Matrix inversion using Gaussian elimination.
    /// </summary>
    private double[,] GaussianElimination(double[,] matrix, int n)
    {
        // Create augmented matrix [A | I]
        var augmented = new double[n, 2 * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                augmented[i, j] = matrix[i, j];
            }
            augmented[i, n + i] = 1.0;
        }

        // Forward elimination
        for (int col = 0; col < n; col++)
        {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(augmented[row, col]) > Math.Abs(augmented[maxRow, col]))
                {
                    maxRow = row;
                }
            }

            // Swap rows
            for (int j = 0; j < 2 * n; j++)
            {
                (augmented[col, j], augmented[maxRow, j]) = (augmented[maxRow, j], augmented[col, j]);
            }

            // Scale pivot row
            double pivot = augmented[col, col];
            if (Math.Abs(pivot) < 1e-10) pivot = 1e-10;
            for (int j = 0; j < 2 * n; j++)
            {
                augmented[col, j] /= pivot;
            }

            // Eliminate column
            for (int row = 0; row < n; row++)
            {
                if (row != col)
                {
                    double factor = augmented[row, col];
                    for (int j = 0; j < 2 * n; j++)
                    {
                        augmented[row, j] -= factor * augmented[col, j];
                    }
                }
            }
        }

        // Extract inverse
        var inverse = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                inverse[i, j] = augmented[i, n + j];
            }
        }

        return inverse;
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
    /// Computes the mean score for the current parameter values.
    /// </summary>
    private double ComputeMeanScore(Vector<T>[] currentParams, Vector<T> y)
    {
        double sum = 0;
        for (int i = 0; i < y.Length; i++)
        {
            var dist = CreateDistributionFromParams(currentParams, i);
            sum += NumOps.ToDouble(_scoringRule.Score(dist, y[i]));
        }
        return sum / y.Length;
    }

    /// <inheritdoc/>
    protected override async Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(featureCount);

        // Aggregate importances from all trees
        var importanceTasks = _trees.SelectMany(iterTrees =>
            iterTrees.Select(tree => Task.Run(() =>
            {
                var treeImportances = new Vector<T>(featureCount);
                var fi = tree.FeatureImportances;
                int copyCount = Math.Min(featureCount, fi.Length);
                for (int i = 0; i < copyCount; i++)
                {
                    treeImportances[i] = fi[i];
                }
                return treeImportances;
            })));

        var allImportances = await ParallelProcessingHelper.ProcessTasksInParallel(importanceTasks);

        for (int i = 0; i < featureCount; i++)
        {
            foreach (var ti in allImportances)
            {
                importances[i] = NumOps.Add(importances[i], ti[i]);
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
            ModelType = ModelType.NGBoost,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfIterations", _trees.Count },
                { "NumberOfParameters", _numParams },
                { "DistributionType", _options.DistributionType.ToString() },
                { "ScoringRule", _options.ScoringRule.ToString() },
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
        writer.Write((int)_options.DistributionType);
        writer.Write((int)_options.ScoringRule);
        writer.Write(_options.UseNaturalGradient);

        // Initial parameters
        writer.Write(_numParams);
        for (int p = 0; p < _numParams; p++)
        {
            writer.Write(NumOps.ToDouble(_initialParameters[p]));
        }

        // Trees
        writer.Write(_trees.Count);
        foreach (var iterTrees in _trees)
        {
            for (int p = 0; p < _numParams; p++)
            {
                byte[] treeData = iterTrees[p].Serialize();
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
        _options.DistributionType = (NGBoostDistributionType)reader.ReadInt32();
        _options.ScoringRule = (NGBoostScoringRuleType)reader.ReadInt32();
        _options.UseNaturalGradient = reader.ReadBoolean();

        // Initial parameters
        _numParams = reader.ReadInt32();
        _initialParameters = new Vector<T>(_numParams);
        for (int p = 0; p < _numParams; p++)
        {
            _initialParameters[p] = NumOps.FromDouble(reader.ReadDouble());
        }

        // Trees
        int numIter = reader.ReadInt32();
        _trees = new List<DecisionTreeRegression<T>[]>(numIter);
        for (int iter = 0; iter < numIter; iter++)
        {
            var iterTrees = new DecisionTreeRegression<T>[_numParams];
            for (int p = 0; p < _numParams; p++)
            {
                int treeLen = reader.ReadInt32();
                byte[] treeData = reader.ReadBytes(treeLen);
                iterTrees[p] = new DecisionTreeRegression<T>(new DecisionTreeOptions());
                iterTrees[p].Deserialize(treeData);
            }
            _trees.Add(iterTrees);
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new NGBoostRegression<T>(_options, Regularization);
    }
}
