using AiDotNet.Classification.Ensemble;
using AiDotNet.Models.Options;

namespace AiDotNet.Classification.Boosting;

/// <summary>
/// Explainable Boosting Machine (EBM) for interpretable classification.
/// </summary>
/// <remarks>
/// <para>
/// EBM is a Generalized Additive Model (GAM) with boosting that provides glass-box
/// interpretability while maintaining high accuracy. It learns smooth functions for
/// each feature and optionally pairwise interactions.
/// </para>
/// <para>
/// <b>For Beginners:</b> EBM is special because it gives you the best of both worlds:
/// - High accuracy (comparable to gradient boosting and random forests)
/// - Full interpretability (you can see exactly why each prediction was made)
///
/// How it works:
/// 1. For each feature, EBM learns a "shape function" that shows how that feature
///    affects the prediction
/// 2. The final prediction is simply the sum of all these shape functions plus
///    an intercept, passed through sigmoid for probability
/// 3. You can plot these shape functions to understand exactly how the model
///    uses each feature
///
/// For example, in predicting loan defaults:
/// - The shape function for "income" might show higher income = lower risk
/// - The shape function for "debt_ratio" might show higher ratio = higher risk
/// - The prediction combines: intercept + f(income) + f(debt_ratio) + ...
///
/// This additive structure makes EBM uniquely interpretable while still being accurate.
/// </para>
/// <para>
/// Reference: Lou, Y., et al. "Intelligible Models for Healthcare: Predicting
/// Pneumonia Risk and Hospital 30-day Readmission" (2012).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExplainableBoostingClassifier<T> : EnsembleClassifierBase<T>
{
    /// <summary>
    /// Shape functions for each feature (additive terms).
    /// Indexed as: _shapeFunction[featureIndex][binIndex]
    /// </summary>
    private T[][] _shapeFunctions;

    /// <summary>
    /// Bin edges for each feature.
    /// Indexed as: _binEdges[featureIndex][edgeIndex]
    /// </summary>
    private T[][] _binEdges;

    /// <summary>
    /// Interaction terms: pairs of features and their joint effect.
    /// </summary>
    private Dictionary<(int, int), T[,]> _interactionTerms;

    /// <summary>
    /// The intercept (baseline log-odds).
    /// </summary>
    private T _intercept;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly ExplainableBoostingClassifierOptions<T> _options;

    /// <summary>
    /// Random number generator.
    /// </summary>
    private Random _random;

    /// <summary>
    /// Gets the intercept (baseline log-odds).
    /// </summary>
    public T Intercept => _intercept;

    /// <summary>
    /// Gets the shape functions for each feature.
    /// </summary>
    public IReadOnlyList<T[]> ShapeFunctions => _shapeFunctions;

    /// <summary>
    /// Gets the bin edges for each feature.
    /// </summary>
    public IReadOnlyList<T[]> BinEdges => _binEdges;

    /// <summary>
    /// Gets the interaction terms.
    /// </summary>
    public IReadOnlyDictionary<(int, int), T[,]> InteractionTerms => _interactionTerms;

    /// <summary>
    /// Initializes a new instance of ExplainableBoostingClassifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public ExplainableBoostingClassifier(
        ExplainableBoostingClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ?? new ExplainableBoostingClassifierOptions<T>(), regularization, new CrossEntropyLoss<T>())
    {
        _options = options ?? new ExplainableBoostingClassifierOptions<T>();
        _shapeFunctions = [];
        _binEdges = [];
        _interactionTerms = [];
        _intercept = NumOps.Zero;
        _numFeatures = 0;
        _random = RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Returns the model type identifier.
    /// </summary>
    protected override ModelType GetModelType() => ModelType.ExplainableBoostingClassifier;

    /// <summary>
    /// Trains the EBM classifier using cyclic gradient boosting.
    /// </summary>
    public override void Train(Matrix<T> x, Vector<T> y)
    {
        if (x.Rows != y.Length)
        {
            throw new ArgumentException("Number of samples in X must match length of y.");
        }

        _numFeatures = x.Columns;
        int n = x.Rows;
        NumFeatures = _numFeatures;
        ClassLabels = ExtractClassLabels(y);
        NumClasses = ClassLabels.Length;
        TaskType = InferTaskType(y);

        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        // Convert to binary labels
        var yBinary = new double[n];
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        int posCount = 0;
        for (int i = 0; i < n; i++)
        {
            yBinary[i] = NumOps.Compare(y[i], positiveClass) == 0 ? 1.0 : 0.0;
            if (yBinary[i] == 1.0) posCount++;
        }

        // Initialize intercept as log-odds of prior
        double p = Math.Max(1e-10, Math.Min(1 - 1e-10, (double)posCount / n));
        _intercept = NumOps.FromDouble(Math.Log(p / (1 - p)));

        // Create bins for each feature
        CreateBins(x);

        // Initialize shape functions to zero
        _shapeFunctions = new T[_numFeatures][];
        for (int f = 0; f < _numFeatures; f++)
        {
            _shapeFunctions[f] = new T[_binEdges[f].Length + 1];
            for (int b = 0; b <= _binEdges[f].Length; b++)
            {
                _shapeFunctions[f][b] = NumOps.Zero;
            }
        }

        // Initialize interaction terms
        _interactionTerms = [];

        // Map samples to bins for each feature
        var sampleBins = new int[_numFeatures][];
        for (int f = 0; f < _numFeatures; f++)
        {
            sampleBins[f] = new int[n];
            for (int i = 0; i < n; i++)
            {
                sampleBins[f][i] = GetBinIndex(NumOps.ToDouble(x[i, f]), f);
            }
        }

        // Cyclic boosting over features
        double bestLoss = double.MaxValue;
        int roundsWithoutImprovement = 0;

        // Maintain running log-odds to avoid recomputing from scratch each iteration
        var runningLogOdds = new double[n];
        for (int i = 0; i < n; i++)
        {
            runningLogOdds[i] = NumOps.ToDouble(_intercept);
            for (int f = 0; f < _numFeatures; f++)
            {
                runningLogOdds[i] += NumOps.ToDouble(_shapeFunctions[f][sampleBins[f][i]]);
            }
        }

        for (int outer = 0; outer < _options.OuterBags; outer++)
        {
            for (int inner = 0; inner < _options.InnerBags; inner++)
            {
                // Cycle through features in round-robin fashion
                for (int f = 0; f < _numFeatures; f++)
                {
                    // Compute gradient and hessian for each bin using running log-odds
                    var binGradients = new double[_binEdges[f].Length + 1];
                    var binHessians = new double[_binEdges[f].Length + 1];

                    for (int i = 0; i < n; i++)
                    {
                        double prob = Sigmoid(runningLogOdds[i]);
                        double grad = yBinary[i] - prob;  // Negative gradient of log-loss
                        double hess = prob * (1 - prob);   // Hessian

                        int bin = sampleBins[f][i];
                        binGradients[bin] += grad;
                        binHessians[bin] += hess;
                    }

                    // Update shape function for this feature and adjust running log-odds
                    for (int b = 0; b <= _binEdges[f].Length; b++)
                    {
                        if (binHessians[b] > 1e-10)
                        {
                            // Newton step with L2 regularization
                            double update = (binGradients[b] / (binHessians[b] + _options.L2Regularization))
                                           * _options.LearningRate;
                            _shapeFunctions[f][b] = NumOps.Add(_shapeFunctions[f][b], NumOps.FromDouble(update));
                        }
                    }

                    // Update running log-odds incrementally for this feature's changes
                    for (int i = 0; i < n; i++)
                    {
                        int bin = sampleBins[f][i];
                        if (binHessians[bin] > 1e-10)
                        {
                            double update = (binGradients[bin] / (binHessians[bin] + _options.L2Regularization))
                                           * _options.LearningRate;
                            runningLogOdds[i] += update;
                        }
                    }
                }
            }

            // Early stopping check
            if (_options.EarlyStoppingRounds.HasValue)
            {
                double loss = ComputeLogLoss(runningLogOdds, yBinary);

                if (loss < bestLoss)
                {
                    bestLoss = loss;
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

            if (_options.Verbose && (outer + 1) % _options.VerboseEval == 0)
            {
                double loss = ComputeLogLoss(runningLogOdds, yBinary);
                Console.WriteLine($"[{outer + 1}] Log Loss: {loss:F6}");
            }
        }

        // Detect and add interaction terms if enabled
        if (_options.MaxInteractions > 0)
        {
            DetectInteractions(x, sampleBins, yBinary);
        }

        // Center shape functions (subtract mean to make intercept interpretable)
        CenterShapeFunctions(sampleBins, n);

        // Calculate feature importances
        CalculateFeatureImportances();
    }

    /// <summary>
    /// Predicts class labels for input samples.
    /// </summary>
    public override Vector<T> Predict(Matrix<T> input)
    {
        var probs = PredictProbabilities(input);
        var predictions = new Vector<T>(input.Rows);

        var classLabels = ClassLabels ?? throw new InvalidOperationException("Model must be trained before making predictions.");
        T threshold = NumOps.FromDouble(0.5);
        for (int i = 0; i < input.Rows; i++)
        {
            predictions[i] = NumOps.Compare(probs[i, 1], threshold) >= 0
                ? classLabels[classLabels.Length - 1]
                : classLabels[0];
        }

        return predictions;
    }

    /// <summary>
    /// Predicts class probabilities for input samples.
    /// </summary>
    public override Matrix<T> PredictProbabilities(Matrix<T> input)
    {
        if (NumClasses != 2)
        {
            throw new NotSupportedException(
                $"EBM classifier currently supports binary classification only (NumClasses={NumClasses}).");
        }

        int n = input.Rows;
        var probs = new Matrix<T>(n, NumClasses);

        for (int i = 0; i < n; i++)
        {
            double logOdds = NumOps.ToDouble(_intercept);

            // Add main effects
            for (int f = 0; f < _numFeatures; f++)
            {
                int bin = GetBinIndex(NumOps.ToDouble(input[i, f]), f);
                logOdds += NumOps.ToDouble(_shapeFunctions[f][bin]);
            }

            // Add interaction effects
            foreach (var kvp in _interactionTerms)
            {
                int f1 = kvp.Key.Item1;
                int f2 = kvp.Key.Item2;
                int bin1 = GetBinIndex(NumOps.ToDouble(input[i, f1]), f1);
                int bin2 = GetBinIndex(NumOps.ToDouble(input[i, f2]), f2);

                // Ensure we don't go out of bounds
                int maxBin1 = kvp.Value.GetLength(0) - 1;
                int maxBin2 = kvp.Value.GetLength(1) - 1;
                bin1 = Math.Min(bin1, maxBin1);
                bin2 = Math.Min(bin2, maxBin2);

                logOdds += NumOps.ToDouble(kvp.Value[bin1, bin2]);
            }

            double prob1 = Sigmoid(logOdds);
            probs[i, 0] = NumOps.FromDouble(1 - prob1);
            probs[i, 1] = NumOps.FromDouble(prob1);
        }

        return probs;
    }

    /// <summary>
    /// Explains a single prediction by showing each feature's contribution.
    /// </summary>
    /// <param name="sample">A single input sample.</param>
    /// <returns>Dictionary mapping feature index to its contribution.</returns>
    public Dictionary<string, T> ExplainPrediction(Vector<T> sample)
    {
        var contributions = new Dictionary<string, T>();

        // Main effects
        for (int f = 0; f < _numFeatures; f++)
        {
            int bin = GetBinIndex(NumOps.ToDouble(sample[f]), f);
            contributions[$"feature_{f}"] = _shapeFunctions[f][bin];
        }

        // Interaction effects
        foreach (var kvp in _interactionTerms)
        {
            int f1 = kvp.Key.Item1;
            int f2 = kvp.Key.Item2;
            int bin1 = GetBinIndex(NumOps.ToDouble(sample[f1]), f1);
            int bin2 = GetBinIndex(NumOps.ToDouble(sample[f2]), f2);

            int maxBin1 = kvp.Value.GetLength(0) - 1;
            int maxBin2 = kvp.Value.GetLength(1) - 1;
            bin1 = Math.Min(bin1, maxBin1);
            bin2 = Math.Min(bin2, maxBin2);

            contributions[$"interaction_{f1}x{f2}"] = kvp.Value[bin1, bin2];
        }

        return contributions;
    }

    /// <summary>
    /// Creates bins for each feature using quantiles.
    /// </summary>
    private void CreateBins(Matrix<T> x)
    {
        _binEdges = new T[_numFeatures][];

        for (int f = 0; f < _numFeatures; f++)
        {
            // Collect unique values
            var values = new List<double>();
            for (int i = 0; i < x.Rows; i++)
            {
                values.Add(NumOps.ToDouble(x[i, f]));
            }
            values.Sort();
            values = values.Distinct().ToList();

            // Create quantile-based bins
            int numBins = Math.Min(_options.MaxBins, values.Count);
            if (numBins <= 1)
            {
                _binEdges[f] = [];
                continue;
            }

            var edges = new List<T>();
            for (int b = 1; b < numBins; b++)
            {
                int idx = (int)((double)b / numBins * values.Count);
                idx = Math.Max(0, Math.Min(idx, values.Count - 1));
                edges.Add(NumOps.FromDouble(values[idx]));
            }
            _binEdges[f] = edges.Distinct().ToArray();
        }
    }

    /// <summary>
    /// Gets the bin index for a value.
    /// </summary>
    private int GetBinIndex(double value, int featureIndex)
    {
        var edges = _binEdges[featureIndex];
        for (int i = 0; i < edges.Length; i++)
        {
            if (value < NumOps.ToDouble(edges[i]))
            {
                return i;
            }
        }
        return edges.Length;
    }

    /// <summary>
    /// Computes log-odds for all samples.
    /// </summary>
    private double[] ComputeLogOdds(Matrix<T> x, int[][] sampleBins)
    {
        int n = x.Rows;
        var logOdds = new double[n];

        for (int i = 0; i < n; i++)
        {
            logOdds[i] = NumOps.ToDouble(_intercept);
            for (int f = 0; f < _numFeatures; f++)
            {
                logOdds[i] += NumOps.ToDouble(_shapeFunctions[f][sampleBins[f][i]]);
            }
        }

        return logOdds;
    }

    /// <summary>
    /// Sigmoid function.
    /// </summary>
    private static double Sigmoid(double x)
    {
        if (x >= 0)
        {
            double ez = Math.Exp(-x);
            return 1.0 / (1.0 + ez);
        }
        else
        {
            double ez = Math.Exp(x);
            return ez / (1.0 + ez);
        }
    }

    /// <summary>
    /// Computes log loss.
    /// </summary>
    private static double ComputeLogLoss(double[] logOdds, double[] yBinary)
    {
        double loss = 0;
        for (int i = 0; i < logOdds.Length; i++)
        {
            double p = Sigmoid(logOdds[i]);
            p = Math.Max(1e-10, Math.Min(1 - 1e-10, p));
            loss -= yBinary[i] * Math.Log(p) + (1 - yBinary[i]) * Math.Log(1 - p);
        }
        return loss / logOdds.Length;
    }

    /// <summary>
    /// Detects important pairwise interactions.
    /// </summary>
    private void DetectInteractions(Matrix<T> x, int[][] sampleBins, double[] yBinary)
    {
        // Simple heuristic: detect interactions that reduce residual variance
        // In production, would use FAST or similar algorithm

        int n = x.Rows;
        var logOdds = ComputeLogOdds(x, sampleBins);
        var residuals = new double[n];
        for (int i = 0; i < n; i++)
        {
            residuals[i] = yBinary[i] - Sigmoid(logOdds[i]);
        }

        // Check top feature pairs
        var candidates = new List<(int, int, double)>();

        int maxFeaturesToConsider = Math.Min(_numFeatures, Math.Max(0, _options.MaxInteractionFeatures));
        if (maxFeaturesToConsider < 2)
        {
            return;
        }

        for (int f1 = 0; f1 < maxFeaturesToConsider; f1++)
        {
            for (int f2 = f1 + 1; f2 < maxFeaturesToConsider; f2++)
            {
                // Simple correlation-based score
                double score = ComputeInteractionScore(sampleBins[f1], sampleBins[f2], residuals);
                candidates.Add((f1, f2, score));
            }
        }

        // Add top interactions
        var topInteractions = candidates.OrderByDescending(c => c.Item3).Take(_options.MaxInteractions);

        foreach (var (f1, f2, _) in topInteractions)
        {
            int numBins1 = _binEdges[f1].Length + 1;
            int numBins2 = _binEdges[f2].Length + 1;

            // Limit interaction bins
            numBins1 = Math.Min(numBins1, _options.MaxInteractionBins);
            numBins2 = Math.Min(numBins2, _options.MaxInteractionBins);

            var interactionTerm = new T[numBins1, numBins2];

            // Initialize with gradient boosting
            var binGradients = new double[numBins1, numBins2];
            var binCounts = new int[numBins1, numBins2];

            for (int i = 0; i < n; i++)
            {
                int b1 = Math.Min(sampleBins[f1][i], numBins1 - 1);
                int b2 = Math.Min(sampleBins[f2][i], numBins2 - 1);
                binGradients[b1, b2] += residuals[i];
                binCounts[b1, b2]++;
            }

            for (int b1 = 0; b1 < numBins1; b1++)
            {
                for (int b2 = 0; b2 < numBins2; b2++)
                {
                    if (binCounts[b1, b2] > 0)
                    {
                        double update = (binGradients[b1, b2] / binCounts[b1, b2]) * _options.LearningRate;
                        interactionTerm[b1, b2] = NumOps.FromDouble(update);
                    }
                }
            }

            _interactionTerms[(f1, f2)] = interactionTerm;
        }
    }

    /// <summary>
    /// Computes interaction score (variance reduction).
    /// </summary>
    private static double ComputeInteractionScore(int[] bins1, int[] bins2, double[] residuals)
    {
        // Group residuals by bin pair and compute variance reduction
        var groups = new Dictionary<(int, int), List<double>>();
        for (int i = 0; i < residuals.Length; i++)
        {
            var key = (bins1[i], bins2[i]);
            if (!groups.ContainsKey(key))
            {
                groups[key] = [];
            }
            groups[key].Add(residuals[i]);
        }

        double totalVar = ComputeVariance(residuals);
        double withinVar = groups.Values
            .Where(g => g.Count > 1)
            .Sum(g => ComputeVariance([.. g]) * g.Count / residuals.Length);

        return totalVar - withinVar;
    }

    /// <summary>
    /// Computes variance.
    /// </summary>
    private static double ComputeVariance(double[] values)
    {
        if (values.Length < 2) return 0;
        double mean = values.Average();
        return values.Sum(v => (v - mean) * (v - mean)) / values.Length;
    }

    /// <summary>
    /// Centers shape functions by subtracting their weighted mean.
    /// </summary>
    private void CenterShapeFunctions(int[][] sampleBins, int n)
    {
        for (int f = 0; f < _numFeatures; f++)
        {
            // Compute weighted mean of shape function
            double weightedSum = 0;
            for (int i = 0; i < n; i++)
            {
                weightedSum += NumOps.ToDouble(_shapeFunctions[f][sampleBins[f][i]]);
            }
            double mean = weightedSum / n;

            // Subtract mean and add to intercept
            for (int b = 0; b < _shapeFunctions[f].Length; b++)
            {
                _shapeFunctions[f][b] = NumOps.Subtract(_shapeFunctions[f][b], NumOps.FromDouble(mean));
            }
            _intercept = NumOps.Add(_intercept, NumOps.FromDouble(mean));
        }
    }

    /// <summary>
    /// Calculates feature importances based on shape function variability.
    /// </summary>
    private void CalculateFeatureImportances()
    {
        var importances = new T[_numFeatures];

        for (int f = 0; f < _numFeatures; f++)
        {
            // Importance = range of shape function values
            double min = double.MaxValue;
            double max = double.MinValue;
            foreach (var val in _shapeFunctions[f])
            {
                double v = NumOps.ToDouble(val);
                min = Math.Min(min, v);
                max = Math.Max(max, v);
            }
            importances[f] = NumOps.FromDouble(max - min);
        }

        // Normalize
        T sum = importances.Aggregate(NumOps.Zero, NumOps.Add);
        if (NumOps.ToDouble(sum) > 0)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                importances[f] = NumOps.Divide(importances[f], sum);
            }
        }

        FeatureImportances = new Vector<T>(importances);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.ExplainableBoostingClassifier,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumFeatures", _numFeatures },
                { "MaxBins", _options.MaxBins },
                { "NumInteractions", _interactionTerms.Count },
                { "OuterBags", _options.OuterBags },
                { "InnerBags", _options.InnerBags }
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

        writer.Write(_numFeatures);
        writer.Write(Convert.ToDouble(_intercept));

        // Shape functions
        for (int f = 0; f < _numFeatures; f++)
        {
            writer.Write(_shapeFunctions[f].Length);
            foreach (var val in _shapeFunctions[f])
            {
                writer.Write(Convert.ToDouble(val));
            }
        }

        // Bin edges
        for (int f = 0; f < _numFeatures; f++)
        {
            writer.Write(_binEdges[f].Length);
            foreach (var edge in _binEdges[f])
            {
                writer.Write(Convert.ToDouble(edge));
            }
        }

        // Interaction terms
        writer.Write(_interactionTerms.Count);
        foreach (var kvp in _interactionTerms)
        {
            writer.Write(kvp.Key.Item1);
            writer.Write(kvp.Key.Item2);
            writer.Write(kvp.Value.GetLength(0));
            writer.Write(kvp.Value.GetLength(1));
            for (int i = 0; i < kvp.Value.GetLength(0); i++)
            {
                for (int j = 0; j < kvp.Value.GetLength(1); j++)
                {
                    writer.Write(Convert.ToDouble(kvp.Value[i, j]));
                }
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

        _numFeatures = reader.ReadInt32();
        _intercept = NumOps.FromDouble(reader.ReadDouble());

        // Shape functions
        _shapeFunctions = new T[_numFeatures][];
        for (int f = 0; f < _numFeatures; f++)
        {
            int len = reader.ReadInt32();
            _shapeFunctions[f] = new T[len];
            for (int b = 0; b < len; b++)
            {
                _shapeFunctions[f][b] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Bin edges
        _binEdges = new T[_numFeatures][];
        for (int f = 0; f < _numFeatures; f++)
        {
            int len = reader.ReadInt32();
            _binEdges[f] = new T[len];
            for (int e = 0; e < len; e++)
            {
                _binEdges[f][e] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Interaction terms
        int numInteractions = reader.ReadInt32();
        _interactionTerms = [];
        for (int k = 0; k < numInteractions; k++)
        {
            int f1 = reader.ReadInt32();
            int f2 = reader.ReadInt32();
            int dim1 = reader.ReadInt32();
            int dim2 = reader.ReadInt32();
            var term = new T[dim1, dim2];
            for (int i = 0; i < dim1; i++)
            {
                for (int j = 0; j < dim2; j++)
                {
                    term[i, j] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            _interactionTerms[(f1, f2)] = term;
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new ExplainableBoostingClassifier<T>(_options, Regularization);
    }
}
