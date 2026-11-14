using AiDotNet.Attributes;
using AiDotNet.Classification.Ensemble;
using AiDotNet.Enums;
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
/// <para><b>Recommended:</b> Use <c>AiModelBuilder</c> for the simplest entry point.</para>
/// <example>
/// <code>
/// // Create EBM classifier for interpretable predictions
/// var options = new ExplainableBoostingClassifierOptions&lt;double&gt;();
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
/// // Train additive model with per-feature shape functions
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new ExplainableBoostingClassifier&lt;double&gt;(options))
///     .Build(features, labels);
///
/// // Predict with full interpretability (shape function contributions visible)
/// var newSample = new Matrix&lt;double&gt;(1, 2);
/// newSample[0, 0] = 1.1; newSample[0, 1] = 1.0;
/// var prediction = result.Predict(newSample);
/// // Result is available in the returned value
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.Interpretable)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("InterpretML: A Unified Framework for Machine Learning Interpretability", "https://arxiv.org/abs/1909.09223", Year = 2019, Authors = "Harsha Nori, Samuel Jenkins, Paul Koch, Rich Caruana")]
public partial class ExplainableBoostingClassifier<T> : EnsembleClassifierBase<T>
{
    /// <summary>
    /// Shape functions for each feature (additive terms).
    /// Indexed as: _shapeFunction[featureIndex][binIndex]
    /// </summary>
    private List<Vector<T>> _shapeFunctions;

    /// <summary>
    /// Bin edges for each feature.
    /// Indexed as: _binEdges[featureIndex][edgeIndex]
    /// </summary>
    [Buffer]
    private List<Vector<T>> _binEdges;

    /// <summary>
    /// Interaction terms: pairs of features and their joint effect.
    /// </summary>
    private Dictionary<(int, int), Matrix<T>> _interactionTerms;

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
    public IReadOnlyList<Vector<T>> ShapeFunctions => _shapeFunctions;

    /// <summary>
    /// Gets the bin edges for each feature.
    /// </summary>
    public IReadOnlyList<Vector<T>> BinEdges => _binEdges;

    /// <summary>
    /// Gets the interaction terms.
    /// </summary>
    public IReadOnlyDictionary<(int, int), Matrix<T>> InteractionTerms => _interactionTerms;

    /// <summary>
    /// Initializes a new instance of ExplainableBoostingClassifier.
    /// </summary>
    /// <param name="options">Configuration options.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public ExplainableBoostingClassifier(
        ExplainableBoostingClassifierOptions<T>? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options ??= new ExplainableBoostingClassifierOptions<T>(), regularization, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options;
        _shapeFunctions = [];
        _binEdges = [];
        _interactionTerms = new Dictionary<(int, int), Matrix<T>>();
        _intercept = NumOps.Zero;
        _numFeatures = 0;
        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Returns the model type identifier.
    /// </summary>

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

        if (NumClasses != 2)
        {
            throw new NotSupportedException(
                $"EBM classifier currently supports binary classification only (found {NumClasses} classes).");
        }

        TaskType = InferTaskType(y);

        _random = _options.Seed.HasValue
            ? RandomHelper.CreateSeededRandom(_options.Seed.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;

        // Convert to binary labels
        T positiveClass = ClassLabels[ClassLabels.Length - 1];
        var yBinary = engine.TensorEquals(y.AsTensor(), positiveClass).ToVector();
        int posCount = engine.TensorCountNonzero(yBinary.AsTensor());

        // Initialize intercept as log-odds of prior
        double p = Math.Max(1e-10, Math.Min(1 - 1e-10, (double)posCount / n));
        _intercept = NumOps.FromDouble(Math.Log(p / (1 - p)));

        // Create bins for each feature
        CreateBins(x);

        // Initialize shape functions to zero
        _shapeFunctions = new List<Vector<T>>(_numFeatures);
        for (int f = 0; f < _numFeatures; f++)
        {
            _shapeFunctions.Add(new Vector<T>(_binEdges[f].Length + 1));
        }

        // Initialize interaction terms
        _interactionTerms = new Dictionary<(int, int), Matrix<T>>();

        // Map samples to bins for each feature
        var sampleBins = new Tensor<int>[_numFeatures];
        for (int f = 0; f < _numFeatures; f++)
        {
            sampleBins[f] = GetBinIndices(x.GetColumn(f), f);
        }

        // Cyclic boosting over features
        T bestLoss = NumOps.MaxValue;
        int roundsWithoutImprovement = 0;

        // Maintain running log-odds to avoid recomputing from scratch each iteration
        var runningLogOdds = ComputeLogOdds(sampleBins, n);

        T learningRate = NumOps.FromDouble(_options.LearningRate);
        T l2Reg = NumOps.FromDouble(_options.L2Regularization);
        T eps = Tolerance(1e-10);

        for (int outer = 0; outer < _options.OuterBags; outer++)
        {
            for (int inner = 0; inner < _options.InnerBags; inner++)
            {
                // Cycle through features in round-robin fashion
                for (int f = 0; f < _numFeatures; f++)
                {
                    int numBins = _binEdges[f].Length + 1;

                    // Newton step per bin (zero where the bin's hessian is not above eps)
                    var binUpdates = ComputeBinUpdates(runningLogOdds, yBinary, sampleBins[f], numBins,
                        learningRate, l2Reg, eps);

                    // Update shape function for this feature and adjust running log-odds
                    _shapeFunctions[f] = engine.Add(_shapeFunctions[f], binUpdates);
                    runningLogOdds = engine.Add(runningLogOdds,
                        engine.TensorGather(binUpdates.AsTensor(), sampleBins[f], 0).ToVector());
                }
            }

            // Early stopping check
            if (_options.EarlyStoppingRounds.HasValue
                && ShouldStopEarly(ComputeLogLoss(runningLogOdds, yBinary), ref bestLoss, ref roundsWithoutImprovement))
            {
                break;
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
        return PredictBinaryLabels(PredictProbabilities(input));
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

        if (input.Columns < _numFeatures)
        {
            throw new ArgumentException(
                $"Input has {input.Columns} columns but model requires at least {_numFeatures} features.");
        }

        int n = input.Rows;
        if (n == 0)
        {
            return new Matrix<T>(0, NumClasses);
        }

        var engine = Engine;
        var bins = new Tensor<int>[_numFeatures];
        for (int f = 0; f < _numFeatures; f++)
        {
            bins[f] = GetBinIndices(input.GetColumn(f), f);
        }

        // Main effects
        var logOdds = ComputeLogOdds(bins, n);

        // Interaction effects (bins clamped to the term's grid)
        foreach (var kvp in _interactionTerms)
        {
            var cells = InteractionCells(bins[kvp.Key.Item1], bins[kvp.Key.Item2], kvp.Value.Rows, kvp.Value.Columns);
            logOdds = engine.Add(logOdds, engine.TensorGather(FlattenMatrix(kvp.Value), cells, 0).ToVector());
        }

        return BinaryProbabilitiesFromLogOdds(logOdds);
    }

    /// <summary>
    /// Explains a single prediction by showing each feature's contribution.
    /// </summary>
    /// <param name="sample">A single input sample.</param>
    /// <returns>Dictionary mapping feature index to its contribution.</returns>
    public Dictionary<string, T> ExplainPrediction(Vector<T> sample)
    {
        var contributions = new Dictionary<string, T>
        {
            // Include intercept so contributions sum to the full log-odds
            ["intercept"] = _intercept
        };

        var engine = Engine;
        var bins = new Tensor<int>[_numFeatures];
        for (int f = 0; f < _numFeatures; f++)
        {
            bins[f] = GetBinIndices(sample.GetSubVector(f, 1), f);
        }

        // Main effects
        for (int f = 0; f < _numFeatures; f++)
        {
            contributions[$"feature_{f}"] = engine.TensorGather(_shapeFunctions[f].AsTensor(), bins[f], 0).ToArray()[0];
        }

        // Interaction effects
        foreach (var kvp in _interactionTerms)
        {
            int f1 = kvp.Key.Item1;
            int f2 = kvp.Key.Item2;
            var cell = InteractionCells(bins[f1], bins[f2], kvp.Value.Rows, kvp.Value.Columns);
            contributions[$"interaction_{f1}x{f2}"] = engine.TensorGather(FlattenMatrix(kvp.Value), cell, 0).ToArray()[0];
        }

        return contributions;
    }

    /// <summary>
    /// Creates bins for each feature using quantiles.
    /// </summary>
    private void CreateBins(Matrix<T> x)
    {
        _binEdges = new List<Vector<T>>(_numFeatures);
        var engine = Engine;

        for (int f = 0; f < _numFeatures; f++)
        {
            // Sorted distinct values of the feature
            var distinctValues = x.Rows == 0
                ? new Tensor<T>(new[] { 0 })
                : engine.TensorUnique(x.GetColumn(f).AsTensor(), sorted: true);
            int distinctCount = distinctValues.Length;

            // Create quantile-based bins
            int numBins = Math.Min(_options.MaxBins, distinctCount);
            if (numBins <= 1)
            {
                _binEdges.Add(new Vector<T>(0));
                continue;
            }

            // Edge b sits at the b/numBins quantile position of the distinct values
            var edgePositions = new int[numBins - 1];
            for (int b = 1; b < numBins; b++)
            {
                int idx = (int)((double)b / numBins * distinctCount);
                edgePositions[b - 1] = Math.Max(0, Math.Min(idx, distinctCount - 1));
            }

            var edges = engine.TensorGather(distinctValues, new Tensor<int>(edgePositions, new[] { edgePositions.Length }), 0);

            // Deduplicate edges (they are sorted, so duplicates are consecutive)
            _binEdges.Add(engine.TensorUniqueConsecutive(edges).ToVector());
        }
    }

    /// <summary>
    /// Gets the bin index of every value: the number of bin edges that are &lt;= the value.
    /// </summary>
    private Tensor<int> GetBinIndices(Vector<T> values, int featureIndex)
    {
        var edges = _binEdges[featureIndex];
        if (edges.Length == 0 || values.Length == 0)
        {
            return new Tensor<int>(new[] { values.Length });
        }

        return Engine.TensorSearchSorted(edges.AsTensor(), values.AsTensor(), right: true);
    }

    /// <summary>
    /// Computes log-odds for all samples: intercept plus every feature's shape value at its bin.
    /// </summary>
    private Vector<T> ComputeLogOdds(Tensor<int>[] sampleBins, int n)
    {
        var engine = Engine;
        var logOdds = FilledVector(n, _intercept);
        if (n == 0)
        {
            return logOdds;
        }

        for (int f = 0; f < _numFeatures; f++)
        {
            logOdds = engine.Add(logOdds, engine.TensorGather(_shapeFunctions[f].AsTensor(), sampleBins[f], 0).ToVector());
        }

        return logOdds;
    }

    /// <summary>
    /// Per-bin Newton step lr * G_b / (H_b + l2) from the gradients y - p and hessians p(1 - p) summed per bin,
    /// zero for bins whose hessian is not above <paramref name="eps"/>.
    /// </summary>
    private Vector<T> ComputeBinUpdates(Vector<T> logOdds, Vector<T> yBinary, Tensor<int> bins, int numBins,
        T learningRate, T l2Reg, T eps)
    {
        var engine = Engine;
        var zeros = new Tensor<T>(new[] { numBins });
        if (logOdds.Length == 0)
        {
            return zeros.ToVector();
        }

        var prob = engine.Sigmoid(logOdds).AsTensor();
        var grad = engine.TensorSubtract(yBinary.AsTensor(), prob);
        var hess = engine.TensorMultiply(prob, engine.ScalarMinusTensor(NumOps.One, prob));

        var binGradients = engine.TensorScatterAdd(zeros, bins, grad, 0);
        var binHessians = engine.TensorScatterAdd(zeros, bins, hess, 0);

        // Bins without hessian mass are skipped, as in the scalar loop, and must not be divided either: with
        // L2 = 0 an empty bin (normal in an interaction grid) has a zero divisor, which throws for decimal
        var active = engine.TensorGreaterThan(binHessians, eps);
        var ones = new Tensor<T>(zeros.Shape.ToArray());
        engine.TensorFill(ones, NumOps.One);
        var step = engine.TensorMultiplyScalar(
            engine.TensorDivide(binGradients, engine.TensorWhere(active, engine.TensorAddScalar(binHessians, l2Reg), ones)),
            learningRate);
        return engine.TensorWhere(active, step, zeros).ToVector();
    }

    /// <summary>
    /// Flat cell index min(b1, rows - 1) * columns + min(b2, columns - 1) of each sample in an interaction grid.
    /// </summary>
    private Tensor<int> InteractionCells(Tensor<int> bins1, Tensor<int> bins2, int rows, int columns)
    {
        var engine = Engine;
        var b1 = engine.TensorClampMax(bins1, rows - 1);
        var b2 = engine.TensorClampMax(bins2, columns - 1);
        return engine.TensorAdd(engine.TensorMultiplyScalar(b1, columns), b2);
    }

    /// <summary>
    /// Row-major rank-1 view of a matrix's values.
    /// </summary>
    private static Tensor<T> FlattenMatrix(Matrix<T> matrix)
    {
        return Tensor<T>.FromMatrix(matrix).Reshape(matrix.Rows * matrix.Columns);
    }

    /// <summary>
    /// Early-stopping bookkeeping: records an improvement or counts a stale round.
    /// </summary>
    /// <returns>True when training should stop.</returns>
    private bool ShouldStopEarly(T loss, ref T bestLoss, ref int roundsWithoutImprovement)
    {
        if (NumOps.LessThan(loss, bestLoss))
        {
            bestLoss = loss;
            roundsWithoutImprovement = 0;
            return false;
        }

        roundsWithoutImprovement++;
        return roundsWithoutImprovement >= (_options.EarlyStoppingRounds ?? int.MaxValue);
    }

    /// <summary>
    /// Computes log loss.
    /// </summary>
    private T ComputeLogLoss(Vector<T> logOdds, Vector<T> yBinary) => ComputeBinaryLogLoss(logOdds, yBinary);

    /// <summary>
    /// Detects important pairwise interactions and trains them with cyclic boosting.
    /// </summary>
    private void DetectInteractions(Matrix<T> x, Tensor<int>[] sampleBins, Vector<T> yBinary)
    {
        int n = x.Rows;
        var engine = Engine;
        var logOdds = ComputeLogOdds(sampleBins, n);
        var residuals = engine.Subtract(yBinary, engine.Sigmoid(logOdds));

        // Check top feature pairs
        var candidates = new List<(int, int)>();
        var candidateScores = new List<T>();

        int maxFeaturesToConsider = Math.Min(_numFeatures, Math.Max(0, _options.MaxInteractionFeatures));
        if (maxFeaturesToConsider < 2)
        {
            return;
        }

        for (int f1 = 0; f1 < maxFeaturesToConsider; f1++)
        {
            for (int f2 = f1 + 1; f2 < maxFeaturesToConsider; f2++)
            {
                candidateScores.Add(ComputeInteractionScore(sampleBins[f1], sampleBins[f2], _binEdges[f2].Length + 1, residuals));
                candidates.Add((f1, f2));
            }
        }

        // Select top interactions (stable descending by score) and initialize their terms
        var topIndices = AiDotNet.ActiveLearning.ActiveLearningEngineOps.SelectTopScoring(
            new Vector<T>(candidateScores), _options.MaxInteractions);

        foreach (int top in topIndices)
        {
            var (f1, f2) = candidates[top];
            int numBins1 = Math.Min(_binEdges[f1].Length + 1, _options.MaxInteractionBins);
            int numBins2 = Math.Min(_binEdges[f2].Length + 1, _options.MaxInteractionBins);
            _interactionTerms[(f1, f2)] = new Matrix<T>(numBins1, numBins2);
        }

        if (_interactionTerms.Count == 0) return;

        // Recompute running log-odds including main effects
        var runningLogOdds = logOdds.Clone();

        T learningRate = NumOps.FromDouble(_options.LearningRate);
        T l2Reg = NumOps.FromDouble(_options.L2Regularization);
        T eps = Tolerance(1e-10);

        // Cyclic boosting over interaction terms (same structure as main effect training)
        var keys = _interactionTerms.Keys.ToList();
        for (int outer = 0; outer < _options.InnerBags; outer++)
        {
            foreach (var key in keys)
            {
                var term = _interactionTerms[key];
                int numBins1 = term.Rows;
                int numBins2 = term.Columns;
                var cells = InteractionCells(sampleBins[key.Item1], sampleBins[key.Item2], numBins1, numBins2);

                var cellUpdates = ComputeBinUpdates(runningLogOdds, yBinary, cells, numBins1 * numBins2,
                    learningRate, l2Reg, eps);

                var updated = engine.TensorAdd(FlattenMatrix(term), cellUpdates.AsTensor());
                _interactionTerms[key] = updated.Reshape(numBins1, numBins2).ToMatrix();

                // Update running log-odds for this interaction's changes
                runningLogOdds = engine.Add(runningLogOdds,
                    engine.TensorGather(cellUpdates.AsTensor(), cells, 0).ToVector());
            }
        }
    }

    /// <summary>
    /// Computes interaction score (variance reduction): total variance of the residuals minus the
    /// sample-weighted variance inside each (bin1, bin2) group.
    /// </summary>
    private T ComputeInteractionScore(Tensor<int> bins1, Tensor<int> bins2, int numBins2, Vector<T> residuals)
    {
        int n = residuals.Length;
        T totalVar = ComputeVariance(residuals);
        if (n < 2)
        {
            return totalVar;
        }

        // Group id of each sample; the bins are in [0, edges + 1), so this is a unique id per pair.
        var engine = Engine;
        var groupIds = engine.TensorAdd(engine.TensorMultiplyScalar(bins1, numBins2), bins2);
        int numGroups = engine.TensorMaxValue(groupIds) + 1;

        // sum_g (|g| / n) * Var(g) = sum_i (r_i - mean_{g(i)})^2 / n; singleton groups contribute zero.
        var r = residuals.AsTensor();
        var zeros = new Tensor<T>(new[] { numGroups });
        var groupSums = engine.TensorScatterAdd(zeros, groupIds, r, 0);
        var groupCounts = CountsToValues(engine.TensorClampMin(engine.TensorBinCount(groupIds, numGroups).Slice(0, 0, numGroups), 1));
        var groupMeans = engine.TensorDivide(groupSums, groupCounts);
        var deviation = engine.TensorSubtract(r, engine.TensorGather(groupMeans, groupIds, 0));
        T withinVar = NumOps.Divide(engine.TensorSumOfSquares(deviation), NumOps.FromDouble(n));

        return NumOps.Subtract(totalVar, withinVar);
    }

    /// <summary>
    /// Computes the population variance.
    /// </summary>
    private T ComputeVariance(Vector<T> values)
    {
        if (values.Length < 2) return NumOps.Zero;
        var engine = Engine;
        T mean = engine.Mean(values);
        var diff = engine.TensorSubtractScalar(values.AsTensor(), mean);
        return NumOps.Divide(engine.TensorSumOfSquares(diff), NumOps.FromDouble(values.Length));
    }

    /// <summary>
    /// Centers shape functions by subtracting their weighted mean.
    /// </summary>
    private void CenterShapeFunctions(Tensor<int>[] sampleBins, int n)
    {
        var engine = Engine;
        var means = new List<T>(_numFeatures);
        for (int f = 0; f < _numFeatures; f++)
        {
            // Weighted mean of the shape function = mean of its value at each sample's bin
            var sf = _shapeFunctions[f];
            T mean = n == 0 ? NumOps.Zero : engine.TensorMean(engine.TensorGather(sf.AsTensor(), sampleBins[f], 0));

            // Subtract mean (the means are added to the intercept below)
            _shapeFunctions[f] = engine.TensorSubtractScalar(sf.AsTensor(), mean).ToVector();
            means.Add(mean);
        }

        if (means.Count > 0)
        {
            _intercept = NumOps.Add(_intercept, engine.Sum(Vector<T>.FromList(means)));
        }
    }

    /// <summary>
    /// Calculates feature importances based on shape function variability.
    /// </summary>
    private void CalculateFeatureImportances()
    {
        var engine = Engine;

        // Importance = range of shape function values
        var maxima = new List<T>(_numFeatures);
        var minima = new List<T>(_numFeatures);
        foreach (var sf in _shapeFunctions)
        {
            maxima.Add(engine.TensorMaxValue(sf.AsTensor()));
            minima.Add(engine.TensorMinValue(sf.AsTensor()));
        }

        var importances = engine.Subtract(Vector<T>.FromList(maxima), Vector<T>.FromList(minima));

        // Normalize
        T sum = _numFeatures == 0 ? NumOps.Zero : engine.Sum(importances);
        if (NumOps.GreaterThan(sum, NumOps.Zero))
        {
            importances = engine.Divide(importances, sum);
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
                { "NumFeatures", _numFeatures },
                { "MaxBins", _options.MaxBins },
                { "NumInteractions", _interactionTerms.Count },
                { "OuterBags", _options.OuterBags },
                { "InnerBags", _options.InnerBags }
            }
        };
    }
}
