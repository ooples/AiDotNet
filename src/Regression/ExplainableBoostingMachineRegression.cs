using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.Regression;

/// <summary>
/// Explainable Boosting Machine (EBM) for interpretable regression.
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
///    an intercept
/// 3. You can plot these shape functions to understand exactly how the model
///    uses each feature
///
/// For example, in predicting house prices:
/// - The shape function for "square footage" might show a linear increase
/// - The shape function for "age" might show older houses have lower prices
/// - The prediction for a specific house is just: intercept + f(sqft) + f(age) + ...
///
/// This additive structure makes EBM uniquely interpretable while still being accurate.
/// </para>
/// <para>
/// Reference: Lou, Y., et al. "Intelligible Models for Healthcare: Predicting
/// Pneumonia Risk and Hospital 30-day Readmission" (2012).
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Create an Explainable Boosting Machine for interpretable regression
/// var options = new ExplainableBoostingMachineOptions();
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = new Matrix&lt;double&gt;(new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 }, { 11, 12 } });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train with per-feature shape functions for interpretability
/// var result = new AiModelBuilder&lt;double, Matrix&lt;double&gt;, Vector&lt;double&gt;&gt;()
///     .ConfigureModel(new ExplainableBoostingMachineRegression&lt;double&gt;(options))
///     .Build(features, targets);
///
/// // Predict for a new sample (sum of per-feature contributions)
/// var newSample = new Matrix&lt;double&gt;(new double[,] { { 13, 14 } });
/// var prediction = result.Predict(newSample);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.Ensemble)]
[ModelCategory(ModelCategory.Interpretable)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Matrix<>), typeof(Vector<>))]
[ResearchPaper("Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission", "https://doi.org/10.1145/2783258.2788613", Year = 2015, Authors = "Rich Caruana, Yin Lou, Johannes Gehrke, Paul Koch, Marc Sturm, Noemie Elhadad")]
public partial class ExplainableBoostingMachineRegression<T> : AsyncDecisionTreeRegressionBase<T>
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
    private List<Vector<T>> _binEdges;

    /// <summary>
    /// Interaction terms: pairs of features and their joint effect.
    /// Indexed as: _interactionTerms[(feat1, feat2)][(bin1, bin2)]
    /// </summary>
    private Dictionary<(int, int), Matrix<T>> _interactionTerms;

    /// <summary>
    /// The intercept (baseline prediction).
    /// </summary>
    private T _intercept;

    /// <summary>
    /// Number of features.
    /// </summary>
    private int _numFeatures;

    /// <summary>
    /// Configuration options.
    /// </summary>
    private readonly ExplainableBoostingMachineOptions _options;

    /// <summary>
    /// Gets the intercept (baseline prediction).
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

    /// <inheritdoc/>
    public override int NumberOfTrees => 1;  // EBM is not truly tree-based, but uses boosting

    /// <summary>
    /// Initializes a new instance of ExplainableBoostingMachineRegression.
    /// </summary>
    /// <param name="options">Configuration options for the algorithm.</param>
    /// <param name="regularization">Optional regularization strategy.</param>
    public ExplainableBoostingMachineRegression(
        ExplainableBoostingMachineOptions? options = null,
        IRegularization<T, Matrix<T>, Vector<T>>? regularization = null)
        : base(options, regularization)
    {
        _options = options ?? new ExplainableBoostingMachineOptions();
        _shapeFunctions = [];
        _binEdges = [];
        _interactionTerms = new Dictionary<(int, int), Matrix<T>>();
        _intercept = NumOps.Zero;
        _numFeatures = 0;
    }

    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        _intercept = Engine.Mean(y);

        CreateBins(x);

        // Precompute every sample's bin per feature (one searchsorted per feature)
        var binIndices = new int[_numFeatures][];
        for (int f = 0; f < _numFeatures; f++)
            binIndices[f] = BinIndices(f, Engine.GetColumn(x, f));

        _shapeFunctions = new List<Vector<T>>(_numFeatures);
        for (int f = 0; f < _numFeatures; f++)
            _shapeFunctions.Add(Vector<T>.CreateDefault(_binEdges[f].Length + 1, NumOps.Zero));

        var residuals = Engine.Subtract(y, Engine.Fill(n, _intercept));

        int[] featureOrder = Enumerable.Range(0, _numFeatures).ToArray();
        T lr = NumOps.FromDouble(_options.LearningRate);

        for (int outer = 0; outer < _options.NumberOfOuterIterations; outer++)
        {
            if (!_options.CyclicTraining)
                ShuffleArray(featureOrder);

            int[] sampleIndices = GetSampleIndices(n);

            foreach (int f in featureOrder)
            {
                for (int inner = 0; inner < _options.NumberOfInnerIterations; inner++)
                    residuals = UpdateShapeFunction(residuals, f, sampleIndices, binIndices[f], lr);
            }
        }

        if (_options.DetectInteractions)
            DetectAndAddInteractions(residuals, binIndices);

        if (_options.RegularizationStrength > 0)
            SmoothShapeFunctions();

        await CalculateFeatureImportancesAsync(x.Columns);
    }

    /// <inheritdoc/>
    public override async Task<Vector<T>> PredictAsync(Matrix<T> input)
    {
        return await Task.Run(() => PredictBatch(input));
    }

    /// <summary>
    /// Intercept plus every main-effect and interaction contribution, gathered for all rows at once.
    /// Contributions are added in the same order as a per-row sum: features first, then interactions.
    /// </summary>
    private Vector<T> PredictBatch(Matrix<T> input)
    {
        int rows = input.Rows;
        var prediction = Engine.Fill(rows, _intercept);
        if (rows == 0)
        {
            return prediction;
        }

        var binIndices = new int[_numFeatures][];
        for (int f = 0; f < _numFeatures; f++)
        {
            binIndices[f] = BinIndices(f, Engine.GetColumn(input, f));
            prediction = Engine.Add(prediction, RegressionEngineOps.Gather(_shapeFunctions[f], binIndices[f]));
        }

        foreach (var ((f1, f2), interactionMatrix) in _interactionTerms)
        {
            // Clamp to interaction matrix bounds
            int[] cells = InteractionCells(binIndices[f1], binIndices[f2],
                interactionMatrix.Rows, interactionMatrix.Columns, clamp: true);
            prediction = Engine.Add(prediction,
                RegressionEngineOps.Gather(RegressionEngineOps.FlattenRowMajor(interactionMatrix), cells));
        }

        return prediction;
    }

    /// <summary>
    /// Flat row-major cell index of each (bin1, bin2) pair in a [rows1, rows2] grid. With
    /// <paramref name="clamp"/> the bins are capped at the last row/column; otherwise they wrap (modulo).
    /// </summary>
    private static int[] InteractionCells(int[] bins1, int[] bins2, int rows1, int rows2, bool clamp)
    {
        var cells = new int[bins1.Length];
        for (int i = 0; i < cells.Length; i++)
        {
            int b1 = clamp ? Math.Min(bins1[i], rows1 - 1) : bins1[i] % rows1;
            int b2 = clamp ? Math.Min(bins2[i], rows2 - 1) : bins2[i] % rows2;
            cells[i] = b1 * rows2 + b2;
        }

        return cells;
    }

    /// <summary>
    /// Predicts the contribution of each feature for a single sample.
    /// </summary>
    /// <param name="sample">Feature values for a single sample.</param>
    /// <returns>Dictionary mapping feature index to its contribution.</returns>
    /// <remarks>
    /// <para>
    /// This method is useful for explaining individual predictions. It shows
    /// exactly how much each feature contributed to the final prediction.
    /// </para>
    /// <para><b>For Beginners:</b> This tells you why the model made a specific prediction.
    /// For example, for a house price prediction of $300,000, it might show:
    /// - Base price: $200,000
    /// - Square footage contribution: +$80,000
    /// - Age contribution: -$20,000
    /// - Neighborhood contribution: +$40,000
    /// Total: $300,000
    /// </para>
    /// </remarks>
    public Dictionary<int, T> GetFeatureContributions(Vector<T> sample)
    {
        if (_numFeatures == 0)
        {
            return new Dictionary<int, T>();
        }

        var contributions = Vector<T>.Concatenate(Enumerable.Range(0, _numFeatures)
            .Select(f => RegressionEngineOps.Gather(_shapeFunctions[f], BinIndices(f, sample.GetSubVector(f, 1))))
            .ToList());

        return contributions.Select((value, f) => (f, value)).ToDictionary(entry => entry.f, entry => entry.value);
    }

    /// <summary>
    /// Gets the learned shape function for a specific feature.
    /// </summary>
    /// <param name="featureIndex">Index of the feature.</param>
    /// <returns>Tuple of (bin centers, shape function values).</returns>
    /// <remarks>
    /// This can be used to visualize how the model uses each feature.
    /// Plot the bin centers on the x-axis and shape values on the y-axis.
    /// </remarks>
    public (Vector<T> BinCenters, Vector<T> ShapeValues) GetShapeFunctionForPlot(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= _numFeatures)
            throw new ArgumentOutOfRangeException(nameof(featureIndex));

        var shape = _shapeFunctions[featureIndex];
        int numBins = shape.Length;
        var edges = _binEdges[featureIndex];
        var values = new Vector<T>(shape);

        // Bin centers: the first bin uses the first edge, the last bin the last edge, middle bins
        // the midpoint of their two edges, and any bin past the edges 0.
        var parts = new List<Vector<T>>();
        if (numBins == 1)
        {
            parts.Add(edges.Length > 0 ? edges.GetSubVector(0, 1) : edges.GetSubVector(edges.Length - 1, 1));
        }
        else if (numBins > 1)
        {
            parts.Add(edges.Length > 0 ? edges.GetSubVector(0, 1) : new Vector<T>(1));
            int middle = numBins - 2;
            int withEdges = Math.Max(0, Math.Min(middle, edges.Length - 1));
            if (withEdges > 0)
            {
                parts.Add(RegressionEngineOps.AdjacentMidpoints(edges.GetSubVector(0, withEdges + 1)));
            }

            if (middle > withEdges)
            {
                parts.Add(new Vector<T>(middle - withEdges));
            }

            parts.Add(edges.GetSubVector(edges.Length - 1, 1));
        }

        var centers = parts.Count == 0 ? new Vector<T>(0) : Vector<T>.Concatenate(parts);
        return (centers, values);
    }

    /// <summary>
    /// Creates bins for each feature using quantile binning.
    /// </summary>
    private void CreateBins(Matrix<T> x)
    {
        _binEdges = new List<Vector<T>>(_numFeatures);

        for (int f = 0; f < _numFeatures; f++)
        {
            // Get unique values for this feature
            var values = new List<double>();
            for (int i = 0; i < x.Rows; i++)
            {
                values.Add(NumOps.ToDouble(x[i, f]));
            }
            values.Sort();

            // Determine number of bins: limit by unique values AND data size
            // to ensure each bin has enough samples for reliable gradient estimates
            int numUnique = values.Count > 0 ? 1 : 0;
            for (int i = 1; i < values.Count; i++)
            {
                if (values[i] != values[i - 1]) numUnique++;
            }
            int maxBinsForData = Math.Max(2, values.Count / Math.Max(1, _options.MinSamplesPerBin));
            int numBins = Math.Min(_options.MaxBins, Math.Min(numUnique, maxBinsForData));

            if (numBins <= 1)
            {
                _binEdges.Add(new Vector<T>(0));
                continue;
            }

            // Create quantile-based bin edges
            var edges = new List<T>();
            for (int b = 1; b < numBins; b++)
            {
                int idx = (int)((double)b / numBins * values.Count);
                idx = Math.Min(idx, values.Count - 1);
                double edge = values[idx];

                // Avoid duplicate edges
                if (edges.Count == 0 || NumOps.ToDouble(edges[^1]) < edge - 1e-10)
                {
                    edges.Add(NumOps.FromDouble(edge));
                }
            }

            _binEdges.Add(new Vector<T>(edges));
        }
    }

    /// <summary>
    /// Bin of each value for the feature: the number of edges at or below it (NaN goes past the last edge).
    /// </summary>
    private int[] BinIndices(int featureIndex, Vector<T> values)
    {
        var edges = _binEdges[featureIndex];
        if (edges.Length == 0 || values.Length == 0)
        {
            return new int[values.Length];
        }

        return Engine.TensorSearchSorted(Tensor<T>.FromVector(edges), Tensor<T>.FromVector(values), right: true).ToArray();
    }

    private Vector<T> UpdateShapeFunction(Vector<T> residuals, int featureIndex,
        int[] sampleIndices, int[] featureBins, T lr)
    {
        int numBins = _shapeFunctions[featureIndex].Length;

        // Per-bin residual sums and counts over the sampled rows (duplicates count each time)
        var sampleBins = new int[sampleIndices.Length];
        for (int k = 0; k < sampleIndices.Length; k++)
            sampleBins[k] = featureBins[sampleIndices[k]];

        var (binSums, binCounts) = BinSumsAndCounts(
            RegressionEngineOps.Gather(residuals, sampleIndices), sampleBins, numBins);

        // Bins with enough samples move by lr * mean residual; the others stay put
        var enough = Engine.TensorGreaterEqual(binCounts, Tensor<T>.FromVector(
            Engine.Fill(numBins, NumOps.FromDouble(_options.MinSamplesPerBin))));
        var binUpdates = Engine.TensorWhere(
            enough,
            Engine.TensorMultiplyScalar(Engine.TensorDivide(binSums, binCounts), lr),
            Tensor<T>.FromVector(Engine.FillZero<T>(numBins))).ToVector();
        _shapeFunctions[featureIndex] = Engine.Add(_shapeFunctions[featureIndex], binUpdates);

        return Engine.Subtract(residuals, RegressionEngineOps.Gather(binUpdates, featureBins));
    }

    /// <summary>
    /// Sums <paramref name="values"/> and counts entries per bin, as [numBins] tensors.
    /// </summary>
    private (Tensor<T> Sums, Tensor<T> Counts) BinSumsAndCounts(Vector<T> values, int[] bins, int numBins)
    {
        var zeros = Tensor<T>.FromVector(Engine.FillZero<T>(numBins));
        if (bins.Length == 0)
        {
            return (zeros, zeros);
        }

        var binTensor = new Tensor<int>(bins, new[] { bins.Length });
        var sums = Engine.TensorScatterAdd(zeros, binTensor, Tensor<T>.FromVector(values), 0);
        var counts = Engine.TensorScatterAdd(zeros, binTensor,
            Tensor<T>.FromVector(Engine.Fill(bins.Length, NumOps.One)), 0);
        return (sums, counts);
    }

    /// <summary>
    /// Detects and adds important pairwise interactions.
    /// </summary>
    private void DetectAndAddInteractions(Vector<T> residuals, int[][] binIndices)
    {
        _interactionTerms = new Dictionary<(int, int), Matrix<T>>();

        var interactionScores = new List<((int, int) pair, double score)>();
        for (int f1 = 0; f1 < _numFeatures; f1++)
            for (int f2 = f1 + 1; f2 < _numFeatures; f2++)
                interactionScores.Add(((f1, f2), ComputeInteractionScore(residuals, binIndices[f1], binIndices[f2], f1, f2)));

        interactionScores.Sort((a, b) => b.score.CompareTo(a.score));

        int numInteractions = Math.Min(_options.MaxInteractionBins, interactionScores.Count);
        for (int i = 0; i < numInteractions; i++)
        {
            var (pair, _) = interactionScores[i];
            residuals = FitInteraction(residuals, pair.Item1, pair.Item2, binIndices[pair.Item1], binIndices[pair.Item2]);
        }
    }

    private double ComputeInteractionScore(Vector<T> residuals, int[] bins1, int[] bins2, int f1, int f2)
    {
        int numBins1 = Math.Min(5, _shapeFunctions[f1].Length);
        int numBins2 = Math.Min(5, _shapeFunctions[f2].Length);
        int cellCount = numBins1 * numBins2;

        int[] cells = InteractionCells(bins1, bins2, numBins1, numBins2, clamp: false);
        var (sums, counts) = BinSumsAndCounts(residuals, cells, cellCount);

        // sum over occupied cells of count * mean^2
        var means = Engine.TensorDivide(sums, counts);
        var terms = Engine.TensorMultiply(counts, Engine.TensorSquare(means));
        var occupied = Engine.TensorGreaterThan(counts, NumOps.Zero);
        var total = Engine.TensorSum(Engine.TensorWhere(occupied, terms, Tensor<T>.FromVector(Engine.FillZero<T>(cellCount))));

        // The ranking score is a double; the single total is converted once.
        return NumOps.ToDouble(total);
    }

    private Vector<T> FitInteraction(Vector<T> residuals, int f1, int f2, int[] bins1, int[] bins2)
    {
        int numBins1 = Math.Min(_shapeFunctions[f1].Length, 32);
        int numBins2 = Math.Min(_shapeFunctions[f2].Length, 32);
        int cellCount = numBins1 * numBins2;

        int[] cells = InteractionCells(bins1, bins2, numBins1, numBins2, clamp: false);
        var (sums, counts) = BinSumsAndCounts(residuals, cells, cellCount);

        var enough = Engine.TensorGreaterEqual(counts, Tensor<T>.FromVector(
            Engine.Fill(cellCount, NumOps.FromDouble(_options.MinSamplesPerBin))));
        var cellMeans = Engine.TensorWhere(
            enough, Engine.TensorDivide(sums, counts), Tensor<T>.FromVector(Engine.FillZero<T>(cellCount))).ToVector();

        _interactionTerms[(f1, f2)] = new Tensor<T>(new[] { numBins1, numBins2 }, cellMeans).ToMatrix();

        return Engine.Subtract(residuals, RegressionEngineOps.Gather(cellMeans, cells));
    }

    /// <summary>
    /// Applies smoothing to shape functions for better interpretability.
    /// </summary>
    private void SmoothShapeFunctions()
    {
        T lambda = NumOps.FromDouble(_options.RegularizationStrength);
        T divisor = NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), lambda));
        T inverseDivisor = NumOps.Divide(NumOps.One, divisor);

        for (int f = 0; f < _numFeatures; f++)
        {
            var sf = _shapeFunctions[f];
            int len = sf.Length;
            if (len <= 2) continue;

            // Neighbours with the end values repeated: left = [sf0, sf0..sf(len-2)], right = [sf1..sf(len-1), sf(len-1)]
            var leftNeighbor = Vector<T>.Concatenate(new[] { sf.GetSubVector(0, 1), sf.GetSubVector(0, len - 1) });
            var rightNeighbor = Vector<T>.Concatenate(new[] { sf.GetSubVector(1, len - 1), sf.GetSubVector(len - 1, 1) });

            // (lambda * left + lambda * right + sf) / (1 + 2 lambda)
            var smoothed = Engine.Add(Engine.Multiply(leftNeighbor, lambda), Engine.Multiply(rightNeighbor, lambda));
            smoothed = Engine.Add(smoothed, sf);
            _shapeFunctions[f] = Engine.Multiply(smoothed, inverseDivisor);
        }
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        // Importance of a feature = range (max - min) of its shape function, normalized to sum to 1
        var shapes = _shapeFunctions.Take(_numFeatures).ToList();
        var importances = shapes.Count == 0
            ? new Vector<T>(_numFeatures)
            : Engine.Subtract(new Vector<T>(shapes.Select(sf => sf.Max())), new Vector<T>(shapes.Select(sf => sf.Min())));

        T sum = importances.Sum();
        if (NumOps.GreaterThan(sum, NumOps.Zero))
            importances.MultiplyInPlace(NumOps.Divide(NumOps.One, sum));

        FeatureImportances = importances;
        return Task.CompletedTask;
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
    /// Shuffles an array in place.
    /// </summary>
    private void ShuffleArray(int[] array)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = Random.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumberOfFeatures", _numFeatures },
                { "NumberOfIterations", _options.NumberOfOuterIterations },
                { "LearningRate", _options.LearningRate },
                { "MaxBins", _options.MaxBins },
                { "NumberOfInteractions", _interactionTerms.Count },
                { "DetectInteractions", _options.DetectInteractions }
            }
        };
    }
}
