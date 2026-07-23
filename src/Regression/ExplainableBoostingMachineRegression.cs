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
/// var options = new ExplainableBoostingMachineRegressionOptions&lt;double&gt;();
/// var model = new ExplainableBoostingMachineRegression&lt;double&gt;(options);
///
/// // Prepare training data: 6 samples with 2 features each
/// var features = Matrix&lt;double&gt;.Build.Dense(6, 2, new double[] {
///     1, 2,  3, 4,  5, 6,  7, 8,  9, 10,  11, 12 });
/// var targets = new Vector&lt;double&gt;(new double[] { 3.0, 7.1, 11.0, 15.2, 19.0, 23.1 });
///
/// // Train with per-feature shape functions for interpretability
/// model.Train(features, targets);
///
/// // Predict for a new sample (sum of per-feature contributions)
/// var newSample = Matrix&lt;double&gt;.Build.Dense(1, 2, new double[] { 13, 14 });
/// var prediction = model.Predict(newSample);
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
public class ExplainableBoostingMachineRegression<T> : AsyncDecisionTreeRegressionBase<T>
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

        // Precompute bin indices into Matrix<int> — O(1) lookup in hot loop
        var binIndices = new Matrix<int>(n, _numFeatures);
        for (int f = 0; f < _numFeatures; f++)
            for (int i = 0; i < n; i++)
                binIndices[i, f] = GetBinIndex(x[i, f], f);

        _shapeFunctions = new List<Vector<T>>(_numFeatures);
        for (int f = 0; f < _numFeatures; f++)
            _shapeFunctions.Add(Vector<T>.CreateDefault(_binEdges[f].Length + 1, NumOps.Zero));

        var residuals = new Vector<T>(y);
        residuals.SubtractInPlace(Vector<T>.CreateDefault(n, _intercept));

        // Pre-allocate gather buffer — reused every UpdateShapeFunction call, zero GC
        var gatherUpdates = new Vector<T>(n);

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
                    UpdateShapeFunction(residuals, f, sampleIndices, binIndices, lr, gatherUpdates);
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
        var predictions = new Vector<T>(input.Rows);

        await Task.Run(() =>
        {
            for (int i = 0; i < input.Rows; i++)
            {
                predictions[i] = PredictSingleInternal(input.GetRow(i));
            }
        });

        return predictions;
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
        var contributions = new Dictionary<int, T>();

        for (int f = 0; f < _numFeatures; f++)
        {
            int binIndex = GetBinIndex(sample[f], f);
            contributions[f] = _shapeFunctions[f][binIndex];
        }

        return contributions;
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

        int numBins = _shapeFunctions[featureIndex].Length;
        var centers = new Vector<T>(numBins);
        var values = new Vector<T>(numBins);

        var edges = _binEdges[featureIndex];

        for (int b = 0; b < numBins; b++)
        {
            // Compute bin center
            if (b == 0 && edges.Length > 0)
            {
                // First bin: use first edge as representative
                centers[b] = edges[0];
            }
            else if (b == numBins - 1)
            {
                // Last bin: use last edge as representative
                centers[b] = edges[^1];
            }
            else if (b < edges.Length)
            {
                // Middle bins: average of edges
                T low = b > 0 ? edges[b - 1] : edges[0];
                T high = edges[b];
                centers[b] = NumOps.Divide(NumOps.Add(low, high), NumOps.FromDouble(2.0));
            }
            else
            {
                centers[b] = NumOps.Zero;
            }

            values[b] = _shapeFunctions[featureIndex][b];
        }

        return (centers, values);
    }

    /// <summary>
    /// Predicts the value for a single sample.
    /// </summary>
    private T PredictSingleInternal(Vector<T> sample)
    {
        T prediction = _intercept;

        // Add main effects (shape functions)
        for (int f = 0; f < _numFeatures; f++)
        {
            int binIndex = GetBinIndex(sample[f], f);
            prediction = NumOps.Add(prediction, _shapeFunctions[f][binIndex]);
        }

        // Add interaction effects
        foreach (var ((f1, f2), interactionMatrix) in _interactionTerms)
        {
            int bin1 = GetBinIndex(sample[f1], f1);
            int bin2 = GetBinIndex(sample[f2], f2);

            // Clamp to interaction matrix bounds
            bin1 = Math.Min(bin1, interactionMatrix.Rows - 1);
            bin2 = Math.Min(bin2, interactionMatrix.Columns - 1);

            prediction = NumOps.Add(prediction, interactionMatrix[bin1, bin2]);
        }

        return prediction;
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
    /// Gets the bin index for a feature value.
    /// </summary>
    private int GetBinIndex(T value, int featureIndex)
    {
        var edges = _binEdges[featureIndex];
        int len = edges.Length;
        if (len == 0) return 0;

        int lo = 0, hi = len;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (NumOps.LessThan(value, edges[mid]))
                hi = mid;
            else
                lo = mid + 1;
        }
        return lo;
    }

    private void UpdateShapeFunction(Vector<T> residuals, int featureIndex,
        int[] sampleIndices, Matrix<int> binIndices, T lr, Vector<T> gatherUpdates)
    {
        int numBins = _shapeFunctions[featureIndex].Length;

        var binSums = Vector<T>.CreateDefault(numBins, NumOps.Zero);
        var binCounts = new int[numBins];
        foreach (int i in sampleIndices)
        {
            int bin = binIndices[i, featureIndex];
            binSums[bin] = NumOps.Add(binSums[bin], residuals[i]);
            binCounts[bin]++;
        }

        var binUpdates = Vector<T>.CreateDefault(numBins, NumOps.Zero);
        int minSamples = _options.MinSamplesPerBin;
        for (int b = 0; b < numBins; b++)
        {
            if (binCounts[b] >= minSamples)
            {
                binUpdates[b] = NumOps.Multiply(lr,
                    NumOps.Divide(binSums[b], NumOps.FromDouble(binCounts[b])));
                _shapeFunctions[featureIndex][b] = NumOps.Add(
                    _shapeFunctions[featureIndex][b], binUpdates[b]);
            }
        }

        // Gather then AVX SIMD in-place subtract — zero allocation
        int n = residuals.Length;
        for (int i = 0; i < n; i++)
            gatherUpdates[i] = binUpdates[binIndices[i, featureIndex]];

        residuals.SubtractInPlace(gatherUpdates);
    }

    /// <summary>
    /// Detects and adds important pairwise interactions.
    /// </summary>
    private void DetectAndAddInteractions(Vector<T> residuals, Matrix<int> binIndices)
    {
        _interactionTerms = new Dictionary<(int, int), Matrix<T>>();

        var interactionScores = new List<((int, int) pair, double score)>();

        for (int f1 = 0; f1 < _numFeatures; f1++)
            for (int f2 = f1 + 1; f2 < _numFeatures; f2++)
                interactionScores.Add(((f1, f2), ComputeInteractionScore(residuals, f1, f2, binIndices)));

        interactionScores.Sort((a, b) => b.score.CompareTo(a.score));
        int numInteractions = Math.Min(_options.MaxInteractionBins, interactionScores.Count);

        for (int i = 0; i < numInteractions; i++)
        {
            var (pair, _) = interactionScores[i];
            FitInteraction(residuals, pair.Item1, pair.Item2, binIndices);
        }
    }

    private double ComputeInteractionScore(Vector<T> residuals, int f1, int f2, Matrix<int> binIndices)
    {
        int numBins1 = Math.Min(5, _shapeFunctions[f1].Length);
        int numBins2 = Math.Min(5, _shapeFunctions[f2].Length);

        var binMeans = new Matrix<T>(numBins1, numBins2);
        var binCounts = new Matrix<int>(numBins1, numBins2);

        int rows = residuals.Length;
        for (int i = 0; i < rows; i++)
        {
            int b1 = binIndices[i, f1] % numBins1;
            int b2 = binIndices[i, f2] % numBins2;
            binMeans[b1, b2] = NumOps.Add(binMeans[b1, b2], residuals[i]);
            binCounts[b1, b2]++;
        }

        double totalVar = 0;
        for (int b1 = 0; b1 < numBins1; b1++)
            for (int b2 = 0; b2 < numBins2; b2++)
                if (binCounts[b1, b2] > 0)
                {
                    double mean = NumOps.ToDouble(binMeans[b1, b2]) / binCounts[b1, b2];
                    totalVar += binCounts[b1, b2] * mean * mean;
                }

        return totalVar;
    }

    private void FitInteraction(Vector<T> residuals, int f1, int f2, Matrix<int> binIndices)
    {
        int numBins1 = Math.Min(_shapeFunctions[f1].Length, 32);
        int numBins2 = Math.Min(_shapeFunctions[f2].Length, 32);

        var interactionMatrix = new Matrix<T>(numBins1, numBins2);
        var binSums = new Matrix<T>(numBins1, numBins2);
        var binCounts = new Matrix<int>(numBins1, numBins2);

        int rows = residuals.Length;
        for (int i = 0; i < rows; i++)
        {
            int b1 = binIndices[i, f1] % numBins1;
            int b2 = binIndices[i, f2] % numBins2;
            binSums[b1, b2] = NumOps.Add(binSums[b1, b2], residuals[i]);
            binCounts[b1, b2]++;
        }

        int minSamples = _options.MinSamplesPerBin;
        for (int b1 = 0; b1 < numBins1; b1++)
            for (int b2 = 0; b2 < numBins2; b2++)
                if (binCounts[b1, b2] >= minSamples)
                    interactionMatrix[b1, b2] = NumOps.Divide(binSums[b1, b2],
                        NumOps.FromDouble(binCounts[b1, b2]));

        _interactionTerms[(f1, f2)] = interactionMatrix;

        // Gather then AVX SIMD in-place subtract
        var gatherUpdates = new Vector<T>(rows);
        for (int i = 0; i < rows; i++)
        {
            int b1 = binIndices[i, f1] % numBins1;
            int b2 = binIndices[i, f2] % numBins2;
            gatherUpdates[i] = interactionMatrix[b1, b2];
        }
        residuals.SubtractInPlace(gatherUpdates);
    }

    /// <summary>
    /// Applies smoothing to shape functions for better interpretability.
    /// </summary>
    private void SmoothShapeFunctions()
    {
        T lambda = NumOps.FromDouble(_options.RegularizationStrength);
        T divisor = NumOps.Add(NumOps.One, NumOps.Multiply(NumOps.FromDouble(2.0), lambda));

        for (int f = 0; f < _numFeatures; f++)
        {
            var sf = _shapeFunctions[f];
            int len = sf.Length;
            if (len <= 2) continue;

            var leftNeighbor = new Vector<T>(len);
            var rightNeighbor = new Vector<T>(len);
            leftNeighbor[0] = sf[0];
            rightNeighbor[len - 1] = sf[len - 1];
            for (int b = 1; b < len; b++)
                leftNeighbor[b] = sf[b - 1];
            for (int b = 0; b < len - 1; b++)
                rightNeighbor[b] = sf[b + 1];

            // In-place: left *= lambda, right *= lambda, left += right, left += sf
            leftNeighbor.MultiplyInPlace(lambda);
            rightNeighbor.MultiplyInPlace(lambda);
            leftNeighbor.AddInPlace(rightNeighbor);
            leftNeighbor.AddInPlace(sf);
            leftNeighbor.MultiplyInPlace(NumOps.Divide(NumOps.One, divisor));
            var smoothed = leftNeighbor;

            _shapeFunctions[f] = smoothed;
        }
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new Vector<T>(_numFeatures);

        for (int f = 0; f < _numFeatures; f++)
            importances[f] = NumOps.Subtract(_shapeFunctions[f].Max(), _shapeFunctions[f].Min());

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

    /// <inheritdoc/>
    public override byte[] Serialize()
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        byte[] baseData = base.Serialize();
        writer.Write(baseData.Length);
        writer.Write(baseData);

        // Options
        writer.Write(_options.NumberOfOuterIterations);
        writer.Write(_options.LearningRate);
        writer.Write(_options.MaxBins);
        writer.Write(_options.DetectInteractions);

        // Model state
        writer.Write(_numFeatures);
        writer.Write(NumOps.ToDouble(_intercept));

        // Bin edges
        for (int f = 0; f < _numFeatures; f++)
        {
            writer.Write(_binEdges[f].Length);
            for (int e = 0; e < _binEdges[f].Length; e++)
            {
                writer.Write(NumOps.ToDouble(_binEdges[f][e]));
            }
        }

        // Shape functions
        for (int f = 0; f < _numFeatures; f++)
        {
            writer.Write(_shapeFunctions[f].Length);
            for (int b = 0; b < _shapeFunctions[f].Length; b++)
            {
                writer.Write(NumOps.ToDouble(_shapeFunctions[f][b]));
            }
        }

        // Interactions
        writer.Write(_interactionTerms.Count);
        foreach (var ((f1, f2), matrix) in _interactionTerms)
        {
            writer.Write(f1);
            writer.Write(f2);
            writer.Write(matrix.Rows);
            writer.Write(matrix.Columns);
            for (int b1 = 0; b1 < matrix.Rows; b1++)
            {
                for (int b2 = 0; b2 < matrix.Columns; b2++)
                {
                    writer.Write(NumOps.ToDouble(matrix[b1, b2]));
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

        // Options
        _options.NumberOfOuterIterations = reader.ReadInt32();
        _options.LearningRate = reader.ReadDouble();
        _options.MaxBins = reader.ReadInt32();
        _options.DetectInteractions = reader.ReadBoolean();

        // Model state
        _numFeatures = reader.ReadInt32();
        _intercept = NumOps.FromDouble(reader.ReadDouble());

        // Bin edges
        _binEdges = new List<Vector<T>>(_numFeatures);
        for (int f = 0; f < _numFeatures; f++)
        {
            int numEdges = reader.ReadInt32();
            var edges = new Vector<T>(numEdges);
            for (int e = 0; e < numEdges; e++)
            {
                edges[e] = NumOps.FromDouble(reader.ReadDouble());
            }
            _binEdges.Add(edges);
        }

        // Shape functions
        _shapeFunctions = new List<Vector<T>>(_numFeatures);
        for (int f = 0; f < _numFeatures; f++)
        {
            int numBins = reader.ReadInt32();
            var sf = new Vector<T>(numBins);
            for (int b = 0; b < numBins; b++)
            {
                sf[b] = NumOps.FromDouble(reader.ReadDouble());
            }
            _shapeFunctions.Add(sf);
        }

        // Interactions
        _interactionTerms = new Dictionary<(int, int), Matrix<T>>();
        int numInteractions = reader.ReadInt32();
        for (int i = 0; i < numInteractions; i++)
        {
            int f1 = reader.ReadInt32();
            int f2 = reader.ReadInt32();
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            var matrix = new Matrix<T>(rows, cols);
            for (int b1 = 0; b1 < rows; b1++)
            {
                for (int b2 = 0; b2 < cols; b2++)
                {
                    matrix[b1, b2] = NumOps.FromDouble(reader.ReadDouble());
                }
            }
            _interactionTerms[(f1, f2)] = matrix;
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Matrix<T>, Vector<T>> CreateNewInstance()
    {
        return new ExplainableBoostingMachineRegression<T>(_options, Regularization);
    }

    public override IFullModel<T, Matrix<T>, Vector<T>> Clone()
    {
        var clone = new ExplainableBoostingMachineRegression<T>(_options, Regularization);
        clone.Deserialize(Serialize());
        return clone;
    }
}
