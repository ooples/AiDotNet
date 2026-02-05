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
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class ExplainableBoostingMachineRegression<T> : AsyncDecisionTreeRegressionBase<T>
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
    /// Indexed as: _interactionTerms[(feat1, feat2)][(bin1, bin2)]
    /// </summary>
    private Dictionary<(int, int), T[,]> _interactionTerms;

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
    public IReadOnlyList<T[]> ShapeFunctions => _shapeFunctions;

    /// <summary>
    /// Gets the bin edges for each feature.
    /// </summary>
    public IReadOnlyList<T[]> BinEdges => _binEdges;

    /// <summary>
    /// Gets the interaction terms.
    /// </summary>
    public IReadOnlyDictionary<(int, int), T[,]> InteractionTerms => _interactionTerms;

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
        _interactionTerms = [];
        _intercept = NumOps.Zero;
        _numFeatures = 0;
    }

    /// <inheritdoc/>
    public override async Task TrainAsync(Matrix<T> x, Vector<T> y)
    {
        _numFeatures = x.Columns;
        int n = x.Rows;

        // Initialize intercept as mean of y
        double mean = 0;
        for (int i = 0; i < n; i++)
        {
            mean += NumOps.ToDouble(y[i]);
        }
        mean /= n;
        _intercept = NumOps.FromDouble(mean);

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

        // Initialize residuals
        var residuals = new T[n];
        for (int i = 0; i < n; i++)
        {
            residuals[i] = NumOps.Subtract(y[i], _intercept);
        }

        // Train using cyclic coordinate descent with boosting
        int[] featureOrder = Enumerable.Range(0, _numFeatures).ToArray();

        for (int outer = 0; outer < _options.NumberOfOuterIterations; outer++)
        {
            // Shuffle feature order if not cyclic
            if (!_options.CyclicTraining)
            {
                ShuffleArray(featureOrder);
            }

            // Subsample
            int[] sampleIndices = GetSampleIndices(n);

            foreach (int f in featureOrder)
            {
                for (int inner = 0; inner < _options.NumberOfInnerIterations; inner++)
                {
                    // Fit a simple model to residuals for this feature
                    UpdateShapeFunction(x, residuals, f, sampleIndices);
                }
            }
        }

        // Detect and add interactions if enabled
        if (_options.DetectInteractions)
        {
            DetectAndAddInteractions(x, y, residuals);
        }

        // Apply smoothing regularization
        if (_options.RegularizationStrength > 0)
        {
            SmoothShapeFunctions();
        }

        // Calculate feature importances
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
    public (T[] BinCenters, T[] ShapeValues) GetShapeFunctionForPlot(int featureIndex)
    {
        if (featureIndex < 0 || featureIndex >= _numFeatures)
            throw new ArgumentOutOfRangeException(nameof(featureIndex));

        int numBins = _shapeFunctions[featureIndex].Length;
        var centers = new T[numBins];
        var values = new T[numBins];

        T[] edges = _binEdges[featureIndex];

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
                double low = b > 0 ? NumOps.ToDouble(edges[b - 1]) : NumOps.ToDouble(edges[0]);
                double high = NumOps.ToDouble(edges[b]);
                centers[b] = NumOps.FromDouble((low + high) / 2);
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
            bin1 = Math.Min(bin1, interactionMatrix.GetLength(0) - 1);
            bin2 = Math.Min(bin2, interactionMatrix.GetLength(1) - 1);

            prediction = NumOps.Add(prediction, interactionMatrix[bin1, bin2]);
        }

        return prediction;
    }

    /// <summary>
    /// Creates bins for each feature using quantile binning.
    /// </summary>
    private void CreateBins(Matrix<T> x)
    {
        _binEdges = new T[_numFeatures][];

        for (int f = 0; f < _numFeatures; f++)
        {
            // Get unique values for this feature
            var values = new List<double>();
            for (int i = 0; i < x.Rows; i++)
            {
                values.Add(NumOps.ToDouble(x[i, f]));
            }
            values.Sort();

            // Determine number of bins based on unique values
            int numUnique = values.Distinct().Count();
            int numBins = Math.Min(_options.MaxBins, numUnique);

            if (numBins <= 1)
            {
                _binEdges[f] = [];
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

            _binEdges[f] = [.. edges];
        }
    }

    /// <summary>
    /// Gets the bin index for a feature value.
    /// </summary>
    private int GetBinIndex(T value, int featureIndex)
    {
        T[] edges = _binEdges[featureIndex];
        if (edges.Length == 0) return 0;

        double v = NumOps.ToDouble(value);
        for (int b = 0; b < edges.Length; b++)
        {
            if (v < NumOps.ToDouble(edges[b]))
            {
                return b;
            }
        }

        return edges.Length;  // Last bin
    }

    /// <summary>
    /// Updates the shape function for a single feature.
    /// </summary>
    private void UpdateShapeFunction(Matrix<T> x, T[] residuals, int featureIndex, int[] sampleIndices)
    {
        int numBins = _shapeFunctions[featureIndex].Length;

        // Accumulate gradients per bin
        var binSums = new double[numBins];
        var binCounts = new int[numBins];

        foreach (int i in sampleIndices)
        {
            int bin = GetBinIndex(x[i, featureIndex], featureIndex);
            binSums[bin] += NumOps.ToDouble(residuals[i]);
            binCounts[bin]++;
        }

        // Update shape function and residuals
        for (int b = 0; b < numBins; b++)
        {
            if (binCounts[b] >= _options.MinSamplesPerBin)
            {
                double update = _options.LearningRate * binSums[b] / binCounts[b];
                _shapeFunctions[featureIndex][b] = NumOps.Add(
                    _shapeFunctions[featureIndex][b],
                    NumOps.FromDouble(update));
            }
        }

        // Update residuals for all samples
        for (int i = 0; i < residuals.Length; i++)
        {
            int bin = GetBinIndex(x[i, featureIndex], featureIndex);
            if (binCounts[bin] >= _options.MinSamplesPerBin)
            {
                double update = _options.LearningRate * binSums[bin] / binCounts[bin];
                residuals[i] = NumOps.Subtract(residuals[i], NumOps.FromDouble(update));
            }
        }
    }

    /// <summary>
    /// Detects and adds important pairwise interactions.
    /// </summary>
    private void DetectAndAddInteractions(Matrix<T> x, Vector<T> y, T[] residuals)
    {
        _interactionTerms = [];

        // Use FAST algorithm to detect interactions
        var interactionScores = new List<((int, int) pair, double score)>();

        // Compute interaction importance for all pairs
        for (int f1 = 0; f1 < _numFeatures; f1++)
        {
            for (int f2 = f1 + 1; f2 < _numFeatures; f2++)
            {
                double score = ComputeInteractionScore(x, residuals, f1, f2);
                interactionScores.Add(((f1, f2), score));
            }
        }

        // Sort by score and take top interactions
        interactionScores.Sort((a, b) => b.score.CompareTo(a.score));
        int numInteractions = Math.Min(_options.MaxInteractionBins, interactionScores.Count);

        for (int i = 0; i < numInteractions; i++)
        {
            var (pair, _) = interactionScores[i];
            FitInteraction(x, residuals, pair.Item1, pair.Item2);
        }
    }

    /// <summary>
    /// Computes the interaction score for a pair of features.
    /// </summary>
    private double ComputeInteractionScore(Matrix<T> x, T[] residuals, int f1, int f2)
    {
        // Simple variance-based score for interaction detection
        int numBins1 = Math.Min(5, _shapeFunctions[f1].Length);
        int numBins2 = Math.Min(5, _shapeFunctions[f2].Length);

        var binMeans = new double[numBins1, numBins2];
        var binCounts = new int[numBins1, numBins2];

        for (int i = 0; i < residuals.Length; i++)
        {
            int b1 = GetBinIndex(x[i, f1], f1) % numBins1;
            int b2 = GetBinIndex(x[i, f2], f2) % numBins2;
            binMeans[b1, b2] += NumOps.ToDouble(residuals[i]);
            binCounts[b1, b2]++;
        }

        // Compute variance explained
        double totalVar = 0;
        for (int b1 = 0; b1 < numBins1; b1++)
        {
            for (int b2 = 0; b2 < numBins2; b2++)
            {
                if (binCounts[b1, b2] > 0)
                {
                    binMeans[b1, b2] /= binCounts[b1, b2];
                    totalVar += binCounts[b1, b2] * binMeans[b1, b2] * binMeans[b1, b2];
                }
            }
        }

        return totalVar;
    }

    /// <summary>
    /// Fits an interaction term for a pair of features.
    /// </summary>
    private void FitInteraction(Matrix<T> x, T[] residuals, int f1, int f2)
    {
        int numBins1 = _shapeFunctions[f1].Length;
        int numBins2 = _shapeFunctions[f2].Length;

        // Limit interaction resolution
        numBins1 = Math.Min(numBins1, 32);
        numBins2 = Math.Min(numBins2, 32);

        var interactionMatrix = new T[numBins1, numBins2];
        var binSums = new double[numBins1, numBins2];
        var binCounts = new int[numBins1, numBins2];

        // Accumulate residuals
        for (int i = 0; i < residuals.Length; i++)
        {
            int b1 = GetBinIndex(x[i, f1], f1) % numBins1;
            int b2 = GetBinIndex(x[i, f2], f2) % numBins2;
            binSums[b1, b2] += NumOps.ToDouble(residuals[i]);
            binCounts[b1, b2]++;
        }

        // Compute interaction effects
        for (int b1 = 0; b1 < numBins1; b1++)
        {
            for (int b2 = 0; b2 < numBins2; b2++)
            {
                if (binCounts[b1, b2] >= _options.MinSamplesPerBin)
                {
                    interactionMatrix[b1, b2] = NumOps.FromDouble(binSums[b1, b2] / binCounts[b1, b2]);
                }
                else
                {
                    interactionMatrix[b1, b2] = NumOps.Zero;
                }
            }
        }

        _interactionTerms[(f1, f2)] = interactionMatrix;

        // Update residuals
        for (int i = 0; i < residuals.Length; i++)
        {
            int b1 = GetBinIndex(x[i, f1], f1) % numBins1;
            int b2 = GetBinIndex(x[i, f2], f2) % numBins2;
            residuals[i] = NumOps.Subtract(residuals[i], interactionMatrix[b1, b2]);
        }
    }

    /// <summary>
    /// Applies smoothing to shape functions for better interpretability.
    /// </summary>
    private void SmoothShapeFunctions()
    {
        double lambda = _options.RegularizationStrength;

        for (int f = 0; f < _numFeatures; f++)
        {
            if (_shapeFunctions[f].Length <= 2) continue;

            // Apply simple moving average smoothing
            var smoothed = new T[_shapeFunctions[f].Length];
            for (int b = 0; b < _shapeFunctions[f].Length; b++)
            {
                double sum = NumOps.ToDouble(_shapeFunctions[f][b]);
                int count = 1;

                if (b > 0)
                {
                    sum += lambda * NumOps.ToDouble(_shapeFunctions[f][b - 1]);
                    count++;
                }
                if (b < _shapeFunctions[f].Length - 1)
                {
                    sum += lambda * NumOps.ToDouble(_shapeFunctions[f][b + 1]);
                    count++;
                }

                smoothed[b] = NumOps.FromDouble(sum / (1 + 2 * lambda));
            }

            _shapeFunctions[f] = smoothed;
        }
    }

    /// <inheritdoc/>
    protected override Task CalculateFeatureImportancesAsync(int featureCount)
    {
        var importances = new T[_numFeatures];

        for (int f = 0; f < _numFeatures; f++)
        {
            // Importance = range of shape function values
            double min = double.MaxValue;
            double max = double.MinValue;

            foreach (T val in _shapeFunctions[f])
            {
                double v = NumOps.ToDouble(val);
                min = Math.Min(min, v);
                max = Math.Max(max, v);
            }

            importances[f] = NumOps.FromDouble(max - min);
        }

        // Normalize
        double sum = importances.Sum(x => NumOps.ToDouble(x));
        if (sum > 0)
        {
            for (int f = 0; f < _numFeatures; f++)
            {
                importances[f] = NumOps.Divide(importances[f], NumOps.FromDouble(sum));
            }
        }

        FeatureImportances = new Vector<T>(importances);
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
            ModelType = ModelType.ExplainableBoostingMachine,
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
        writer.Write(Convert.ToDouble(_intercept));

        // Bin edges
        for (int f = 0; f < _numFeatures; f++)
        {
            writer.Write(_binEdges[f].Length);
            foreach (T edge in _binEdges[f])
            {
                writer.Write(Convert.ToDouble(edge));
            }
        }

        // Shape functions
        for (int f = 0; f < _numFeatures; f++)
        {
            writer.Write(_shapeFunctions[f].Length);
            foreach (T val in _shapeFunctions[f])
            {
                writer.Write(Convert.ToDouble(val));
            }
        }

        // Interactions
        writer.Write(_interactionTerms.Count);
        foreach (var ((f1, f2), matrix) in _interactionTerms)
        {
            writer.Write(f1);
            writer.Write(f2);
            writer.Write(matrix.GetLength(0));
            writer.Write(matrix.GetLength(1));
            for (int b1 = 0; b1 < matrix.GetLength(0); b1++)
            {
                for (int b2 = 0; b2 < matrix.GetLength(1); b2++)
                {
                    writer.Write(Convert.ToDouble(matrix[b1, b2]));
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
        _binEdges = new T[_numFeatures][];
        for (int f = 0; f < _numFeatures; f++)
        {
            int numEdges = reader.ReadInt32();
            _binEdges[f] = new T[numEdges];
            for (int e = 0; e < numEdges; e++)
            {
                _binEdges[f][e] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Shape functions
        _shapeFunctions = new T[_numFeatures][];
        for (int f = 0; f < _numFeatures; f++)
        {
            int numBins = reader.ReadInt32();
            _shapeFunctions[f] = new T[numBins];
            for (int b = 0; b < numBins; b++)
            {
                _shapeFunctions[f][b] = NumOps.FromDouble(reader.ReadDouble());
            }
        }

        // Interactions
        _interactionTerms = [];
        int numInteractions = reader.ReadInt32();
        for (int i = 0; i < numInteractions; i++)
        {
            int f1 = reader.ReadInt32();
            int f2 = reader.ReadInt32();
            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();
            var matrix = new T[rows, cols];
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
}
