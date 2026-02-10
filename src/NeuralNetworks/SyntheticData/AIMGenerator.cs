using AiDotNet.Helpers;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// AIM (Adaptive Iterative Mechanism) generator for differentially private synthetic data
/// generation using marginal-based measurements and iterative optimization.
/// </summary>
/// <remarks>
/// <para>
/// AIM is a non-neural, statistical approach that works by:
///
/// <code>
///  Real Data ──► Discretize ──► Marginal Selection ──► Noisy Measurement ──► Synthetic Optimization
///                    │              (exponential            (Gaussian           (iteratively refine
///                    │               mechanism)              noise)              synthetic data)
///                    ▼
///              Bin edges for         "Which column         "What are the         "Adjust synthetic
///              each column"           pairs matter?"        noisy counts?"        data to match"
/// </code>
///
/// No neural networks are used — only statistics and privacy-preserving mechanisms.
/// </para>
/// <para>
/// <b>For Beginners:</b> AIM works like a privacy-preserving census:
///
/// 1. First, simplify the data into categories (binning)
/// 2. Then, privately count how many people are in each category combination
/// 3. Finally, create synthetic data that matches those noisy counts
///
/// It's simpler than deep learning approaches but often works better for:
/// - Small datasets
/// - When you need formal privacy guarantees
/// - When training time is limited
/// </para>
/// <para>
/// Reference: "AIM: An Adaptive and Iterative Mechanism for Differentially Private
/// Synthetic Data" (McKenna et al., 2022)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class AIMGenerator<T> : SyntheticTabularGeneratorBase<T>
{
    private readonly AIMOptions<T> _options;

    // Discretization info
    private readonly List<double[]> _binEdges = new();
    private readonly List<int> _numBinsPerCol = new();

    // Measured marginals (noisy)
    private readonly List<MarginalMeasurement> _measurements = new();

    // Synthetic data representation (probabilities over discretized space)
    private Matrix<T>? _syntheticData;
    private int _numCols;

    private record MarginalMeasurement(int[] ColumnIndices, double[] NoisyCounts);

    /// <summary>
    /// Initializes a new AIM generator.
    /// </summary>
    /// <param name="options">Configuration options for AIM.</param>
    public AIMGenerator(AIMOptions<T> options)
        : base(options.Seed)
    {
        _options = options;
    }

    /// <inheritdoc />
    protected override void FitInternal(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns, int epochs)
    {
        _numCols = columns.Count;

        // Step 1: Discretize all columns
        DiscretizeColumns(data, columns);

        // Discretize the data
        var discretized = DiscretizeData(data);

        // Step 2: Allocate privacy budget
        double epsPerIteration = _options.Epsilon / _options.NumIterations;
        double epsSelection = epsPerIteration * 0.5;
        double epsMeasurement = epsPerIteration * 0.5;

        // Step 3: Iterative marginal selection and measurement
        _measurements.Clear();

        // Initialize synthetic data from uniform
        int numRows = data.Rows;
        _syntheticData = new Matrix<T>(numRows, _numCols);
        for (int i = 0; i < numRows; i++)
        {
            for (int j = 0; j < _numCols; j++)
            {
                int numBins = _numBinsPerCol[j];
                _syntheticData[i, j] = NumOps.FromDouble(Random.Next(numBins));
            }
        }

        for (int iter = 0; iter < _options.NumIterations; iter++)
        {
            // Select marginals using exponential mechanism
            var selectedMarginals = SelectMarginals(discretized, epsSelection);

            // Measure selected marginals with Gaussian noise
            foreach (var marginal in selectedMarginals)
            {
                var noisyCounts = MeasureMarginal(discretized, marginal, epsMeasurement);
                _measurements.Add(new MarginalMeasurement(marginal, noisyCounts));
            }

            // Update synthetic data to better match measurements
            UpdateSyntheticData(numRows);
        }
    }

    /// <inheritdoc />
    protected override Matrix<T> GenerateInternal(int numSamples, Vector<T>? conditionColumn, Vector<T>? conditionValue)
    {
        if (_syntheticData is null)
        {
            throw new InvalidOperationException("Generator is not fitted.");
        }

        var result = new Matrix<T>(numSamples, _numCols);

        for (int i = 0; i < numSamples; i++)
        {
            // Sample a row from synthetic data (with replacement)
            int srcRow = Random.Next(_syntheticData.Rows);

            for (int j = 0; j < _numCols; j++)
            {
                // Convert discretized value back to continuous
                int bin = Math.Min(Math.Max((int)Math.Round(NumOps.ToDouble(_syntheticData[srcRow, j])),
                    0), _numBinsPerCol[j] - 1);

                if (j < _binEdges.Count && _binEdges[j].Length > 1)
                {
                    // Add uniform noise within the bin for continuous columns
                    double low = _binEdges[j][bin];
                    double high = bin + 1 < _binEdges[j].Length ? _binEdges[j][bin + 1] : low;
                    result[i, j] = NumOps.FromDouble(low + Random.NextDouble() * (high - low));
                }
                else
                {
                    result[i, j] = NumOps.FromDouble(bin);
                }
            }
        }

        return result;
    }

    #region Discretization

    private void DiscretizeColumns(Matrix<T> data, IReadOnlyList<ColumnMetadata> columns)
    {
        _binEdges.Clear();
        _numBinsPerCol.Clear();

        for (int c = 0; c < columns.Count; c++)
        {
            if (columns[c].IsNumerical)
            {
                // Compute quantile-based bin edges
                var values = new List<double>();
                for (int r = 0; r < data.Rows; r++)
                    values.Add(NumOps.ToDouble(data[r, c]));
                values.Sort();

                int numBins = _options.NumBins;
                var edges = new double[numBins + 1];
                for (int b = 0; b <= numBins; b++)
                {
                    int idx = (int)((long)b * (values.Count - 1) / numBins);
                    edges[b] = values[Math.Min(idx, values.Count - 1)];
                }

                _binEdges.Add(edges);
                _numBinsPerCol.Add(numBins);
            }
            else
            {
                // Categorical: each category is a bin
                int numCats = Math.Max(2, columns[c].Categories.Count);
                _binEdges.Add(Array.Empty<double>());
                _numBinsPerCol.Add(numCats);
            }
        }
    }

    private int[,] DiscretizeData(Matrix<T> data)
    {
        var discretized = new int[data.Rows, _numCols];

        for (int r = 0; r < data.Rows; r++)
        {
            for (int c = 0; c < _numCols; c++)
            {
                double val = NumOps.ToDouble(data[r, c]);

                if (_binEdges[c].Length > 1)
                {
                    discretized[r, c] = FindBin(val, _binEdges[c]);
                }
                else
                {
                    discretized[r, c] = Math.Min(Math.Max((int)Math.Round(val), 0), _numBinsPerCol[c] - 1);
                }
            }
        }

        return discretized;
    }

    private static int FindBin(double value, double[] edges)
    {
        for (int b = 0; b < edges.Length - 1; b++)
        {
            if (value <= edges[b + 1]) return b;
        }
        return Math.Max(0, edges.Length - 2);
    }

    #endregion

    #region Marginal Selection

    private List<int[]> SelectMarginals(int[,] discretized, double epsilon)
    {
        var selected = new List<int[]>();
        var allMarginals = GenerateCandidateMarginals();

        for (int m = 0; m < _options.MarginalsPerIteration && allMarginals.Count > 0; m++)
        {
            // Score each candidate marginal by how much it would improve the synthetic data
            var scores = new double[allMarginals.Count];
            for (int i = 0; i < allMarginals.Count; i++)
            {
                scores[i] = ScoreMarginal(discretized, allMarginals[i]);
            }

            // Exponential mechanism: sample proportional to exp(eps * score / (2 * sensitivity))
            double sensitivity = 1.0; // Sensitivity of the scoring function
            int selectedIdx = ExponentialMechanism(scores, epsilon, sensitivity);

            selected.Add(allMarginals[selectedIdx]);
            allMarginals.RemoveAt(selectedIdx);
        }

        return selected;
    }

    private List<int[]> GenerateCandidateMarginals()
    {
        var candidates = new List<int[]>();

        // Order 1: individual columns
        for (int i = 0; i < _numCols; i++)
        {
            candidates.Add([i]);
        }

        // Order 2: pairs (if allowed)
        if (_options.MaxMarginalOrder >= 2)
        {
            for (int i = 0; i < _numCols; i++)
            {
                for (int j = i + 1; j < _numCols; j++)
                {
                    candidates.Add([i, j]);
                }
            }
        }

        return candidates;
    }

    private double ScoreMarginal(int[,] discretized, int[] colIndices)
    {
        // Score based on how much the synthetic data differs from real data on this marginal
        if (_syntheticData is null) return 0;

        var realCounts = ComputeMarginalCounts(discretized, colIndices);
        var synCounts = ComputeSyntheticMarginalCounts(colIndices);

        double totalError = 0;
        int totalBins = realCounts.Length;
        int numRows = discretized.GetLength(0);

        for (int b = 0; b < totalBins; b++)
        {
            double realFreq = (double)realCounts[b] / numRows;
            double synFreq = synCounts[b] / _syntheticData.Rows;
            totalError += Math.Abs(realFreq - synFreq);
        }

        return totalError;
    }

    private int ExponentialMechanism(double[] scores, double epsilon, double sensitivity)
    {
        double maxScore = double.MinValue;
        foreach (double s in scores) if (s > maxScore) maxScore = s;

        var probs = new double[scores.Length];
        double sumProbs = 0;
        for (int i = 0; i < scores.Length; i++)
        {
            probs[i] = Math.Exp(epsilon * scores[i] / (2.0 * sensitivity) - maxScore);
            sumProbs += probs[i];
        }

        double u = Random.NextDouble() * sumProbs;
        double cumSum = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            cumSum += probs[i];
            if (u <= cumSum) return i;
        }

        return probs.Length - 1;
    }

    #endregion

    #region Marginal Measurement

    private double[] MeasureMarginal(int[,] discretized, int[] colIndices, double epsilon)
    {
        var counts = ComputeMarginalCounts(discretized, colIndices);

        // Add Gaussian noise calibrated to the privacy budget
        double sigma = Math.Sqrt(2.0 * Math.Log(1.25 / 1e-5)) / epsilon;
        var noisyCounts = new double[counts.Length];
        for (int i = 0; i < counts.Length; i++)
        {
            double noise = NumOps.ToDouble(SampleStandardNormal()) * sigma;
            noisyCounts[i] = Math.Max(0, counts[i] + noise);
        }

        return noisyCounts;
    }

    private int[] ComputeMarginalCounts(int[,] discretized, int[] colIndices)
    {
        int totalBins = 1;
        foreach (int c in colIndices)
            totalBins *= _numBinsPerCol[c];

        var counts = new int[totalBins];
        int numRows = discretized.GetLength(0);

        for (int r = 0; r < numRows; r++)
        {
            int flatIdx = 0;
            int multiplier = 1;
            for (int k = colIndices.Length - 1; k >= 0; k--)
            {
                flatIdx += discretized[r, colIndices[k]] * multiplier;
                multiplier *= _numBinsPerCol[colIndices[k]];
            }
            if (flatIdx >= 0 && flatIdx < counts.Length)
                counts[flatIdx]++;
        }

        return counts;
    }

    private double[] ComputeSyntheticMarginalCounts(int[] colIndices)
    {
        if (_syntheticData is null) return Array.Empty<double>();

        int totalBins = 1;
        foreach (int c in colIndices)
            totalBins *= _numBinsPerCol[c];

        var counts = new double[totalBins];

        for (int r = 0; r < _syntheticData.Rows; r++)
        {
            int flatIdx = 0;
            int multiplier = 1;
            for (int k = colIndices.Length - 1; k >= 0; k--)
            {
                int bin = Math.Min(Math.Max(
                    (int)Math.Round(NumOps.ToDouble(_syntheticData[r, colIndices[k]])),
                    0), _numBinsPerCol[colIndices[k]] - 1);
                flatIdx += bin * multiplier;
                multiplier *= _numBinsPerCol[colIndices[k]];
            }
            if (flatIdx >= 0 && flatIdx < counts.Length)
                counts[flatIdx] += 1.0;
        }

        return counts;
    }

    #endregion

    #region Synthetic Data Update

    private void UpdateSyntheticData(int numRows)
    {
        if (_syntheticData is null || _measurements.Count == 0) return;

        double lr = _options.LearningRate;

        // For each measurement, nudge synthetic data to better match
        foreach (var measurement in _measurements)
        {
            var synCounts = ComputeSyntheticMarginalCounts(measurement.ColumnIndices);
            var noisyCounts = measurement.NoisyCounts;

            // Compute total for normalization
            double totalNoisy = 0;
            double totalSyn = 0;
            for (int b = 0; b < noisyCounts.Length; b++)
            {
                totalNoisy += noisyCounts[b];
                totalSyn += synCounts[b];
            }
            if (totalNoisy < 1e-10 || totalSyn < 1e-10) continue;

            // Identify bins that need more/fewer samples
            for (int r = 0; r < numRows; r++)
            {
                int flatIdx = 0;
                int multiplier = 1;
                var colIndices = measurement.ColumnIndices;
                for (int k = colIndices.Length - 1; k >= 0; k--)
                {
                    int bin = Math.Min(Math.Max(
                        (int)Math.Round(NumOps.ToDouble(_syntheticData[r, colIndices[k]])),
                        0), _numBinsPerCol[colIndices[k]] - 1);
                    flatIdx += bin * multiplier;
                    multiplier *= _numBinsPerCol[colIndices[k]];
                }

                if (flatIdx < 0 || flatIdx >= noisyCounts.Length) continue;

                double noisyFreq = noisyCounts[flatIdx] / totalNoisy;
                double synFreq = synCounts[flatIdx] / totalSyn;

                // If this bin is overrepresented, randomly reassign some values
                if (synFreq > noisyFreq + 0.01 && Random.NextDouble() < lr * (synFreq - noisyFreq))
                {
                    // Reassign one of the columns to a random bin
                    int colToChange = colIndices[Random.Next(colIndices.Length)];
                    _syntheticData[r, colToChange] = NumOps.FromDouble(Random.Next(_numBinsPerCol[colToChange]));
                }
            }
        }
    }

    #endregion
}
