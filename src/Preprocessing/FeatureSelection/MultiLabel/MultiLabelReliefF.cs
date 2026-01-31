using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiLabel;

/// <summary>
/// Multi-Label ReliefF for multi-label feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Multi-Label ReliefF extends the ReliefF algorithm to handle multi-label
/// classification problems where each instance can belong to multiple classes
/// simultaneously. It considers label correlations when computing feature weights.
/// </para>
/// <para><b>For Beginners:</b> In multi-label problems (like tagging photos with
/// multiple keywords), a single image might be "sunset", "beach", and "romantic"
/// all at once. This algorithm finds features that help distinguish between
/// different label combinations, not just individual labels.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MultiLabelReliefF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly int _nIterations;
    private readonly int? _randomState;

    private double[]? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public int NIterations => _nIterations;
    public double[]? FeatureWeights => _featureWeights;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MultiLabelReliefF(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        int nIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MultiLabelReliefF requires label matrix. Use Fit(Matrix<T> data, Matrix<T> labels) instead.");
    }

    public void Fit(Matrix<T> data, Matrix<T> labels)
    {
        if (data.Rows != labels.Rows)
            throw new ArgumentException("Number of samples must match between data and labels.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int numLabels = labels.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        _featureWeights = new double[p];

        // Precompute label vectors for each instance
        var labelVectors = new bool[n][];
        for (int i = 0; i < n; i++)
        {
            labelVectors[i] = new bool[numLabels];
            for (int l = 0; l < numLabels; l++)
                labelVectors[i][l] = NumOps.ToDouble(labels[i, l]) > 0.5;
        }

        // Main ReliefF loop
        int numIterations = Math.Min(_nIterations, n);

        for (int iter = 0; iter < numIterations; iter++)
        {
            int idx = random.Next(n);

            // Find nearest hits and misses based on label similarity
            var similarities = new List<(int Index, double Similarity, double Distance)>();

            for (int j = 0; j < n; j++)
            {
                if (j == idx) continue;

                double labelSim = ComputeLabelSimilarity(labelVectors[idx], labelVectors[j]);
                double dist = ComputeEuclideanDistance(data, idx, j, p);

                similarities.Add((j, labelSim, dist));
            }

            // Find nearest hits (similar labels)
            var nearestHits = similarities
                .Where(s => s.Similarity > 0.5)
                .OrderBy(s => s.Distance)
                .Take(_nNeighbors)
                .ToList();

            // Find nearest misses (dissimilar labels)
            var nearestMisses = similarities
                .Where(s => s.Similarity <= 0.5)
                .OrderBy(s => s.Distance)
                .Take(_nNeighbors)
                .ToList();

            // Update feature weights
            for (int j = 0; j < p; j++)
            {
                double hitContrib = 0;
                double missContrib = 0;

                foreach (var hit in nearestHits)
                {
                    double diff = Math.Abs(NumOps.ToDouble(data[idx, j]) - NumOps.ToDouble(data[hit.Index, j]));
                    hitContrib += diff * (1 - hit.Similarity);
                }

                foreach (var miss in nearestMisses)
                {
                    double diff = Math.Abs(NumOps.ToDouble(data[idx, j]) - NumOps.ToDouble(data[miss.Index, j]));
                    missContrib += diff * (1 - miss.Similarity);
                }

                int hitCount = Math.Max(1, nearestHits.Count);
                int missCount = Math.Max(1, nearestMisses.Count);

                _featureWeights[j] += missContrib / missCount - hitContrib / hitCount;
            }
        }

        // Normalize weights
        for (int j = 0; j < p; j++)
            _featureWeights[j] /= numIterations;

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _featureWeights
            .Select((w, idx) => (Weight: w, Index: idx))
            .OrderByDescending(x => x.Weight)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        // Convert single-label to multi-label format
        int n = data.Rows;
        var uniqueLabels = new HashSet<int>();
        for (int i = 0; i < n; i++)
            uniqueLabels.Add((int)Math.Round(NumOps.ToDouble(target[i])));

        var labelList = uniqueLabels.OrderBy(x => x).ToList();
        int numLabels = labelList.Count;

        var labelsArray = new T[n, numLabels];
        for (int i = 0; i < n; i++)
        {
            int label = (int)Math.Round(NumOps.ToDouble(target[i]));
            int labelIdx = labelList.IndexOf(label);
            labelsArray[i, labelIdx] = NumOps.FromDouble(1.0);
        }

        Fit(data, new Matrix<T>(labelsArray));
    }

    private double ComputeLabelSimilarity(bool[] labels1, bool[] labels2)
    {
        // Jaccard similarity
        int intersection = 0;
        int union = 0;

        for (int i = 0; i < labels1.Length; i++)
        {
            if (labels1[i] && labels2[i]) intersection++;
            if (labels1[i] || labels2[i]) union++;
        }

        return union > 0 ? (double)intersection / union : 0;
    }

    private double ComputeEuclideanDistance(Matrix<T> data, int i1, int i2, int p)
    {
        double sum = 0;
        for (int j = 0; j < p; j++)
        {
            double diff = NumOps.ToDouble(data[i1, j]) - NumOps.ToDouble(data[i2, j]);
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Matrix<T> labels)
    {
        Fit(data, labels);
        return Transform(data);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiLabelReliefF has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MultiLabelReliefF does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiLabelReliefF has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
