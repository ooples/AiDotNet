using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models.Options;
using AiDotNet.Preprocessing;

namespace AiDotNet.CausalDiscovery;

/// <summary>
/// Feature selector that uses any causal discovery algorithm to select features based on causal relationships.
/// </summary>
/// <remarks>
/// <para>
/// This wrapper allows any <see cref="ICausalDiscoveryAlgorithm{T}"/> to be used as a feature selector
/// within the preprocessing pipeline. It discovers the causal graph, then selects the top features
/// that have the strongest causal connections to the target variable.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of selecting features by correlation or mutual information, this
/// selector uses causal discovery to find features that actually CAUSE the target variable (or are
/// caused by it). This often leads to better, more robust models because causal features remain
/// predictive even when the data distribution changes.
/// </para>
/// <para>
/// <b>Usage:</b>
/// <code>
/// var selector = new CausalDiscoverySelector&lt;double&gt;(
///     CausalDiscoveryAlgorithmType.NOTEARSLinear, maxFeatures: 10);
/// selector.Fit(data, target);
/// var selectedData = selector.Transform(data);
/// </code>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class CausalDiscoverySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly ICausalDiscoveryAlgorithm<T> _algorithm;
    private readonly int _maxFeatures;
    private readonly double _edgeThreshold;

    private int[]? _selectedIndices;
    private double[]? _connectionStrengths;
    private CausalGraph<T>? _causalGraph;
    private int _nInputFeatures;

    private readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets the discovered causal graph from the most recent Fit() call.
    /// </summary>
    public CausalGraph<T>? CausalGraph => _causalGraph;

    /// <summary>
    /// Gets the selected feature indices.
    /// </summary>
    public int[]? SelectedIndices => _selectedIndices;

    /// <summary>
    /// Gets the connection strengths of each feature to the target.
    /// </summary>
    public double[]? ConnectionStrengths => _connectionStrengths;

    /// <inheritdoc/>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new CausalDiscoverySelector using the specified algorithm type.
    /// </summary>
    /// <param name="algorithmType">The causal discovery algorithm to use.</param>
    /// <param name="maxFeatures">Maximum number of features to select. Default: 20.</param>
    /// <param name="edgeThreshold">Minimum edge weight to consider a causal connection. Default: 0.1.</param>
    /// <param name="options">Optional algorithm configuration.</param>
    /// <param name="columnIndices">Optional column indices to restrict analysis to.</param>
    public CausalDiscoverySelector(
        CausalDiscoveryAlgorithmType algorithmType = CausalDiscoveryAlgorithmType.NOTEARSLinear,
        int maxFeatures = 20,
        double edgeThreshold = 0.1,
        CausalDiscoveryOptions? options = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _algorithm = CausalDiscoveryAlgorithmFactory<T>.Create(algorithmType, options);
        _maxFeatures = maxFeatures;
        _edgeThreshold = edgeThreshold;
    }

    /// <summary>
    /// Creates a new CausalDiscoverySelector using a pre-created algorithm instance.
    /// </summary>
    /// <param name="algorithm">The causal discovery algorithm to use.</param>
    /// <param name="maxFeatures">Maximum number of features to select.</param>
    /// <param name="edgeThreshold">Minimum edge weight to consider a causal connection.</param>
    /// <param name="columnIndices">Optional column indices to restrict analysis to.</param>
    public CausalDiscoverySelector(
        ICausalDiscoveryAlgorithm<T> algorithm,
        int maxFeatures = 20,
        double edgeThreshold = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        _algorithm = algorithm;
        _maxFeatures = maxFeatures;
        _edgeThreshold = edgeThreshold;
    }

    /// <inheritdoc/>
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "CausalDiscoverySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <summary>
    /// Fits the selector by discovering the causal graph and selecting features connected to the target.
    /// </summary>
    /// <param name="data">Data matrix [n_samples, n_features].</param>
    /// <param name="target">Target variable vector.</param>
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Create augmented matrix with target as last column
        var augmented = new Matrix<T>(n, p + 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
                augmented[i, j] = data[i, j];
            augmented[i, p] = target[i];
        }

        // Generate feature names
        var featureNames = new string[p + 1];
        for (int j = 0; j < p; j++)
            featureNames[j] = $"Feature{j}";
        featureNames[p] = "Target";

        // Discover causal structure
        _causalGraph = _algorithm.DiscoverStructure(augmented, featureNames);

        // Compute connection strengths to target (index p)
        int targetIdx = p;
        _connectionStrengths = new double[p];
        for (int j = 0; j < p; j++)
        {
            // Consider both directions: feature→target and target→feature
            double toTarget = _numOps.ToDouble(_causalGraph.GetEdgeWeight(j, targetIdx));
            double fromTarget = _numOps.ToDouble(_causalGraph.GetEdgeWeight(targetIdx, j));
            _connectionStrengths[j] = Math.Max(Math.Abs(toTarget), Math.Abs(fromTarget));
        }

        // Select top features above threshold
        int numToSelect = Math.Min(_maxFeatures, p);
        _selectedIndices = Enumerable.Range(0, p)
            .Where(j => _connectionStrengths[j] >= _edgeThreshold)
            .OrderByDescending(j => _connectionStrengths[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        // If no features pass the threshold, take the top N by strength
        if (_selectedIndices.Length == 0)
        {
            _selectedIndices = Enumerable.Range(0, p)
                .OrderByDescending(j => _connectionStrengths[j])
                .Take(numToSelect)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    /// <summary>
    /// Fits the selector and transforms the data in one step.
    /// </summary>
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc/>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CausalDiscoverySelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    /// <summary>
    /// Gets a boolean mask indicating which input features are selected.
    /// </summary>
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CausalDiscoverySelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc/>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CausalDiscoverySelector has not been fitted.");

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices.Select(i => inputFeatureNames[i]).ToArray();
    }
}
