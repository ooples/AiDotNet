using AiDotNet.CausalDiscovery;

namespace AiDotNet.Tests.Helpers;

/// <summary>
/// Lets the double-based causal-discovery invariant scaffold exercise an algorithm at FP32.
/// Only the test boundary is converted; the wrapped algorithm and all of its optimization state
/// remain <see cref="float"/> throughout discovery.
/// </summary>
internal sealed class FloatCausalDiscoveryAdapter : ICausalDiscoveryAlgorithm<double>
{
    private readonly ICausalDiscoveryAlgorithm<float> _inner;

    public FloatCausalDiscoveryAdapter(ICausalDiscoveryAlgorithm<float> inner)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    }

    public string Name => _inner.Name;
    public AiDotNet.Enums.CausalDiscoveryCategory Category => _inner.Category;
    public bool SupportsLatentConfounders => _inner.SupportsLatentConfounders;
    public bool SupportsTimeSeries => _inner.SupportsTimeSeries;
    public bool SupportsNonlinear => _inner.SupportsNonlinear;
    public bool SupportsMixedData => _inner.SupportsMixedData;

    public CausalGraph<double> DiscoverStructure(Matrix<double> data, string[]? featureNames = null)
        => ConvertGraph(_inner.DiscoverStructure(ConvertMatrix(data), featureNames));

    public CausalGraph<double> DiscoverStructure(
        Matrix<double> data,
        Vector<double> target,
        string[]? featureNames = null)
        => ConvertGraph(_inner.DiscoverStructure(ConvertMatrix(data), ConvertVector(target), featureNames));

    private static Matrix<float> ConvertMatrix(Matrix<double> source)
    {
        var result = new Matrix<float>(source.Rows, source.Columns);
        for (int row = 0; row < source.Rows; row++)
            for (int column = 0; column < source.Columns; column++)
                result[row, column] = (float)source[row, column];
        return result;
    }

    private static Vector<float> ConvertVector(Vector<double> source)
    {
        var result = new Vector<float>(source.Length);
        for (int index = 0; index < source.Length; index++)
            result[index] = (float)source[index];
        return result;
    }

    private static CausalGraph<double> ConvertGraph(CausalGraph<float> source)
    {
        var adjacency = new Matrix<double>(source.AdjacencyMatrix.Rows, source.AdjacencyMatrix.Columns);
        for (int row = 0; row < adjacency.Rows; row++)
            for (int column = 0; column < adjacency.Columns; column++)
                adjacency[row, column] = source.AdjacencyMatrix[row, column];
        return new CausalGraph<double>(adjacency, source.FeatureNames);
    }
}
