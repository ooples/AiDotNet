using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the HTMNetwork.
/// </summary>
public class HTMNetworkOptions : ModelHyperparameterOptions
{

    /// <summary>
    /// Gets or sets cells per column. Default: <c>32</c>.
    /// </summary>
    public int CellsPerColumn { get; set; } = 32;

    /// <summary>
    /// Gets or sets column count. Default: <c>2048</c>.
    /// </summary>
    public int ColumnCount { get; set; } = 2048;

    /// <summary>
    /// Gets or sets sparsity threshold. Default: <c>0.02</c>.
    /// </summary>
    public double SparsityThreshold { get; set; } = 0.02;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(CellsPerColumn, nameof(CellsPerColumn));
        Require(ColumnCount, nameof(ColumnCount));
        if (SparsityThreshold < 0.0 || SparsityThreshold > 1.0)
        {
            throw new ArgumentException(
                $"{GetType().Name}.SparsityThreshold is {SparsityThreshold}, but it must be between 0.0 and 1.0.",
                OptionsParameterName);
        }
    }
}
