namespace AiDotNet.Data.Graph;

/// <summary>
/// Thin wrapper around <see cref="MolecularDatasetLoader{T}"/> for the ZINC dataset.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// ZINC contains ~250K drug-like molecules from the ZINC database.
/// Standard benchmark for graph regression on constrained solubility.
/// The 12K subset is commonly used for benchmarking GNNs.
/// </para>
/// </remarks>
public class ZincDataLoader<T> : MolecularDatasetLoader<T>
{
    /// <summary>
    /// Initializes a new ZINC data loader.
    /// </summary>
    /// <param name="options">Optional configuration.</param>
    public ZincDataLoader(ZincDataLoaderOptions? options = null)
        : base(
            MolecularDataset.ZINC,
            dataPath: options?.DataPath,
            autoDownload: options?.AutoDownload ?? true)
    {
    }
}
