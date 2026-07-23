namespace AiDotNet.Data.Graph;

/// <summary>
/// Thin wrapper around <see cref="MolecularDatasetLoader{T}"/> for the QM9 dataset.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// QM9 contains ~134K small organic molecules (up to 9 heavy atoms: C, H, N, O, F)
/// with 19 quantum mechanical properties computed with DFT.
/// Standard benchmark for molecular property prediction.
/// </para>
/// </remarks>
public class Qm9DataLoader<T> : MolecularDatasetLoader<T>
{
    /// <summary>
    /// Initializes a new QM9 data loader.
    /// </summary>
    /// <param name="options">Optional configuration.</param>
    public Qm9DataLoader(Qm9DataLoaderOptions? options = null)
        : base(
            MolecularDataset.QM9,
            dataPath: options?.DataPath,
            autoDownload: options?.AutoDownload ?? true)
    {
    }
}
