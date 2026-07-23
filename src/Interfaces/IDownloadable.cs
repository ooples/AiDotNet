namespace AiDotNet.Interfaces;

/// <summary>
/// Defines capability to automatically download and cache datasets.
/// </summary>
/// <remarks>
/// <para>
/// Data loaders that implement this interface can fetch datasets from remote sources
/// and cache them locally, making it easy to use standard benchmark datasets
/// without manual setup.
/// </para>
/// <para><b>For Beginners:</b> Many standard datasets (MNIST, CIFAR, Cora, etc.) are available online.
/// Instead of manually downloading and extracting files, the data loader can do it for you
/// automatically and remember where the files are stored.
/// </para>
/// </remarks>
public interface IDownloadable
{
    /// <summary>
    /// Gets whether the dataset has been downloaded and is available locally.
    /// </summary>
    bool IsDownloaded { get; }

    /// <summary>
    /// Gets the local path where the dataset is cached.
    /// </summary>
    string CachePath { get; }

    /// <summary>
    /// Gets the URLs where the dataset can be downloaded from.
    /// </summary>
    IReadOnlyList<string> DownloadUrls { get; }

    /// <summary>
    /// Downloads the dataset asynchronously if not already cached.
    /// </summary>
    /// <param name="forceRedownload">If true, redownloads even if already cached.</param>
    /// <param name="progress">Optional progress reporter (0.0 to 1.0).</param>
    /// <param name="cancellationToken">Token to cancel the download.</param>
    /// <returns>A task that completes when download is finished.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method:
    /// 1. Checks if the data already exists locally
    /// 2. If not (or forceRedownload is true), downloads from the internet
    /// 3. Extracts and prepares the files
    /// 4. Reports progress so you can show a progress bar
    ///
    /// Example usage:
    /// <code>
    /// await loader.DownloadAsync(progress: new Progress&lt;double&gt;(p =&gt;
    ///     Console.WriteLine($"Download: {p:P0}")));
    /// </code>
    /// </para>
    /// </remarks>
    Task DownloadAsync(
        bool forceRedownload = false,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Deletes the locally cached dataset files.
    /// </summary>
    /// <remarks>
    /// Use this to free disk space or force a fresh download next time.
    /// </remarks>
    void ClearCache();
}
