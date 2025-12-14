namespace AiDotNet.Interfaces;

/// <summary>
/// Base interface for all data loaders providing common data loading capabilities.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// IDataLoader defines the foundation for all specialized data loaders in the system.
/// It provides:
/// - Basic metadata (name, description)
/// - Load/unload lifecycle management
/// - Reset capability for multi-epoch training
/// - Progress tracking through ICountable
/// </para>
/// <para><b>For Beginners:</b> Think of IDataLoader as the foundation that all data loaders build upon.
///
/// Just like all vehicles (cars, trucks, motorcycles) share common features (wheels, engine),
/// all data loaders share these common features:
/// - A name and description so you know what data it loads
/// - The ability to load and unload data
/// - The ability to track how much data there is and where you are in processing it
///
/// Specific types of data loaders (for images, graphs, text, etc.) add their own
/// specialized features on top of this foundation.
/// </para>
/// </remarks>
public interface IDataLoader<T> : IResettable, ICountable
{
    /// <summary>
    /// Gets the human-readable name of this data loader.
    /// </summary>
    /// <remarks>
    /// Examples: "MNIST", "Cora Citation Network", "IMDB Reviews"
    /// </remarks>
    string Name { get; }

    /// <summary>
    /// Gets a description of the dataset and its intended use.
    /// </summary>
    string Description { get; }

    /// <summary>
    /// Gets whether the data has been loaded and is ready for iteration.
    /// </summary>
    bool IsLoaded { get; }

    /// <summary>
    /// Loads the data asynchronously, preparing it for iteration.
    /// </summary>
    /// <param name="cancellationToken">Token to cancel the loading operation.</param>
    /// <returns>A task that completes when loading is finished.</returns>
    /// <remarks>
    /// <para>
    /// This method should be called before attempting to iterate through data.
    /// It may perform operations like:
    /// - Reading files from disk
    /// - Downloading data if implementing IDownloadable
    /// - Parsing and preprocessing data
    /// - Building indices for efficient access
    /// </para>
    /// <para><b>For Beginners:</b> Call this once at the start to prepare your data.
    /// It's async so your program stays responsive while loading large datasets.
    /// </para>
    /// </remarks>
    Task LoadAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Unloads the data and releases associated resources.
    /// </summary>
    /// <remarks>
    /// Call this when you're done with the dataset to free memory.
    /// The loader can be loaded again by calling LoadAsync().
    /// </remarks>
    void Unload();
}
