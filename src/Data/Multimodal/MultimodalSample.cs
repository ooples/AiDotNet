namespace AiDotNet.Data.Multimodal;

/// <summary>
/// A collection of modality samples representing one multimodal data point.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// A MultimodalSample groups multiple modalities together as a single training example.
/// For instance, an image-text pair for captioning would contain an Image modality
/// and a Text modality.
/// </para>
/// <para><b>For Beginners:</b> This is like a folder containing different types of data
/// that all describe the same thing. For example:
/// <code>
/// // Image captioning: one image + one text description
/// var sample = new MultimodalSample&lt;float&gt;(
///     new ModalitySample&lt;float&gt;(ModalityType.Image, imageTensor, "photo"),
///     new ModalitySample&lt;float&gt;(ModalityType.Text, captionTensor, "caption")
/// );
///
/// // Access by key
/// Tensor&lt;float&gt; image = sample["photo"].Data;
/// Tensor&lt;float&gt; caption = sample["caption"].Data;
/// </code>
/// </para>
/// </remarks>
public class MultimodalSample<T>
{
    private readonly List<ModalitySample<T>> _modalities;
    private readonly Dictionary<string, int> _keyIndex;

    /// <summary>
    /// Gets the modality samples in this multimodal sample.
    /// </summary>
    public IReadOnlyList<ModalitySample<T>> Modalities => _modalities;

    /// <summary>
    /// Gets the number of modalities in this sample.
    /// </summary>
    public int ModalityCount => _modalities.Count;

    /// <summary>
    /// Gets the available keys for accessing modalities.
    /// </summary>
    public IReadOnlyCollection<string> Keys => _keyIndex.Keys;

    /// <summary>
    /// Gets optional label information for this sample.
    /// </summary>
    public Tensor<T>? Label { get; set; }

    /// <summary>
    /// Gets or sets an optional string label (for classification).
    /// </summary>
    public string? LabelName { get; set; }

    /// <summary>
    /// Gets or sets an optional integer class index.
    /// </summary>
    public int ClassIndex { get; set; } = -1;

    /// <summary>
    /// Creates a new multimodal sample from a collection of modality samples.
    /// </summary>
    /// <param name="modalities">The modality samples to include.</param>
    public MultimodalSample(params ModalitySample<T>[] modalities)
    {
        if (modalities is null || modalities.Length == 0)
            throw new ArgumentException("At least one modality sample is required.", nameof(modalities));

        _modalities = new List<ModalitySample<T>>(modalities);
        _keyIndex = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        for (int i = 0; i < modalities.Length; i++)
        {
            if (modalities[i] is null)
            {
                throw new ArgumentException($"Modality at index {i} is null.", nameof(modalities));
            }

            string key = modalities[i].Key;
            if (_keyIndex.ContainsKey(key))
            {
                throw new ArgumentException(
                    $"Duplicate modality key '{key}'. Each modality must have a unique key.",
                    nameof(modalities));
            }

            _keyIndex[key] = i;
        }
    }

    /// <summary>
    /// Creates a new multimodal sample from a list of modality samples.
    /// </summary>
    /// <param name="modalities">The modality samples to include.</param>
    public MultimodalSample(IEnumerable<ModalitySample<T>> modalities)
        : this(modalities?.ToArray() ?? Array.Empty<ModalitySample<T>>())
    {
    }

    /// <summary>
    /// Gets a modality sample by its key.
    /// </summary>
    /// <param name="key">The modality key.</param>
    /// <returns>The modality sample.</returns>
    public ModalitySample<T> this[string key]
    {
        get
        {
            if (!_keyIndex.TryGetValue(key, out int index))
            {
                throw new KeyNotFoundException(
                    $"Modality key '{key}' not found. Available keys: {string.Join(", ", _keyIndex.Keys)}");
            }

            return _modalities[index];
        }
    }

    /// <summary>
    /// Tries to get a modality sample by key.
    /// </summary>
    /// <param name="key">The key to look up.</param>
    /// <param name="sample">The found sample, or null.</param>
    /// <returns>True if the key was found.</returns>
    public bool TryGetModality(string key, out ModalitySample<T>? sample)
    {
        if (_keyIndex.TryGetValue(key, out int index))
        {
            sample = _modalities[index];
            return true;
        }

        sample = null;
        return false;
    }

    /// <summary>
    /// Gets all modality samples of a specific type.
    /// </summary>
    /// <param name="modality">The modality type to filter by.</param>
    /// <returns>All matching modality samples.</returns>
    public IEnumerable<ModalitySample<T>> GetByType(ModalityType modality)
    {
        return _modalities.Where(m => m.Modality == modality);
    }

    /// <summary>
    /// Checks whether a modality key exists in this sample.
    /// </summary>
    /// <param name="key">The key to check.</param>
    /// <returns>True if the key exists.</returns>
    public bool HasKey(string key)
    {
        return _keyIndex.ContainsKey(key);
    }

    /// <summary>
    /// Checks whether this sample contains any modality of the specified type.
    /// </summary>
    /// <param name="modality">The modality type to check.</param>
    /// <returns>True if at least one modality of the specified type exists.</returns>
    public bool HasModality(ModalityType modality)
    {
        return _modalities.Any(m => m.Modality == modality);
    }
}
