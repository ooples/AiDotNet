namespace AiDotNet.Data.Multimodal;

/// <summary>
/// Represents a single modality's data within a multimodal sample.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// Each ModalitySample wraps a tensor of data along with metadata about what modality
/// it represents and an optional key for identification.
/// </para>
/// <para><b>For Beginners:</b> Think of this as a labeled container.
/// It holds data (a tensor) plus a tag saying what kind of data it is (image, text, etc.).
/// </para>
/// </remarks>
public class ModalitySample<T>
{
    /// <summary>
    /// Gets the modality type of this sample.
    /// </summary>
    public ModalityType Modality { get; }

    /// <summary>
    /// Gets the data tensor for this modality.
    /// </summary>
    public Tensor<T> Data { get; }

    /// <summary>
    /// Gets an optional key to identify this modality within a multimodal sample.
    /// For example, "input_image", "caption_tokens", "audio_spectrogram".
    /// </summary>
    public string Key { get; }

    /// <summary>
    /// Gets optional metadata associated with this modality sample.
    /// </summary>
    public IReadOnlyDictionary<string, object> Metadata { get; }

    /// <summary>
    /// Creates a new modality sample.
    /// </summary>
    /// <param name="modality">The type of modality.</param>
    /// <param name="data">The data tensor.</param>
    /// <param name="key">Optional key for identification. Defaults to the modality name.</param>
    /// <param name="metadata">Optional metadata dictionary.</param>
    public ModalitySample(
        ModalityType modality,
        Tensor<T> data,
        string? key = null,
        IReadOnlyDictionary<string, object>? metadata = null)
    {
        Modality = modality;
        Guard.NotNull(data);
        Data = data;
        Key = key ?? modality.ToString().ToLowerInvariant();
        Metadata = metadata ?? new Dictionary<string, object>();
    }

    /// <summary>
    /// Gets the shape of the data tensor.
    /// </summary>
    public int[] Shape => Data.Shape;

    /// <summary>
    /// Gets the total number of elements in the data tensor.
    /// </summary>
    public int ElementCount
    {
        get
        {
            int count = 1;
            for (int i = 0; i < Data.Shape.Length; i++)
            {
                count *= Data.Shape[i];
            }
            return count;
        }
    }
}
