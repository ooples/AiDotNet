namespace AiDotNet.Data.Multimodal;

/// <summary>
/// Identifies the type of data modality in a multimodal sample.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> A "modality" means a type of data. Modern AI models
/// often work with multiple types at once - for example, an image captioning model
/// uses both images and text. This enum identifies which type each piece of data is.
/// </para>
/// </remarks>
public enum ModalityType
{
    /// <summary>Image data (2D pixel grids, typically [H, W, C] or [C, H, W]).</summary>
    Image = 0,

    /// <summary>Text data (token sequences, embeddings, or raw strings).</summary>
    Text = 1,

    /// <summary>Audio data (waveform samples or spectrograms).</summary>
    Audio = 2,

    /// <summary>Video data (sequences of frames, typically [T, H, W, C]).</summary>
    Video = 3,

    /// <summary>Structured/tabular data (feature vectors).</summary>
    Tabular = 4,

    /// <summary>3D point cloud data (sets of 3D coordinates with optional features).</summary>
    PointCloud = 5,

    /// <summary>Time series data (sequential measurements over time).</summary>
    TimeSeries = 6,

    /// <summary>Graph-structured data (nodes and edges).</summary>
    Graph = 7,

    /// <summary>Custom or unspecified modality type.</summary>
    Custom = 99
}
