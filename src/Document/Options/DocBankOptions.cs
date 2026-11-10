using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the DocBank document model.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> DocBank labels the regions of a scanned page — which part is a heading,
/// a paragraph, a table, a figure and so on. The values here are the ones it ships with.
/// </para>
/// </remarks>
public class DocBankOptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    /// <remarks>
    /// <para>
    /// ImageSize, BackboneChannels, NumClasses and HiddenDim are declared by
    /// DocumentNeuralNetworkOptions and were previously left unset, with the values living in
    /// constructor parameters instead. DocBank (Li et al. 2020) labels 13 region types on a
    /// 1024-pixel page.
    /// </para>
    /// </remarks>
    public DocBankOptions()
    {
        ImageSize = 1024;
        BackboneChannels = 256;
        NumClasses = 13;
        HiddenDim = 256;
    }

    /// <summary>
    /// Gets or sets whether the model consumes per-token text features alongside the image.
    /// Default: false.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> DocBank pages come with the words already extracted. Turning
    /// this on lets the model read that text as well as look at the picture.</para>
    /// </remarks>
    public bool UseTextFeatures { get; set; }
}
