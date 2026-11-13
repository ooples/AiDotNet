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

    /// <summary>
    /// Gets or sets a learning rate that overrides the paper's training recipe.
    /// Default: <c>null</c> (the paper's rate, 5e-5).
    /// </summary>
    /// <remarks>
    /// <para>
    /// DocBank trains with the recipe Li et al. (2020, Sec. 4) report: AdamW at an initial
    /// learning rate of 5e-5. Setting this replaces that rate, and the recipe's learning-rate
    /// schedule is built from the replacement, so the whole schedule follows it. Leaving it
    /// <c>null</c> keeps the paper's rate. Supplying an optimizer explicitly bypasses both.
    /// </para>
    /// <para><b>For Beginners:</b> leave this empty to train the way the paper did; set it only
    /// when you want the model to take larger or smaller steps than the paper's.</para>
    /// </remarks>
    public double? LearningRate { get; set; }

    /// <summary>
    /// Gets or sets the configuration of the LayoutLM token classifier DocBank's baselines use (Li et al. 2020,
    /// Sec. 4). Default: <c>null</c> (LayoutLM-base). Its class count is always DocBank's twelve token labels.
    /// </summary>
    public LayoutLMOptions? TokenClassifierOptions { get; set; }
}
