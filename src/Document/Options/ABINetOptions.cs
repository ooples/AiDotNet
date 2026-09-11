using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the ABINet document model.
/// </summary>
public class ABINetOptions : DocumentNeuralNetworkOptions
{
    /// <summary>Initializes a new instance carrying the model's published defaults.</summary>
    /// <remarks>
    /// <para>
    /// ImageWidth, ImageHeight, MaxSequenceLength, VisionDim and VisionLayers are declared by
    /// DocumentNeuralNetworkOptions and were previously left unset, with the values living in
    /// constructor parameters instead. They carry ABINet's published values here (Fang et al.,
    /// CVPR 2021: 32x128 input, 26-character sequences, a 512-wide 3-layer vision branch).
    /// </para>
    /// </remarks>
    public ABINetOptions()
    {
        ImageWidth = 128;
        ImageHeight = 32;
        MaxSequenceLength = 26;
        VisionDim = 512;
        VisionLayers = 3;
        LanguageDim = 512;
        LanguageLayers = 4;
        NumIterations = 3;
    }

    /// <summary>
    /// Gets or sets the width of the language branch. Default: 512.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ABINet reads the image and separately reasons about what the
    /// word is likely to be. This is how wide that language-reasoning part is.</para>
    /// </remarks>
    public int LanguageDim { get; set; }

    /// <summary>
    /// Gets or sets the number of transformer layers in the language branch. Default: 4.
    /// </summary>
    public int LanguageLayers { get; set; }

    /// <summary>
    /// Gets or sets how many times the model refines its reading. Default: 3.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model guesses the text, then re-reads its own guess to
    /// correct it, and repeats. This is how many of those passes it makes.</para>
    /// </remarks>
    public int NumIterations { get; set; }

    /// <summary>
    /// Gets or sets the recognisable character set. Null uses the model's default charset.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The character count is the output width of the recognition head, so this changes the
    /// model's shape and not merely how results are decoded.
    /// </para>
    /// </remarks>
    public string? Charset { get; set; }

    /// <summary>
    /// Gets or sets lambda_v, the weight on the vision model's loss term. Defaults to the
    /// paper's 1.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ABINet (Fang et al., CVPR 2021, arXiv:2103.06495) trains all three branches jointly with
    /// L = lambda_v * L_v + lambda_l * L_l + L_f (Eq. 5), and sets both weights to 1.0.
    /// </para>
    /// <para><b>For Beginners:</b> How much the image-reading part's own mistakes count toward
    /// the total score. Raise it to make the model care more about getting the visual reading
    /// right on its own.</para>
    /// </remarks>
    public double VisionLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets lambda_l, the weight on the language model's loss term. Defaults to the
    /// paper's 1.0.
    /// </summary>
    /// <remarks>
    /// <para>
    /// See <see cref="VisionLossWeight"/> for the objective this participates in.
    /// </para>
    /// <para><b>For Beginners:</b> How much the language-reasoning part's own mistakes count
    /// toward the total score.</para>
    /// </remarks>
    public double LanguageLossWeight { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the optimizer's initial learning rate. Defaults to the paper's 1e-4.
    /// </summary>
    /// <remarks>
    /// <para>
    /// ABINet (Fang et al., CVPR 2021, arXiv:2103.06495 §4.2) trains with ADAM at an initial
    /// learning rate of 1e-4, decayed to 1e-5. The model previously constructed its Adam
    /// optimizer with no options at all, so it silently ran at the optimizer's own 1e-3 default
    /// — 10x the paper's rate, which the multi-task objective's three summed loss terms then
    /// amplify into a rising loss.
    /// </para>
    /// <para><b>For Beginners:</b> How big a step the model takes each time it learns. Too big
    /// and it overshoots and gets worse instead of better.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-4;
}
