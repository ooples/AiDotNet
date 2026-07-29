using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the ABINet document model.
/// </summary>
public class ABINetOptions : DocumentNeuralNetworkOptions
{
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
