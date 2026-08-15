using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the FlowDiffuser model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure the FlowDiffuser model for diffusion-based optical flow estimation.
/// Default values follow the original paper recommendations.</para>
/// </remarks>
public class FlowDiffuserOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Gets or sets an optional Adam learning rate for native training.
    /// </summary>
    /// <remarks>
    /// When <see langword="null"/>, FlowDiffuser retains the framework's default optimizer behavior.
    /// </remarks>
    public double? LearningRate { get; set; }
}
