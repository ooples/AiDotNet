namespace AiDotNet.Models.Options;

/// <summary>
/// Specifies the federated learning paradigm to use.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Federated learning comes in two main flavors:</para>
/// <list type="bullet">
/// <item><description><b>Horizontal:</b> Each client has the same features for different samples
/// (e.g., multiple hospitals each have patient records with the same columns).</description></item>
/// <item><description><b>Vertical:</b> Each client has different features for the same samples
/// (e.g., a bank has income data and a hospital has medical data for the same people).</description></item>
/// </list>
/// </remarks>
public enum FederatedLearningMode
{
    /// <summary>
    /// Horizontal (sample-partitioned) federated learning.
    /// Each client has the same model architecture and feature space,
    /// but holds different data samples. This is the traditional FL paradigm.
    /// </summary>
    Horizontal = 0,

    /// <summary>
    /// Vertical (feature-partitioned) federated learning.
    /// Each client holds different features for the same entities.
    /// Requires entity alignment (PSI), split neural networks, and secure gradient exchange.
    /// </summary>
    Vertical = 1
}
