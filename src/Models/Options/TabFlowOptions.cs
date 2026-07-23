namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for TabFlow, a flow matching model for generating synthetic tabular data
/// using continuous normalizing flows with optimal transport paths.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// TabFlow uses flow matching (a continuous normalizing flow approach) to learn a deterministic
/// mapping from noise to data. Unlike diffusion models that use stochastic differential equations,
/// TabFlow uses ordinary differential equations (ODEs) with optimal transport conditional paths.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabFlow learns to gradually transform random noise into realistic data
/// through a smooth, deterministic path (no randomness during generation):
///
/// 1. <b>Training</b>: Learn a velocity field that moves noise toward data along straight paths
/// 2. <b>Generation</b>: Start from noise and follow the velocity field using an ODE solver
///
/// Advantages over diffusion models:
/// - Faster generation (fewer steps needed since paths are straighter)
/// - Deterministic sampling (same noise → same output)
/// - Often higher quality for tabular data
///
/// Example:
/// <code>
/// var options = new TabFlowOptions&lt;double&gt;
/// {
///     MLPDimensions = new[] { 256, 256, 256 },
///     NumSteps = 100,
///     Epochs = 500
/// };
/// var tabflow = new TabFlowGenerator&lt;double&gt;(options);
/// </code>
/// </para>
/// <para>
/// Reference: "Flow Matching for Tabular Data" (2024)
/// </para>
/// </remarks>
public class TabFlowOptions<T> : RiskModelOptions<T>
{
    /// <summary>
    /// Gets or sets the hidden layer sizes for the velocity field MLP.
    /// </summary>
    /// <value>Array of layer dimensions. Defaults to [256, 256, 256].</value>
    public int[] MLPDimensions { get; set; } = [256, 256, 256];

    /// <summary>
    /// Gets or sets the number of ODE solver steps for generation.
    /// </summary>
    /// <value>Number of Euler/RK4 steps, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> More steps = more accurate but slower generation.
    /// Flow matching typically needs fewer steps than diffusion (100 vs 1000)
    /// because the paths are straighter. Try 50-200.
    /// </para>
    /// </remarks>
    public int NumSteps { get; set; } = 100;

    /// <summary>
    /// Gets or sets the training batch size.
    /// </summary>
    /// <value>The batch size, defaulting to 1024.</value>
    public int BatchSize { get; set; } = 1024;

    /// <summary>
    /// Gets or sets the number of training epochs.
    /// </summary>
    /// <value>The number of epochs, defaulting to 500.</value>
    public int Epochs { get; set; } = 500;

    /// <summary>
    /// Gets or sets the learning rate.
    /// </summary>
    /// <value>The learning rate, defaulting to 1e-3.</value>
    public double LearningRate { get; set; } = 1e-3;

    /// <summary>
    /// Gets or sets the dropout rate for the MLP.
    /// </summary>
    /// <value>The dropout rate, defaulting to 0.0.</value>
    public double DropoutRate { get; set; } = 0.0;

    /// <summary>
    /// Gets or sets the dimension of the time embedding.
    /// </summary>
    /// <value>The time embedding dimension, defaulting to 64.</value>
    public int TimeEmbeddingDimension { get; set; } = 64;

    /// <summary>
    /// Gets or sets the ODE solver type.
    /// </summary>
    /// <value>"euler" or "rk4". Defaults to "euler".</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The ODE solver determines how accurately we follow the velocity field:
    /// - <b>"euler"</b>: Simple, fast, good enough with 100+ steps
    /// - <b>"rk4"</b>: More accurate, can use fewer steps (20-50) for same quality
    /// </para>
    /// </remarks>
    public string Solver { get; set; } = "euler";

    /// <summary>
    /// Gets or sets the sigma for the optimal transport conditional flow.
    /// </summary>
    /// <value>The noise scale, defaulting to 0.001.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> A small amount of noise added to make the flow smoother.
    /// Smaller values = straighter paths but potentially harder to train. Default is fine.
    /// </para>
    /// </remarks>
    public double Sigma { get; set; } = 0.001;

    /// <summary>
    /// Gets or sets the number of VGM modes for continuous columns.
    /// </summary>
    /// <value>Number of mixture modes, defaulting to 10.</value>
    public int VGMModes { get; set; } = 10;

}
