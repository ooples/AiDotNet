namespace AiDotNet.Enums;

/// <summary>
/// The distance a diffusion model that predicts the clean sample x_0 trains with.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> After the model guesses the clean series, training measures how far off the guess is.
/// L1 adds up the absolute errors (robust to occasional large mistakes); L2 adds up the squared errors.</para>
/// </remarks>
public enum DiffusionReconstructionLoss
{
    /// <summary>Mean absolute error (the Diffusion-TS reference default).</summary>
    L1,

    /// <summary>Mean squared error.</summary>
    L2
}
