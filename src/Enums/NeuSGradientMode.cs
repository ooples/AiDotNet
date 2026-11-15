namespace AiDotNet.Enums;

/// <summary>
/// How a NeuS-style SDF reconstruction computes the spatial gradient <c>∇f</c> it uses for normals, the rendering
/// slope and the Eikonal term (instant-nsr-pl <c>geometry.grad_type</c>).
/// </summary>
public enum NeuSGradientMode
{
    /// <summary>The exact derivative of the SDF network (<c>grad_type: analytic</c>; the NeuS paper).</summary>
    Analytic,

    /// <summary>
    /// Central differences <c>(f(x + εeᵢ) − f(x − εeᵢ)) / 2ε</c> over six extra SDF evaluations
    /// (<c>grad_type: finite_difference</c>; Neuralangelo, Li et al. 2023, Sec. 3.1, which Wonder3D's
    /// reconstruction uses).
    /// </summary>
    FiniteDifference,
}
