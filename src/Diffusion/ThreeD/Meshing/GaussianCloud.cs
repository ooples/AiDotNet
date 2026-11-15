using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Diffusion.ThreeD.Meshing;

/// <summary>
/// A set of 3D Gaussians in activated form (Kerbl et al., "3D Gaussian Splatting", SIGGRAPH 2023): centres, per-axis
/// standard deviations, rotations and opacities.
/// </summary>
/// <typeparam name="T">The numeric type of the parameters.</typeparam>
/// <remarks>
/// <para>
/// Gaussian i has covariance <c>Sigma_i = R_i S_i S_i^T R_i^T</c>, with R_i the rotation of the unit quaternion
/// <c>(w, x, y, z)</c> (normalised here, as the reference code does) and S_i the diagonal of standard deviations.
/// The values are the activated ones: scales after the exponential and opacities after the sigmoid of the reference
/// parameterisation.
/// </para>
/// </remarks>
public sealed class GaussianCloud<T>
{
    /// <summary>
    /// Creates a Gaussian cloud.
    /// </summary>
    /// <param name="positions">Centres, [N, 3].</param>
    /// <param name="scales">Standard deviations along the Gaussian's own axes, [N, 3], all positive.</param>
    /// <param name="rotations">Rotations as quaternions (w, x, y, z), [N, 4]; they need not be unit length.</param>
    /// <param name="opacities">Opacities in [0, 1], [N].</param>
    public GaussianCloud(Tensor<T> positions, Tensor<T> scales, Tensor<T> rotations, Tensor<T> opacities)
    {
        if (positions is null)
            throw new ArgumentNullException(nameof(positions));
        if (scales is null)
            throw new ArgumentNullException(nameof(scales));
        if (rotations is null)
            throw new ArgumentNullException(nameof(rotations));
        if (opacities is null)
            throw new ArgumentNullException(nameof(opacities));
        if (positions.Shape.Length != 2 || positions.Shape[1] != 3)
            throw new ArgumentException("Positions must have shape [N, 3].", nameof(positions));

        int count = positions.Shape[0];
        if (scales.Shape.Length != 2 || scales.Shape[0] != count || scales.Shape[1] != 3)
            throw new ArgumentException("Scales must have shape [N, 3].", nameof(scales));
        if (rotations.Shape.Length != 2 || rotations.Shape[0] != count || rotations.Shape[1] != 4)
            throw new ArgumentException("Rotations must have shape [N, 4].", nameof(rotations));
        if (opacities.Shape.Length != 1 || opacities.Shape[0] != count)
            throw new ArgumentException("Opacities must have shape [N].", nameof(opacities));

        Positions = positions;
        Scales = scales;
        Rotations = rotations;
        Opacities = opacities;
    }

    /// <summary>Centres, [N, 3].</summary>
    public Tensor<T> Positions { get; }

    /// <summary>Standard deviations along each Gaussian's axes, [N, 3].</summary>
    public Tensor<T> Scales { get; }

    /// <summary>Rotation quaternions (w, x, y, z), [N, 4].</summary>
    public Tensor<T> Rotations { get; }

    /// <summary>Opacities in [0, 1], [N].</summary>
    public Tensor<T> Opacities { get; }

    /// <summary>The number of Gaussians.</summary>
    public int Count => Positions.Shape[0];
}
