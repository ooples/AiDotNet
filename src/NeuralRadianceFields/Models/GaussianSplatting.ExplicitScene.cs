using AiDotNet.Attributes;
using AiDotNet.Models.Options;

namespace AiDotNet.NeuralRadianceFields.Models;

public partial class GaussianSplatting<T>
{
    /// <summary>
    /// Builds a scene from fully specified Gaussians, such as the ones a feed-forward reconstructor (LGM, GRM,
    /// GS-LRM) predicts, instead of from a point cloud whose shapes are then optimized.
    /// </summary>
    /// <param name="options">Scene options. Densification and learning rates apply only if the scene is trained
    /// further.</param>
    /// <param name="positions">Centres, [N, 3].</param>
    /// <param name="scales">Per-axis standard deviations (linear, not log), [N, 3].</param>
    /// <param name="rotations">Rotation quaternions (w, x, y, z), [N, 4]; normalized when the covariance is
    /// built.</param>
    /// <param name="opacities">Opacities in [0, 1] (activated, not logits), [N] or [N, 1].</param>
    /// <param name="colors">RGB colours in [0, 1], [N, 3]. With spherical harmonics enabled they become the DC
    /// coefficients, exactly as for a point-cloud initialization.</param>
    /// <returns>A scene holding exactly these N Gaussians.</returns>
    /// <remarks>
    /// <para>
    /// The scene stores opacity as a logit (Kerbl et al. 2023), so each opacity is converted with
    /// <c>logit(clamp(p, 1e-6, 1 - 1e-6))</c> and the renderer's sigmoid gives the same opacity back. Each covariance
    /// is <c>R S S^T R^T</c> from the given scale and rotation.
    /// </para>
    /// <para><b>For Beginners:</b> use this when a network has already told you where every blob is, how big, how
    /// turned, how see-through and what colour. The scene can then render images from any camera.</para>
    /// </remarks>
    [ScalarKernel("A scene keeps each Gaussian's attributes in that Gaussian's own small vectors, so assigning N "
        + "explicit Gaussians writes them one Gaussian at a time.")]
    public static GaussianSplatting<T> FromGaussians(
        GaussianSplattingOptions options,
        Tensor<T> positions,
        Tensor<T> scales,
        Tensor<T> rotations,
        Tensor<T> opacities,
        Tensor<T> colors)
    {
        if (options is null) throw new ArgumentNullException(nameof(options));
        if (positions is null) throw new ArgumentNullException(nameof(positions));
        if (scales is null) throw new ArgumentNullException(nameof(scales));
        if (rotations is null) throw new ArgumentNullException(nameof(rotations));
        if (opacities is null) throw new ArgumentNullException(nameof(opacities));
        if (colors is null) throw new ArgumentNullException(nameof(colors));
        if (positions.Rank != 2 || positions.Shape[1] != 3)
            throw new ArgumentException("Positions must have shape [N, 3].", nameof(positions));

        int count = positions.Shape[0];
        RequireRows(scales, count, 3, nameof(scales));
        RequireRows(rotations, count, 4, nameof(rotations));
        RequireRows(colors, count, 3, nameof(colors));
        if (opacities.Length != count || !(opacities.Rank == 1 || (opacities.Rank == 2 && opacities.Shape[1] == 1)))
            throw new ArgumentException($"Opacities must have shape [{count}] or [{count}, 1].", nameof(opacities));
        if (count > options.MaxGaussians)
            throw new ArgumentException(
                $"{count} Gaussians exceed options.MaxGaussians ({options.MaxGaussians}).", nameof(positions));

        // The point-cloud constructor places the centres and colours (including the SH DC mapping); the explicit
        // shape, orientation and opacity then replace its neighbour-distance initialization.
        var scene = new GaussianSplatting<T>(options, positions.ToMatrix(), colors.ToMatrix());
        if (scene._gaussians.Count != count)
            throw new InvalidOperationException(
                $"The scene initialized {scene._gaussians.Count} Gaussians from {count} positions.");

        var numOps = MathHelper.GetNumericOperations<T>();
        var scaleData = scales.Contiguous().ToArray();
        var rotationData = rotations.Contiguous().ToArray();
        var opacityData = opacities.Contiguous().ToArray();
        for (int i = 0; i < count; i++)
        {
            var gaussian = scene._gaussians[i];
            for (int k = 0; k < 3; k++) gaussian.Scale[k] = scaleData[i * 3 + k];
            for (int k = 0; k < 4; k++) gaussian.Rotation[k] = rotationData[i * 4 + k];
            gaussian.Opacity = numOps.FromDouble(Logit(numOps.ToDouble(opacityData[i])));
            scene.ComputeCovariance(gaussian);
        }

        scene.MarkSpatialIndexDirty();
        return scene;
    }

    private static void RequireRows(Tensor<T> tensor, int count, int width, string name)
    {
        if (tensor.Rank != 2 || tensor.Shape[0] != count || tensor.Shape[1] != width)
            throw new ArgumentException($"Expected shape [{count}, {width}].", name);
    }
}
