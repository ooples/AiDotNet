namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// The 3D Gaussians LGM predicts for one object: one Gaussian per pixel of every splatter map.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Each attribute has already been through the paper's activation (Tang et al. 2024, <c>core/models.py</c>):
/// </para>
/// <list type="bullet">
/// <item><description><see cref="Positions"/>: <c>clamp(x, -1, 1)</c>, world-space centres in the [-1, 1]
/// cube.</description></item>
/// <item><description><see cref="Opacities"/>: <c>sigmoid(x)</c>, in (0, 1).</description></item>
/// <item><description><see cref="Scales"/>: <c>0.1 * softplus(x)</c>, positive per-axis standard
/// deviations.</description></item>
/// <item><description><see cref="Rotations"/>: <c>normalize(x)</c>, unit quaternions ordered (w, x, y, z), the
/// order the Gaussian rasterizer reads.</description></item>
/// <item><description><see cref="Colors"/>: <c>0.5 * tanh(x) + 0.5</c>, RGB in (0, 1).</description></item>
/// </list>
/// <para>
/// Gaussian <c>n</c> belongs to view <c>n / (S*S)</c> and splatter-map pixel <c>n % (S*S)</c> (row-major), with
/// <c>S</c> the splatter size.
/// </para>
/// <para><b>For Beginners:</b> a Gaussian is a soft coloured blob in 3D. Tens of thousands of them, drawn together,
/// look like the object.</para>
/// </remarks>
public sealed class LGMGaussians<T>
{
    /// <summary>
    /// Creates a Gaussian set from activated attributes.
    /// </summary>
    /// <param name="positions">Centres, [N, 3].</param>
    /// <param name="opacities">Opacities, [N, 1].</param>
    /// <param name="scales">Per-axis scales, [N, 3].</param>
    /// <param name="rotations">Unit quaternions (w, x, y, z), [N, 4].</param>
    /// <param name="colors">RGB colours, [N, 3].</param>
    public LGMGaussians(Tensor<T> positions, Tensor<T> opacities, Tensor<T> scales, Tensor<T> rotations,
        Tensor<T> colors)
    {
        if (positions is null) throw new ArgumentNullException(nameof(positions));
        if (opacities is null) throw new ArgumentNullException(nameof(opacities));
        if (scales is null) throw new ArgumentNullException(nameof(scales));
        if (rotations is null) throw new ArgumentNullException(nameof(rotations));
        if (colors is null) throw new ArgumentNullException(nameof(colors));
        int count = positions.Rank == 2 ? positions.Shape[0] : -1;
        RequireShape(positions, count, 3, nameof(positions));
        RequireShape(opacities, count, 1, nameof(opacities));
        RequireShape(scales, count, 3, nameof(scales));
        RequireShape(rotations, count, 4, nameof(rotations));
        RequireShape(colors, count, 3, nameof(colors));

        Positions = positions;
        Opacities = opacities;
        Scales = scales;
        Rotations = rotations;
        Colors = colors;
    }

    /// <summary>Number of Gaussians.</summary>
    public int Count => Positions.Shape[0];

    /// <summary>Centres, [N, 3], in [-1, 1].</summary>
    public Tensor<T> Positions { get; }

    /// <summary>Opacities, [N, 1], in (0, 1).</summary>
    public Tensor<T> Opacities { get; }

    /// <summary>Per-axis scales, [N, 3], positive.</summary>
    public Tensor<T> Scales { get; }

    /// <summary>Unit rotation quaternions (w, x, y, z), [N, 4].</summary>
    public Tensor<T> Rotations { get; }

    /// <summary>RGB colours, [N, 3], in (0, 1).</summary>
    public Tensor<T> Colors { get; }

    /// <summary>
    /// The attributes packed as the official model returns them: <c>[N, 14]</c> = position (3), opacity (1),
    /// scale (3), rotation (4), RGB (3).
    /// </summary>
    public Tensor<T> ToTensor()
        => AiDotNetEngine.Current.TensorConcatenate(new[] { Positions, Opacities, Scales, Rotations, Colors }, 1);

    private static void RequireShape(Tensor<T> tensor, int count, int width, string name)
    {
        if (count < 0 || tensor.Rank != 2 || tensor.Shape[0] != count || tensor.Shape[1] != width)
            throw new ArgumentException($"Expected shape [{count}, {width}].", name);
    }
}
