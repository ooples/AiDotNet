namespace AiDotNet.Enums;

/// <summary>
/// How a triplane field combines the features it samples from its three planes before the decoder MLP.
/// </summary>
/// <remarks>
/// <para>
/// A triplane stores a 3D field as three axis-aligned feature planes (XY, XZ and YZ). A query point is projected onto
/// each plane, each plane is sampled bilinearly, and the three feature vectors are combined into one decoder input.
/// TripoSR's <c>TriplaneNeRFRenderer</c> (Tochilkin et al. 2024) supports both reductions and ships with
/// <see cref="Concat"/>.
/// </para>
/// <para>
/// <b>For Beginners:</b> Each 3D point looks itself up on three flat feature maps. This setting decides whether the
/// three lookups are placed side by side (concatenated) or averaged.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new TripoSROptions { FeatureReduction = TriplaneFeatureReduction.Mean };
/// </code>
/// </example>
public enum TriplaneFeatureReduction
{
    /// <summary>
    /// Concatenate the three plane features plane by plane (XY, XZ, YZ), giving a decoder input of three times the
    /// plane channels. TripoSR's default (<c>feature_reduction: concat</c>).
    /// </summary>
    Concat,

    /// <summary>
    /// Average the three plane features, giving a decoder input of the plane channels.
    /// </summary>
    Mean
}
