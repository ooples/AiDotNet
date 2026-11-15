using AiDotNet.Enums;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// The input of Wonder3D's domain switcher and camera embedding: the <c>[sin(x), cos(x)]</c> encoding of
/// <c>x = (conditioning elevation, elevation, azimuth, normal flag, color flag)</c> that the U-Net's
/// <c>class_embedding</c>, a <c>Linear(10, d) → SiLU → Linear(d, d)</c> MLP, maps to the vector added to the
/// timestep embedding.
/// </summary>
/// <remarks>
/// <para>
/// The released U-Net is configured with <c>class_embed_type = "projection"</c> and
/// <c>projection_class_embeddings_input_dim = 10</c>, and the pipeline's <c>prepare_camera_embedding</c> builds the
/// 10 values with the <c>e_de_da_sincos</c> encoding: <c>torch.cat([sin(e), cos(e)], -1)</c> of the five values
/// above. The one-hot pair is the paper's domain switcher (Sec. 3.2): normal <c>[1, 0]</c>, color <c>[0, 1]</c>.
/// The MLP itself is two dense layers of <see cref="Wonder3DModel{T}"/>, so it trains and serializes with the model.
/// </para>
/// <para><b>For Beginners:</b> This turns "which view, and normal or color?" into numbers the U-Net can read, so one
/// network can draw every view in both domains.</para>
/// </remarks>
internal static class Wonder3DDomainEmbedding
{
    /// <summary>The raw values per view: three camera values and the two domain flags.</summary>
    internal const int RawDim = 5;

    /// <summary>The encoded width, sine and cosine of every raw value.</summary>
    internal const int InputDim = 2 * RawDim;

    /// <summary>
    /// The 10 encoded values of one view: <c>[sin(x), cos(x)]</c> of
    /// <c>x = (elevationCondition, elevation, azimuth, normal flag, color flag)</c>.
    /// </summary>
    /// <param name="elevationCondition">The input view's elevation (radians).</param>
    /// <param name="elevation">The target view's elevation value (radians).</param>
    /// <param name="azimuth">The target view's azimuth value (radians).</param>
    /// <param name="domain">The output domain.</param>
    internal static double[] Encode(double elevationCondition, double elevation, double azimuth, Wonder3DDomain domain)
    {
        var raw = new[]
        {
            elevationCondition, elevation, azimuth,
            domain == Wonder3DDomain.Normal ? 1.0 : 0.0,
            domain == Wonder3DDomain.Color ? 1.0 : 0.0,
        };
        var encoded = new double[InputDim];
        for (int i = 0; i < RawDim; i++)
        {
            encoded[i] = Math.Sin(raw[i]);
            encoded[RawDim + i] = Math.Cos(raw[i]);
        }

        return encoded;
    }
}
