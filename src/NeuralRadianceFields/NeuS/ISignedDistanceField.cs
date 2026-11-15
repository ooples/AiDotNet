namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// A signed distance field: negative inside the surface, positive outside, zero on it (the NeuS convention).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Mesh extraction needs only <see cref="SampleGrid"/>: marching cubes at iso-level 0 over the returned grid gives the
/// surface, with grid index <c>[i, j, k]</c> at the world position <see cref="SdfGridBounds.GridPoints"/> lists.
/// </para>
/// <para><b>For Beginners:</b> For any point in space the field says how far the nearest surface is, with a minus
/// sign when the point is inside the object. The surface is where the value is exactly zero.</para>
/// </remarks>
internal interface ISignedDistanceField<T>
{
    /// <summary>The signed distance at each point.</summary>
    /// <param name="points">World positions, <c>[N, 3]</c>.</param>
    /// <returns>The signed distances, <c>[N]</c>.</returns>
    Tensor<T> QuerySdf(Tensor<T> points);

    /// <summary>
    /// The signed distance on a dense grid of <paramref name="resolution"/>³ samples spanning <paramref name="bounds"/>.
    /// </summary>
    /// <param name="resolution">Samples per axis (at least 2).</param>
    /// <param name="bounds">The sampled box.</param>
    /// <returns>The grid, <c>[R, R, R]</c>, indexed <c>[x, y, z]</c>.</returns>
    Tensor<T> SampleGrid(int resolution, SdfGridBounds bounds);
}
