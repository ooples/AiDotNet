namespace AiDotNet.Helpers;

/// <summary>
/// Noise tensors drawn from a caller-supplied random number generator, for models whose sampling must be
/// reproducible from a seed.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> diffusion and flow models start from random noise. To get the same output twice from
/// the same seed, the noise has to come from the model's own random number generator, one value after another in a
/// fixed order. This class does that once for every model instead of each model carrying its own copy of the
/// loop.</para>
/// </remarks>
internal static class SeededNoiseTensors
{
    /// <summary>
    /// A tensor of normal draws, filled in row-major order by the Box-Muller cosine branch. Each element takes two
    /// uniforms from <paramref name="random"/> in order and is
    /// <c>scale * sqrt(-2 ln(1 - u1)) * cos(2 pi u2)</c>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="random">The generator the uniforms are taken from.</param>
    /// <param name="scale">The standard deviation; 1 for standard normal values.</param>
    [AiDotNet.Attributes.ScalarKernel("Draws each element from a sequential seeded Random stream; Box-Muller (Log, Sqrt, Cos) dominates, and writing the span directly avoids a staging array and a Cast copy.")]
    internal static Tensor<T> StandardNormal<T>(int[] shape, Random random, double scale = 1.0)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (random is null) throw new ArgumentNullException(nameof(random));

        int count = 1;
        foreach (int dimension in shape)
            count = checked(count * dimension);
        if (count == 0) return new Tensor<T>(shape);

        // Written straight into the result's span (no double[] staging, no Cast copy), as
        // CompiledStepRandom<T>.FillStandardNormal does.
        var result = new Tensor<T>(shape);
        var span = result.AsWritableSpan();
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < count; i++)
        {
            double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
            span[i] = ops.FromDouble(scale * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        return result;
    }
}
