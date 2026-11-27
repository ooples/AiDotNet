namespace AiDotNet.Helpers;

/// <summary>
/// How a Box-Muller draw keeps its first uniform away from zero, where the logarithm is undefined.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> the Box-Muller method turns two random numbers between 0 and 1 into a bell-curve
/// random number, and it takes the logarithm of the first one. The logarithm of 0 is not a number, so the first
/// value has to be nudged away from 0. Models do this in a few slightly different ways; this setting names which
/// one, so that a seeded model keeps producing exactly the same numbers.</para>
/// <para>The floored forms take their floor from the caller (<c>floor</c>, 1e-10 unless given): models floor at
/// 1e-10, 1e-12, 1e-15 or <see cref="double.Epsilon"/>, and the floor sets how far the tail reaches
/// (<c>sqrt(-2 ln floor)</c>: about 6.8 sigma at 1e-10, 7.4 at 1e-12, 8.3 at 1e-15, 38.6 at double.Epsilon).</para>
/// </remarks>
internal enum BoxMullerGuard
{
    /// <summary>Use <c>1 - u</c>, which lies in (0, 1].</summary>
    Complement,

    /// <summary>Use <c>max(1 - u, floor)</c>.</summary>
    FlooredComplement,

    /// <summary>Use <c>max(floor, u)</c>.</summary>
    Floored,
}

/// <summary>
/// Normal noise drawn from a caller-supplied random number generator, for models whose sampling must be
/// reproducible from a seed. The one definition of the seeded Box-Muller draw: every model, generator and compiled
/// training step that fills a tensor, vector or span with seeded normal values goes through here.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> diffusion, flow and generative models start from random noise. To get the same output
/// twice from the same seed, the noise has to come from the model's own random number generator, one value after
/// another in a fixed order. This class does that once for every model instead of each model carrying its own copy
/// of the loop.</para>
/// <para>The draws stay a host loop over <see cref="Random.NextDouble"/>: a seeded run consumes the generator in
/// element order, and no engine operation draws from a <see cref="Random"/>. Each value is converted once, straight
/// into the destination, with no staging array and no cast copy.</para>
/// </remarks>
internal static class SeededNoiseTensors
{
    /// <summary>
    /// Writes normal draws into <paramref name="destination"/> by the Box-Muller step (the cosine of <c>u2</c> unless <c>branch</c> and <c>secondUniform</c> say otherwise), in order. Each element
    /// takes two uniforms <c>(u, u2)</c> from <paramref name="random"/> and is
    /// <c>scale * sqrt(-2 ln u1) * cos(2 pi u2)</c>, with <c>u1</c> derived from <c>u</c> by <paramref name="guard"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the destination.</typeparam>
    /// <param name="destination">The elements to overwrite.</param>
    /// <param name="random">The generator the uniforms are taken from.</param>
    /// <param name="scale">The standard deviation; 1 for standard normal values.</param>
    /// <param name="guard">How the first uniform is kept away from zero.</param>
    /// <param name="floor">The lower bound the floored guards apply; ignored by <see cref="BoxMullerGuard.Complement"/>.</param>
    /// <param name="branch">Which Box-Muller output each draw takes.</param>
    /// <param name="secondUniform">Whether the angle's uniform is the draw or its complement.</param>
    /// <param name="mean">Added to every draw (<c>mean + scale * z</c>); 0 leaves the draws unshifted.</param>
    [AiDotNet.Attributes.ScalarKernel("Draws each element from a sequential seeded Random stream; Box-Muller (Log, Sqrt, Cos) dominates, and writing the span directly avoids a staging array and a Cast copy.")]
    internal static void WriteStandardNormal<T>(Span<T> destination, Random random, double scale = 1.0,
        BoxMullerGuard guard = BoxMullerGuard.Complement, double mean = 0.0, double floor = DefaultFloor,
        BoxMullerBranch branch = BoxMullerBranch.Cosine,
        BoxMullerSecondUniform secondUniform = BoxMullerSecondUniform.Raw)
    {
        if (random is null) throw new ArgumentNullException(nameof(random));
        ValidateFloor(floor);
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < destination.Length; i++)
        {
            double value = Draw(random, scale, guard, floor, branch, secondUniform);
            destination[i] = ops.FromDouble(mean == 0.0 ? value : mean + value);
        }
    }

    /// <summary>
    /// One standard normal draw by the Box-Muller step (the cosine of <c>u2</c> unless <c>branch</c> and <c>secondUniform</c> say otherwise): two uniforms from <paramref name="random"/>, in order,
    /// and <c>sqrt(-2 ln u1) * cos(2 pi u2)</c>.
    /// </summary>
    /// <param name="random">The generator the uniforms are taken from.</param>
    /// <param name="guard">How the first uniform is kept away from zero.</param>
    /// <param name="floor">The lower bound the floored guards apply; ignored by <see cref="BoxMullerGuard.Complement"/>.</param>
    /// <param name="branch">Which Box-Muller output each draw takes.</param>
    /// <param name="secondUniform">Whether the angle's uniform is the draw or its complement.</param>
    internal static double NextStandardNormal(Random random, BoxMullerGuard guard = BoxMullerGuard.Complement,
        double floor = DefaultFloor, BoxMullerBranch branch = BoxMullerBranch.Cosine,
        BoxMullerSecondUniform secondUniform = BoxMullerSecondUniform.Raw)
    {
        if (random is null) throw new ArgumentNullException(nameof(random));
        ValidateFloor(floor);
        return Draw(random, 1.0, guard, floor, branch, secondUniform);
    }

    /// <summary>The floor the floored guards use when the caller gives none.</summary>
    internal const double DefaultFloor = 1e-10;

    private static void ValidateFloor(double floor)
    {
        if (!(floor > 0.0) || floor >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(floor), "The floor must lie in (0, 1).");
    }

    // The one Box-Muller draw: two uniforms, in order, the first guarded away from zero. The scale multiplies the radius
    // before the angle term, as (scale * sqrt(-2 ln u1)) * cos(2 pi u2) in the loops this replaced; scaling the product
    // instead rounds differently in the last bit.
    private static double Draw(Random random, double scale, BoxMullerGuard guard, double floor, BoxMullerBranch branch,
        BoxMullerSecondUniform secondUniform)
    {
        double first = random.NextDouble();
        double second = random.NextDouble();
        double u2 = secondUniform == BoxMullerSecondUniform.OneMinus ? 1.0 - second : second;
        double u1 = guard switch
        {
            BoxMullerGuard.Complement => 1.0 - first,
            BoxMullerGuard.FlooredComplement => Math.Max(1.0 - first, floor),
            _ => Math.Max(floor, first),
        };
        double radius = scale * Math.Sqrt(-2.0 * Math.Log(u1));
        return branch == BoxMullerBranch.Sine
            ? radius * Math.Sin(2.0 * Math.PI * u2)
            : radius * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>
    /// A tensor of normal draws, filled in row-major order; see <see cref="WriteStandardNormal{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="random">The generator the uniforms are taken from.</param>
    /// <param name="scale">The standard deviation; 1 for standard normal values.</param>
    /// <param name="guard">How the first uniform is kept away from zero.</param>
    /// <param name="floor">The lower bound the floored guards apply; ignored by <see cref="BoxMullerGuard.Complement"/>.</param>
    /// <param name="branch">Which Box-Muller output each draw takes.</param>
    /// <param name="secondUniform">Whether the angle's uniform is the draw or its complement.</param>
    internal static Tensor<T> StandardNormal<T>(int[] shape, Random random, double scale = 1.0,
        BoxMullerGuard guard = BoxMullerGuard.Complement, double floor = DefaultFloor,
        BoxMullerBranch branch = BoxMullerBranch.Cosine,
        BoxMullerSecondUniform secondUniform = BoxMullerSecondUniform.Raw)
    {
        int count = ElementCount(shape);
        if (random is null) throw new ArgumentNullException(nameof(random));
        var result = new Tensor<T>(shape);
        if (count != 0) WriteStandardNormal(result.AsWritableSpan(), random, scale, guard, 0.0, floor, branch, secondUniform);
        return result;
    }

    /// <summary>
    /// A vector of <paramref name="length"/> normal draws; see <see cref="WriteStandardNormal{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="length">The number of draws.</param>
    /// <param name="random">The generator the uniforms are taken from.</param>
    /// <param name="scale">The standard deviation; 1 for standard normal values.</param>
    /// <param name="guard">How the first uniform is kept away from zero.</param>
    /// <param name="floor">The lower bound the floored guards apply; ignored by <see cref="BoxMullerGuard.Complement"/>.</param>
    /// <param name="branch">Which Box-Muller output each draw takes.</param>
    /// <param name="secondUniform">Whether the angle's uniform is the draw or its complement.</param>
    internal static Vector<T> StandardNormalVector<T>(int length, Random random, double scale = 1.0,
        BoxMullerGuard guard = BoxMullerGuard.Complement, double floor = DefaultFloor,
        BoxMullerBranch branch = BoxMullerBranch.Cosine,
        BoxMullerSecondUniform secondUniform = BoxMullerSecondUniform.Raw)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (random is null) throw new ArgumentNullException(nameof(random));
        var values = new Vector<T>(length);
        if (length != 0) WriteStandardNormal(values.AsWritableSpan(), random, scale, guard, 0.0, floor, branch, secondUniform);
        return values;
    }

    /// <summary>
    /// Overwrites every element of <paramref name="target"/> with a normal draw, in row-major order; see
    /// <see cref="WriteStandardNormal{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="target">The tensor to overwrite.</param>
    /// <param name="random">The generator the uniforms are taken from.</param>
    /// <param name="scale">The standard deviation; 1 for standard normal values.</param>
    /// <param name="guard">How the first uniform is kept away from zero.</param>
    /// <param name="floor">The lower bound the floored guards apply; ignored by <see cref="BoxMullerGuard.Complement"/>.</param>
    /// <param name="branch">Which Box-Muller output each draw takes.</param>
    /// <param name="secondUniform">Whether the angle's uniform is the draw or its complement.</param>
    /// <param name="mean">Added to every draw; 0 leaves the draws unshifted.</param>
    internal static void FillStandardNormal<T>(Tensor<T> target, Random random, double scale = 1.0,
        BoxMullerGuard guard = BoxMullerGuard.Complement, double mean = 0.0, double floor = DefaultFloor,
        BoxMullerBranch branch = BoxMullerBranch.Cosine,
        BoxMullerSecondUniform secondUniform = BoxMullerSecondUniform.Raw)
    {
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (random is null) throw new ArgumentNullException(nameof(random));
        if (target.Length != 0) WriteStandardNormal(target.AsWritableSpan(), random, scale, guard, mean, floor, branch, secondUniform);
    }

    /// <summary>
    /// Writes standard normal draws that use both Box-Muller outputs: each pair of uniforms <c>(u1, u2)</c> from
    /// <paramref name="random"/> fills two consecutive elements with <c>r cos(theta)</c> and <c>r sin(theta)</c>, where
    /// <c>r = scale * sqrt(-2 ln(1 - u1))</c> and <c>theta = 2 pi u2</c>. An odd element count drops the last sine.
    /// </summary>
    /// <typeparam name="T">The numeric type of the destination.</typeparam>
    /// <param name="destination">The elements to overwrite.</param>
    /// <param name="random">The generator the uniforms are taken from.</param>
    /// <param name="scale">The standard deviation; 1 for standard normal values.</param>
    [AiDotNet.Attributes.ScalarKernel("Draws each pair from a sequential seeded Random stream; Box-Muller (Log, Sqrt, Cos, Sin) dominates.")]
    internal static void WriteStandardNormalPairs<T>(Span<T> destination, Random random, double scale = 1.0)
    {
        if (random is null) throw new ArgumentNullException(nameof(random));
        var ops = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < destination.Length; i += 2)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = random.NextDouble();
            double radius = scale * Math.Sqrt(-2.0 * Math.Log(u1));
            double angle = 2.0 * Math.PI * u2;
            destination[i] = ops.FromDouble(radius * Math.Cos(angle));
            if (i + 1 < destination.Length)
                destination[i + 1] = ops.FromDouble(radius * Math.Sin(angle));
        }
    }

    /// <summary>
    /// A tensor of standard normal draws filled in row-major order by both Box-Muller outputs; see
    /// <see cref="WriteStandardNormalPairs{T}"/>.
    /// </summary>
    /// <typeparam name="T">The numeric type of the model.</typeparam>
    /// <param name="shape">The shape of the tensor.</param>
    /// <param name="random">The generator the uniforms are taken from.</param>
    internal static Tensor<T> StandardNormalPairs<T>(int[] shape, Random random)
    {
        int count = ElementCount(shape);
        if (random is null) throw new ArgumentNullException(nameof(random));
        var result = new Tensor<T>(shape);
        if (count != 0) WriteStandardNormalPairs(result.AsWritableSpan(), random);
        return result;
    }

    private static int ElementCount(int[] shape)
    {
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        int count = 1;
        foreach (int dimension in shape)
            count = checked(count * dimension);
        return count;
    }
}
