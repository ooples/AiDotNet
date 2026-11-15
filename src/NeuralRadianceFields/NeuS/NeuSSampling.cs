namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// NeuS's ray sampling, on the host in double precision: the bounding-sphere near/far, the uniform and perturbed
/// samples, the hierarchical up-sampling with fixed sharpness, and inverse-CDF sampling.
/// </summary>
/// <remarks>
/// <para>
/// The reference runs all of this under <c>torch.no_grad()</c> and detaches the resulting depths, so the sample
/// positions are constants of the differentiable render, exactly as here. Arrays are row-major <c>[rays, samples]</c>.
/// </para>
/// </remarks>
internal static class NeuSSampling
{
    /// <summary>
    /// The near and far depths of each ray's chord through the sphere's bounding interval: the ray's closest
    /// approach to the origin ∓ <paramref name="radius"/> (reference <c>near_far_from_sphere</c>).
    /// </summary>
    /// <param name="origins">Ray origins, <c>[R, 3]</c>.</param>
    /// <param name="directions">Ray directions, <c>[R, 3]</c>.</param>
    /// <param name="radius">The sphere radius (1 in NeuS).</param>
    public static (double[] Near, double[] Far) NearFarFromSphere(double[] origins, double[] directions, double radius = 1.0)
    {
        int rays = origins.Length / 3;
        var near = new double[rays];
        var far = new double[rays];
        for (int r = 0; r < rays; r++)
        {
            double a = 0, b = 0;
            for (int c = 0; c < 3; c++)
            {
                a += directions[r * 3 + c] * directions[r * 3 + c];
                b += 2.0 * origins[r * 3 + c] * directions[r * 3 + c];
            }

            double mid = 0.5 * (-b) / a;
            near[r] = mid - radius;
            far[r] = mid + radius;
        }

        return (near, far);
    }

    /// <summary>
    /// <paramref name="samples"/> depths evenly spaced from near to far (both included), shifted per ray by
    /// <c>(u − 0.5) · 2 / samples</c> when <paramref name="random"/> is given (NeuS's <c>perturb</c>).
    /// </summary>
    public static double[] UniformDepths(double[] near, double[] far, int samples, Random? random)
    {
        int rays = near.Length;
        var z = new double[rays * samples];
        for (int r = 0; r < rays; r++)
        {
            double shift = random is null ? 0.0 : (random.NextDouble() - 0.5) * 2.0 / samples;
            for (int s = 0; s < samples; s++)
            {
                double t = samples == 1 ? 0.0 : (double)s / (samples - 1);
                z[r * samples + s] = near[r] + (far[r] - near[r]) * t + shift;
            }
        }

        return z;
    }

    /// <summary>
    /// Evenly spaced depths with stratified jitter inside each sample's interval (SparseNeuS's <c>perturb</c>: the
    /// bounds are the midpoints between neighbors and the end samples).
    /// </summary>
    public static double[] StratifiedDepths(double[] near, double[] far, int samples, Random? random)
    {
        var z = UniformDepths(near, far, samples, null);
        if (random is null) return z;
        int rays = near.Length;
        var jittered = new double[z.Length];
        for (int r = 0; r < rays; r++)
            for (int s = 0; s < samples; s++)
            {
                int i = r * samples + s;
                double lower = s == 0 ? z[i] : 0.5 * (z[i] + z[i - 1]);
                double upper = s == samples - 1 ? z[i] : 0.5 * (z[i] + z[i + 1]);
                jittered[i] = lower + (upper - lower) * random.NextDouble();
            }

        return jittered;
    }

    /// <summary>
    /// One round of NeuS's hierarchical up-sampling (reference <c>up_sample</c>): estimate each section's SDF slope as
    /// the smaller of its own and its predecessor's finite differences, clamped to [<paramref name="minSlope"/>, 0]
    /// and zeroed off the valid region, form the section alpha with the fixed sharpness <paramref name="inverseS"/>,
    /// and draw <paramref name="count"/> new depths from the resulting weights by deterministic inverse-CDF sampling.
    /// </summary>
    /// <param name="origins">Ray origins, <c>[R, 3]</c>.</param>
    /// <param name="directions">Ray directions, <c>[R, 3]</c>.</param>
    /// <param name="z">The current depths, <c>[R, S]</c>, sorted per ray.</param>
    /// <param name="sdf">The SDF at those depths, <c>[R, S]</c>.</param>
    /// <param name="samples">S.</param>
    /// <param name="count">New depths per ray.</param>
    /// <param name="inverseS">The S-density sharpness for this round (64·2ⁱ in the reference).</param>
    /// <param name="minSlope">The slope clamp's lower end (−1000 in NeuS, −10 in SparseNeuS).</param>
    /// <param name="sectionValid">Per section <c>[R, S − 1]</c> validity, or null for NeuS's rule (either end inside the unit sphere).</param>
    /// <param name="zeroInvalidAlpha">Whether an invalid section also gets zero opacity (SparseNeuS) rather than only a
    /// zero slope (SyncDreamer). Applies only with <paramref name="sectionValid"/>.</param>
    public static double[] UpSample(double[] origins, double[] directions, double[] z, double[] sdf, int samples, int count,
        double inverseS, double minSlope, bool[]? sectionValid, bool zeroInvalidAlpha = true)
    {
        int rays = origins.Length / 3;
        int sections = samples - 1;
        var weights = new double[rays * sections];
        for (int r = 0; r < rays; r++)
        {
            double previousSlope = 0.0;
            double transmittance = 1.0;
            for (int s = 0; s < sections; s++)
            {
                int i = r * samples + s;
                double prevZ = z[i], nextZ = z[i + 1];
                double prevSdf = sdf[i], nextSdf = sdf[i + 1];
                double slope = (nextSdf - prevSdf) / (nextZ - prevZ + 1e-5);
                double used = Math.Min(previousSlope, slope);
                previousSlope = slope;
                used = Math.Max(minSlope, Math.Min(0.0, used));
                bool valid = sectionValid is not null
                    ? sectionValid[r * sections + s]
                    : Radius(origins, directions, r, prevZ) < 1.0 || Radius(origins, directions, r, nextZ) < 1.0;
                if (!valid) used = 0.0;
                double dist = nextZ - prevZ;
                double mid = 0.5 * (prevSdf + nextSdf);
                double prevCdf = Sigmoid((mid - used * dist * 0.5) * inverseS);
                double nextCdf = Sigmoid((mid + used * dist * 0.5) * inverseS);
                double alpha = (prevCdf - nextCdf + 1e-5) / (prevCdf + 1e-5);
                if (sectionValid is not null && zeroInvalidAlpha && !valid) alpha = 0.0;
                weights[r * sections + s] = alpha * transmittance;
                transmittance *= 1.0 - alpha + 1e-7;
            }
        }

        return SamplePdf(z, weights, samples, count, null);
    }

    /// <summary>
    /// Whether each sample of <c>[R, S]</c> depths lies in the cube <c>|x|, |y|, |z| &lt;= halfSide</c>, row-major
    /// (SyncDreamer's <c>get_inner_mask</c>).
    /// </summary>
    public static bool[] InsideCube(double[] origins, double[] directions, double[] z, int samples, double halfSide)
    {
        var points = Points(origins, directions, z, samples);
        int n = points.Length / 3;
        var inside = new bool[n];
        for (int i = 0; i < n; i++)
            inside[i] = Math.Abs(points[i * 3]) <= halfSide && Math.Abs(points[i * 3 + 1]) <= halfSide
                && Math.Abs(points[i * 3 + 2]) <= halfSide;
        return inside;
    }

    /// <summary>
    /// Per section <c>[R, S − 1]</c>, whether either end is inside the cube (SyncDreamer's <c>upsample</c>:
    /// <c>inner_mask[:, :-1] | inner_mask[:, 1:]</c>).
    /// </summary>
    public static bool[] SectionsTouchingCube(double[] origins, double[] directions, double[] z, int samples, double halfSide)
    {
        var inside = InsideCube(origins, directions, z, samples, halfSide);
        int rays = z.Length / samples, sections = samples - 1;
        var valid = new bool[rays * sections];
        for (int r = 0; r < rays; r++)
            for (int s = 0; s < sections; s++)
                valid[r * sections + s] = inside[r * samples + s] || inside[r * samples + s + 1];
        return valid;
    }

    /// <summary>
    /// NeRF's inverse-CDF sampling (reference <c>sample_pdf</c>): <paramref name="count"/> depths per ray from the
    /// piecewise-constant density with <c>S − 1</c> section weights over <c>S</c> bins, at the deterministic quantiles
    /// <c>(k + 0.5) / count</c> when <paramref name="random"/> is null.
    /// </summary>
    public static double[] SamplePdf(double[] bins, double[] weights, int binCount, int count, Random? random)
    {
        int rays = bins.Length / binCount;
        int sections = binCount - 1;
        var result = new double[rays * count];
        var cdf = new double[binCount];
        var u = new double[count];
        for (int r = 0; r < rays; r++)
        {
            double total = 0;
            for (int s = 0; s < sections; s++) total += weights[r * sections + s] + 1e-5;
            cdf[0] = 0.0;
            for (int s = 0; s < sections; s++) cdf[s + 1] = cdf[s] + (weights[r * sections + s] + 1e-5) / total;
            for (int k = 0; k < count; k++)
                u[k] = random is null ? (count == 1 ? 0.5 : 0.5 / count + k * (1.0 - 1.0 / count) / (count - 1)) : random.NextDouble();
            for (int k = 0; k < count; k++)
            {
                // searchsorted(cdf, u, right=True): the first index whose cdf exceeds u.
                int index = 0;
                while (index < binCount && cdf[index] <= u[k]) index++;
                int below = Math.Max(0, index - 1);
                int above = Math.Min(binCount - 1, index);
                double denom = cdf[above] - cdf[below];
                if (denom < 1e-5) denom = 1.0;
                double t = (u[k] - cdf[below]) / denom;
                double binBelow = bins[r * binCount + below], binAbove = bins[r * binCount + above];
                result[r * count + k] = binBelow + t * (binAbove - binBelow);
            }
        }

        return result;
    }

    /// <summary>
    /// Merges new depths into each ray's sorted depths (reference <c>cat_z_vals</c>), carrying the per-sample values
    /// along; ties keep the existing sample first.
    /// </summary>
    /// <param name="z">The current depths, <c>[R, S]</c>.</param>
    /// <param name="values">The values at the current depths, <c>[R, S]</c>, or null.</param>
    /// <param name="samples">S.</param>
    /// <param name="newZ">The new depths, <c>[R, K]</c>.</param>
    /// <param name="newValues">The values at the new depths, <c>[R, K]</c>, or null.</param>
    /// <param name="count">K.</param>
    public static (double[] Z, double[]? Values) Merge(double[] z, double[]? values, int samples, double[] newZ,
        double[]? newValues, int count)
    {
        int rays = z.Length / samples;
        int total = samples + count;
        var mergedZ = new double[rays * total];
        var mergedValues = values is not null && newValues is not null ? new double[rays * total] : null;
        var order = new int[total];
        var keys = new double[total];
        for (int r = 0; r < rays; r++)
        {
            for (int s = 0; s < samples; s++) keys[s] = z[r * samples + s];
            for (int k = 0; k < count; k++) keys[samples + k] = newZ[r * count + k];
            for (int i = 0; i < total; i++) order[i] = i;
            // Stable: equal depths keep their index order.
            Array.Sort(order, (a, b) => keys[a] != keys[b] ? keys[a].CompareTo(keys[b]) : a.CompareTo(b));
            for (int i = 0; i < total; i++)
            {
                int source = order[i];
                mergedZ[r * total + i] = keys[source];
                if (mergedValues is not null && values is not null && newValues is not null)
                    mergedValues[r * total + i] = source < samples ? values[r * samples + source] : newValues[r * count + source - samples];
            }
        }

        return (mergedZ, mergedValues);
    }

    /// <summary>The points <c>o + d·z</c> for every ray and depth, <c>[R·S, 3]</c>.</summary>
    public static double[] Points(double[] origins, double[] directions, double[] z, int samples)
    {
        int rays = origins.Length / 3;
        var points = new double[rays * samples * 3];
        for (int r = 0; r < rays; r++)
            for (int s = 0; s < samples; s++)
                for (int c = 0; c < 3; c++)
                    points[(r * samples + s) * 3 + c] = origins[r * 3 + c] + directions[r * 3 + c] * z[r * samples + s];
        return points;
    }

    private static double Radius(double[] origins, double[] directions, int ray, double depth)
    {
        double sum = 0;
        for (int c = 0; c < 3; c++)
        {
            double p = origins[ray * 3 + c] + directions[ray * 3 + c] * depth;
            sum += p * p;
        }

        return Math.Sqrt(sum);
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
}
