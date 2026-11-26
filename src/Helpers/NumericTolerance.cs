namespace AiDotNet.Helpers;

/// <summary>
/// Type-aware small constants (floors, epsilons, thresholds) for code that evaluates in a generic numeric type.
/// </summary>
/// <remarks>
/// <para>
/// A constant such as 1e-10 is a meaningful floor in double and single precision, but it rounds to zero in half
/// precision (the smallest normal half is 2^-14, about 6.1e-5). A floor of zero no longer keeps a logarithm or a
/// divisor away from zero, so code written as <c>FromDouble(1e-10)</c> silently stops protecting anything when it
/// runs in <see cref="Half"/>. <see cref="Of{T}"/> returns the requested value when <typeparamref name="T"/> can hold it with at least ten
/// significant bits, and the smallest such power of two otherwise (2^-14 for half precision).
/// </para>
/// <para><b>For Beginners:</b> Some number types cannot store very small values. This picks the smallest value the
/// type can actually hold whenever the value you asked for would round away to zero.</para>
/// </remarks>
internal static class NumericTolerance
{
    /// <summary>
    /// <paramref name="requested"/> in <typeparamref name="T"/> when <typeparamref name="T"/> holds it with ten
    /// significant bits; otherwise the smallest power of two that it does.
    /// </summary>
    internal static T Of<T>(double requested)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        return numOps.FromDouble(Math.Max(requested, SmallestPrecise<T>.Value));
    }

    private static class SmallestPrecise<T>
    {
        internal static readonly double Value = Compute();

        /// <summary>
        /// The smallest power of two 2^-k such that 2^-k and a value 2^-10 above it both survive the round trip
        /// through <typeparamref name="T"/> for every step down to it: the second check fails once the type runs
        /// out of mantissa bits, i.e. at the subnormal range. Integral types stop at one.
        /// </summary>
        private static double Compute()
        {
            const int steps = 1074;
            const double probeOffset = 1.0 / 1024.0;

            // Candidates 2^-1 .. 2^-1074 and their probes, round-tripped through T in one cast each way
            var candidates = new double[2 * steps];
            double power = 1.0;
            for (int step = 0; step < steps; step++)
            {
                power /= 2.0;
                candidates[2 * step] = power;
                candidates[2 * step + 1] = power * (1.0 + probeOffset);
            }

            var roundTrip = new Tensor<double>(candidates, new[] { candidates.Length }).Cast<T>().Contiguous().Cast<double>().ToArray();

            double smallest = 1.0;
            for (int step = 0; step < steps; step++)
            {
                double half = candidates[2 * step];
                if (half == 0.0 || roundTrip[2 * step] != half || roundTrip[2 * step + 1] != candidates[2 * step + 1])
                {
                    break;
                }

                smallest = half;
            }

            return smallest;
        }
    }
}
