namespace AiDotNet.NeuralRadianceFields.NeuS;

/// <summary>
/// The real spherical-harmonics encoding of a unit direction in tiny-cuda-nn's <c>SphericalHarmonics</c> layout
/// (Müller et al. 2022, Sec. 5.4; the instant-nsr-pl <c>dir_encoding_config</c>): degree d gives the d² terms of bands
/// 0 … d − 1, with tiny-cuda-nn's sign convention.
/// </summary>
/// <remarks>
/// tiny-cuda-nn reads its input in <c>[0, 1]</c> and maps it back with <c>2u − 1</c>; instant-nsr-pl feeds it
/// <c>(d + 1) / 2</c>, so the basis is evaluated at the direction itself, which is what this takes.
/// </remarks>
internal static class NeuSSphericalHarmonics
{
    /// <summary>The largest supported degree (bands 0 … 3, 16 terms).</summary>
    public const int MaxDegree = 4;

    /// <summary>The basis at <c>N</c> directions, <c>N · degree²</c> values (row-major).</summary>
    /// <param name="directions">The directions, <c>N · 3</c> values.</param>
    /// <param name="degree">The degree, 1 to <see cref="MaxDegree"/>.</param>
    public static double[] Evaluate(double[] directions, int degree)
    {
        if (directions is null || directions.Length % 3 != 0)
            throw new ArgumentException("The directions must be N · 3 values.", nameof(directions));
        if (degree < 1 || degree > MaxDegree) throw new ArgumentOutOfRangeException(nameof(degree));
        int n = directions.Length / 3;
        int terms = degree * degree;
        var result = new double[n * terms];
        for (int p = 0; p < n; p++)
        {
            double x = directions[p * 3], y = directions[p * 3 + 1], z = directions[p * 3 + 2];
            int o = p * terms;
            result[o] = 0.28209479177387814;
            if (degree <= 1) continue;
            result[o + 1] = -0.48860251190291987 * y;
            result[o + 2] = 0.48860251190291987 * z;
            result[o + 3] = -0.48860251190291987 * x;
            if (degree <= 2) continue;
            double x2 = x * x, y2 = y * y, z2 = z * z, xy = x * y, xz = x * z, yz = y * z;
            result[o + 4] = 1.0925484305920792 * xy;
            result[o + 5] = -1.0925484305920792 * yz;
            result[o + 6] = 0.94617469575755997 * z2 - 0.31539156525251999;
            result[o + 7] = -1.0925484305920792 * xz;
            result[o + 8] = 0.54627421529603959 * x2 - 0.54627421529603959 * y2;
            if (degree <= 3) continue;
            result[o + 9] = 0.59004358992664352 * y * (-3.0 * x2 + y2);
            result[o + 10] = 2.8906114426405538 * xy * z;
            result[o + 11] = 0.45704579946446572 * y * (1.0 - 5.0 * z2);
            result[o + 12] = 0.3731763325901154 * z * (5.0 * z2 - 3.0);
            result[o + 13] = 0.45704579946446572 * x * (1.0 - 5.0 * z2);
            result[o + 14] = 1.4453057213202769 * z * (x2 - y2);
            result[o + 15] = 0.59004358992664352 * x * (-x2 + 3.0 * y2);
        }

        return result;
    }
}
