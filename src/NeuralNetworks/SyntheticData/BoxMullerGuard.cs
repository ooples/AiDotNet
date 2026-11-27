namespace AiDotNet.NeuralNetworks.SyntheticData;

/// <summary>
/// How a Box-Muller draw keeps its first uniform away from zero, where the logarithm is undefined.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> the Box-Muller method turns two random numbers between 0 and 1 into a bell-curve
/// random number, and it takes the logarithm of the first one. The logarithm of 0 is not a number, so the first
/// value has to be nudged away from 0. The generators do this in three slightly different ways; this setting names
/// which one, so that a seeded generator keeps producing exactly the same numbers.</para>
/// </remarks>
internal enum BoxMullerGuard
{
    /// <summary>Use <c>1 - u</c>, which lies in (0, 1].</summary>
    Complement,

    /// <summary>Use <c>max(1 - u, 1e-10)</c>.</summary>
    FlooredComplement,

    /// <summary>Use <c>max(1e-10, u)</c>.</summary>
    Floored,
}
