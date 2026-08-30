namespace AiDotNet.Enums;

/// <summary>
/// How strictly a layer must reproduce its defined output when a faster route is available.
/// </summary>
/// <remarks>
/// <para>
/// A layer often has more than one way to compute the same thing: a fused kernel, an in-place
/// variant that writes into a caller-owned buffer, or a fast convolution algorithm such as Winograd.
/// These are not always numerically identical. Winograd in particular trades accuracy for speed
/// (Lavin and Gray, "Fast Algorithms for Convolutional Neural Networks", CVPR 2016), and an
/// accumulation performed in a different order rounds differently.
/// </para>
/// <para>
/// This makes that trade explicit and the caller's to make. It is modelled on Intel MKL's
/// Conditional Numerical Reproducibility, where reproducibility is a contract the library commits
/// to rather than a property a caller has to hope for. Crucially, the setting is the ONLY thing
/// that selects an algorithm: whether a gradient tape is recording, and whether the layer is in
/// training mode, never change what is computed. PyTorch draws the same line -- <c>no_grad</c>
/// governs whether a graph is built, not the values that flow through it.
/// </para>
/// <para><b>For Beginners:</b> leave this alone. The default already gives you the accurate answer,
/// Use <see cref="Fast"/> explicitly when last-bit replay equality is less important than throughput.
/// </para>
/// </remarks>
public enum NumericalReproducibility
{
    /// <summary>
    /// The layer's output must match its reference computation exactly, bit for bit.
    /// </summary>
    /// <remarks>
    /// The default. Exact always uses the canonical computation. A successful comparison for one
    /// input cannot certify a parallel fused reduction for later inputs, even when the shape is unchanged.
    /// </remarks>
    Exact = 0,

    /// <summary>
    /// Prefer the fastest available route, accepting that it may differ from the reference in the
    /// last few bits.
    /// </summary>
    /// <remarks>
    /// Opt in only when throughput matters more than reproducibility and you have satisfied yourself
    /// that the difference is immaterial for your model. Results may then vary between training and
    /// inference, and between builds that select different kernels.
    /// </remarks>
    Fast = 1,
}
