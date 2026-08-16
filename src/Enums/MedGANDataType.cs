namespace AiDotNet.Enums;

/// <summary>
/// The kind of variables medGAN's autoencoder is being fitted to. This selects the autoencoder's
/// activations and its pre-training reconstruction loss.
/// </summary>
/// <remarks>
/// <para>
/// medGAN (Choi et al., arXiv:1703.06490) is defined for "high-dimensional discrete variables (e.g.,
/// binary and count features)" and gives a separate activation/loss pair for each. §3.2 states the
/// count case as the mean squared loss (Eq. 2) and the binary case as the cross entropy loss (Eq. 3).
/// </para>
/// <para>
/// <b>For Beginners:</b> Patient records come in two shapes. Either a column says "does this patient
/// have diagnosis X?" — a yes/no, which is <see cref="Binary"/> — or it says "how many times did
/// this happen?" — a count, which is <see cref="Count"/>. The two need different maths, so you tell
/// the model which one you have. <see cref="MixedTabular"/> is for the common real-world case where
/// a single table has some of each plus ordinary numeric columns.
/// </para>
/// </remarks>
public enum MedGANDataType
{
    /// <summary>
    /// Binary variables. Encoder activation tanh, decoder activation sigmoid, pre-training loss the
    /// cross entropy of Eq. 3: <c>sum(x_i log x'_i + (1 - x_i) log(1 - x'_i))</c>.
    /// </summary>
    Binary = 0,

    /// <summary>
    /// Count variables. Encoder and decoder activation ReLU, pre-training loss the mean squared loss
    /// of Eq. 2: <c>sum(||x_i - x'_i||^2)</c>.
    /// </summary>
    Count = 1,

    /// <summary>
    /// Mixed tabular data as produced by <c>TabularDataTransformer</c>: each continuous column
    /// becomes one mode-normalized scalar plus a one-hot mode indicator, and each categorical column
    /// becomes a one-hot group.
    /// </summary>
    /// <remarks>
    /// This is not a case the paper enumerates, because medGAN was evaluated on pure binary/count
    /// EHR code matrices. It is the faithful generalization of the two cases it does enumerate: the
    /// squared loss of Eq. 2 is applied to the continuous scalars and the cross entropy of Eq. 3 is
    /// applied to each one-hot group (where it is exactly a softmax cross entropy, the multi-class
    /// form of the binary expression). A table that happens to be all-binary reduces to
    /// <see cref="Binary"/> and one that is all-continuous reduces to <see cref="Count"/>.
    /// </remarks>
    MixedTabular = 2,
}
