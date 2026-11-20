namespace AiDotNet.Enums;

/// <summary>
/// The normalization layer TabR builds wherever the reference model calls <c>Normalization(d_main)</c>: before
/// every block but the first encoder block, before the key projection when mixer normalization is on, and at the
/// start of the prediction head.
/// </summary>
/// <remarks>
/// <para>
/// The reference <c>yandex-research/tabular-dl-tabr</c> resolves the layer with <c>getattr(torch.nn, normalization)</c>,
/// so any <c>torch.nn</c> module taking the width is accepted. These are the ones with a meaning there.
/// </para>
/// <para><b>For Beginners:</b> normalization rescales each hidden vector so training stays stable. Keep
/// <see cref="LayerNorm"/>, which is what the paper uses, unless you are experimenting.</para>
/// </remarks>
public enum TabRNormalization
{
    /// <summary><c>nn.LayerNorm(d_main)</c>, normalizing each row over its features: the paper's default.</summary>
    LayerNorm = 0,

    /// <summary><c>nn.BatchNorm1d(d_main)</c>, normalizing each feature over the batch, with running statistics
    /// at prediction.</summary>
    BatchNorm = 1,

    /// <summary><c>nn.Identity</c>: no normalization at all.</summary>
    Identity = 2
}
