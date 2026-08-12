namespace AiDotNet.Models;

/// <summary>
/// How much of the original's storage a copy is allowed to share.
/// </summary>
/// <remarks>
/// <para>
/// Orthogonal to the rest of <see cref="CloneOptions"/>. Those decide WHAT a copy carries —
/// parameters, optimizer state, buffers; this decides whether what it carries is copied or shared.
/// </para>
/// <para><b>For Beginners:</b> the difference is what happens when you train the copy.
/// With <see cref="Deep"/> and <see cref="CopyOnWrite"/>, nothing happens to the original. With
/// <see cref="Shared"/>, you train both, because there is only one set of weights.</para>
/// </remarks>
public enum CloneMode
{
    /// <summary>
    /// The copy owns its own storage. Training it never touches the original.
    /// </summary>
    /// <remarks>
    /// The default, and the least surprising reading of the word "clone": the same thing,
    /// separately.
    /// </remarks>
    Deep = 0,

    /// <summary>
    /// The copy shares each weight tensor's storage until either side writes to it.
    /// </summary>
    /// <remarks>
    /// Observationally identical to <see cref="Deep"/> — the first write on either side splits them
    /// — but O(1) until that happens instead of allocating a second full set of weights. Worth
    /// choosing when clones are made far more often than they are trained, as in population search
    /// or evaluating a checkpoint.
    /// </remarks>
    CopyOnWrite = 1,

    /// <summary>
    /// The copy points at the SAME parameters as the original.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Not a copy.</b> Training the result trains the original, because they are one set of
    /// weights behind two handles. It exists for read-only fan-out — running the same weights over
    /// several inputs concurrently — where allocating anything at all is waste.
    /// </para>
    /// <para>
    /// Anything that mutates one side is a shared-state hazard. If that is not obviously fine for
    /// what you are doing, use <see cref="CopyOnWrite"/>, which is as cheap right up until the
    /// moment sharing would have been wrong.
    /// </para>
    /// </remarks>
    Shared = 2,
}
