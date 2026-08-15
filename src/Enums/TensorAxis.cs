namespace AiDotNet.Enums;

/// <summary>
/// The SEMANTIC role of one axis of a tensor — what the numbers along that axis actually mean.
/// </summary>
/// <remarks>
/// <para>
/// This exists because a shape on its own is ambiguous and nothing in the system can resolve it.
/// <c>[8, 3, 32]</c> is equally well a batch of 8 three-channel signals of length 32 (NCW), a batch of 8
/// sequences of 3 timesteps with 32 features (NTC), or 8 frames of a 3x32 image. The layer cannot tell,
/// the test generator cannot tell, and a reader cannot tell. Today the generator guesses from the model's
/// DOMAIN tags, which is why it carries a long hand-written list of per-class fixture overrides — every
/// entry on that list is a case where the guess was wrong.
/// </para>
/// <para>
/// Naming an axis by ROLE removes the ambiguity at its source: NCW and NTC stop being the same shape,
/// because one declares <c>Channels, Width</c> and the other <c>Time, Features</c>.
/// </para>
/// <para><b>For Beginners:</b> A tensor is just a block of numbers with a shape like [8, 3, 32]. The shape
/// alone does not say whether "3" means three colour channels, three timesteps, or three of something
/// else. This enum lets code state which it is, so mistakes are caught by name rather than by a crash.</para>
/// </remarks>
public enum TensorAxis
{
    /// <summary>Independent samples processed together. Often optional — see the layout attribute.</summary>
    Batch = 0,

    /// <summary>Feature maps / colour channels. The C in NCHW.</summary>
    Channels = 1,

    /// <summary>Spatial rows.</summary>
    Height = 2,

    /// <summary>Spatial columns.</summary>
    Width = 3,

    /// <summary>Spatial depth, for volumetric (3-D) data.</summary>
    Depth = 4,

    /// <summary>
    /// Sequence position where the ordering is TEMPORAL and the model treats it as time.
    /// </summary>
    /// <remarks>
    /// Deliberately distinct from <see cref="Length"/>. Both are "a sequence axis", but a layer that
    /// applies causal masking or recurrence is only correct over Time, and conflating the two is exactly
    /// the class of mistake this enum exists to prevent.
    /// </remarks>
    Time = 5,

    /// <summary>Sequence position with no temporal meaning — the L in NCL, a 1-D signal's samples.</summary>
    Length = 6,

    /// <summary>Feature vector components, as consumed by a fully-connected layer.</summary>
    Features = 7,

    /// <summary>Video frames, when frames are a separate axis from Time or Batch.</summary>
    Frames = 8,

    /// <summary>Attention heads.</summary>
    Heads = 9,

    /// <summary>Discrete output classes, e.g. a classifier's logit axis.</summary>
    Classes = 10,

    /// <summary>
    /// A real axis whose role is genuinely model-specific and not worth a shared name.
    /// </summary>
    /// <remarks>
    /// An escape hatch, and it should stay rare. An axis marked Other participates in RANK checks but
    /// cannot be role-checked, so overusing it quietly returns the system to guessing from shape.
    /// </remarks>
    Other = 99,
}
