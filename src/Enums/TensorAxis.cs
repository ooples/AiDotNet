namespace AiDotNet.Enums;

/// <summary>
/// The ROLE an axis plays in a tensor layout, as declared by
/// <see cref="AiDotNet.Attributes.TensorLayoutAttribute"/>.
/// </summary>
/// <remarks>
/// <para>
/// A shape like <c>[32, 3, 224, 224]</c> says nothing about what those numbers mean; a layout like
/// <c>[Batch, Channels, Height, Width]</c> does. This enum is that vocabulary, and it exists so a
/// declared layout can be checked against a real tensor at build time rather than discovered as a
/// mis-shaped input at run time.
/// </para>
/// <para>
/// THESE ARE ROLES, NOT SIZES. Nothing here records how long an axis is; a layout constrains rank and
/// meaning only. Tied relationships between axis lengths — a super-resolution model's
/// <c>output Width = ScaleFactor * input Width</c> — are runtime facts and live on
/// <see cref="AiDotNet.Interfaces.IShapeContract"/> instead.
/// </para>
/// <para>
/// The names are the wire format. <c>ShapeDeclarationValidationGenerator</c> runs inside the compiler,
/// where it reads the attribute's arguments as symbols and compares axis MEMBER NAMES as strings — it
/// cannot load this enum. Renaming a member therefore changes generator behaviour silently, so treat
/// these names as part of the public contract and add rather than rename.
/// </para>
/// <para><b>For Beginners:</b> These are labels for each dimension of your data. An image batch is
/// <c>[Batch, Channels, Height, Width]</c>: how many images, how many colour channels each, and how
/// tall and wide they are.</para>
/// </remarks>
public enum TensorAxis
{
    /// <summary>How many independent samples are stacked together.</summary>
    /// <remarks>
    /// This is the axis <see cref="AiDotNet.Attributes.TensorLayoutAttribute.BatchOptional"/> refers to,
    /// and it is only ever meaningful as the FIRST axis — an optional axis in the middle of a layout
    /// would make rank ambiguous, which is the thing layouts exist to remove.
    /// </remarks>
    Batch = 0,

    /// <summary>Position in a sequence: a timestep, a token, or a frame.</summary>
    Time = 1,

    /// <summary>Feature channels — colour planes on an image, filter maps inside a convolutional stack.</summary>
    Channels = 2,

    /// <summary>The third spatial extent, for volumetric and video-as-volume data.</summary>
    Depth = 3,

    /// <summary>The vertical spatial extent.</summary>
    Height = 4,

    /// <summary>The horizontal spatial extent.</summary>
    Width = 5,

    /// <summary>A flat feature vector, as consumed by a dense layer.</summary>
    Features = 6,
}
