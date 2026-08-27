namespace AiDotNet.Models;

/// <summary>
/// Provides an independent constructor argument that carries configuration but not runtime state.
/// </summary>
/// <remarks>
/// <para>
/// Some constructor arguments are mutable blueprints rather than models themselves. Passing one of
/// those objects directly to a reconstructed model can make the source and clone share the mutable
/// objects the blueprint owns. Implementing this contract lets the central clone engine duplicate
/// that configuration without teaching it about every blueprint type.
/// </para>
/// <para>
/// Runtime parameters are deliberately outside this contract. The model and layer copy-on-write
/// paths restore those after construction, so configuration cloning stays cheap.
/// </para>
/// </remarks>
internal interface IConfigurationCloneable
{
    /// <summary>
    /// Creates an independent copy containing only constructor-level configuration.
    /// </summary>
    object CloneConfiguration();
}
