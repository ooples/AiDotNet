namespace AiDotNet.Interfaces;

/// <summary>A program descriptor that publishes a hash of its own configuration, not only its name and type.</summary>
/// <remarks>
/// <para>
/// <c>ProgramDescriptorSet.VersionHash</c> is what stops a resumed run from reinterpreting a checkpointed archive
/// against a differently shaped grid, but a descriptor's name and implementation type do not describe how it was
/// configured. A diversity descriptor built against one reference set and a diversity descriptor built against a
/// completely different one share both, so without this interface swapping the reference set would leave the
/// compatibility hash unchanged and a resumed run would silently re-bin every restored elite against coordinates
/// it never produced.
/// </para>
/// <para>
/// An implementation returns a stable, culture-independent string that changes whenever anything that can change a
/// computed descriptor value changes, and that is identical for two instances configured the same way. A
/// descriptor whose behaviour is fully determined by its type and name has nothing to add and may simply not
/// implement this interface; <c>ProgramDescriptorSet</c> then records it as unversioned.
/// </para>
/// <para><b>For Beginners:</b> The archive sorts candidates into pigeonholes, and a checkpoint can only be resumed
/// if the pigeonholes still mean the same thing. Most descriptors are fully described by their name — "length"
/// always measures length — but some are set up with extra data, such as a list of example programs to compare
/// against. If you change that data the numbers change, so the run must not resume an old checkpoint. Implement
/// this interface on your descriptor and return a string built from its settings, and the safety check happens
/// automatically.</para>
/// </remarks>
public interface IVersionedProgramDescriptor : IProgramDescriptor
{
    /// <summary>Gets a stable hash of every setting that can change a value this descriptor computes.</summary>
    string VersionHash { get; }
}
