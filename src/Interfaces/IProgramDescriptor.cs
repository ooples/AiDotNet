using AiDotNet.Evolution.Programs;

namespace AiDotNet.Interfaces;

/// <summary>Computes one named, finite behaviour descriptor from a candidate program's text alone.</summary>
/// <remarks>
/// <para>
/// Quality-diversity search keeps the best candidate of every kind rather than only the single best candidate, and
/// descriptors are what define "kind". A program descriptor reads only the genome — never an execution result — so
/// it can be computed before a candidate is run, costs nothing, and stays identical on every machine. That is what
/// makes descriptor values safe to fold into an archive whose cells must line up across a checkpoint resume.
/// </para>
/// <para>
/// Implementations must be pure and deterministic: the same genome must always produce the same finite value, with
/// no clock, no file system, no ambient randomness, and no dependence on how many candidates have been seen so
/// far. A descriptor that needs a corpus should take a fixed reference set at construction time rather than
/// consulting a live archive.
/// </para>
/// <para><b>For Beginners:</b> Imagine sorting candidate programs into pigeonholes so the search keeps a variety
/// of good answers rather than ten near-copies of one answer. A descriptor is one label on those pigeonholes, such
/// as "how long is the code" or "how many tokens does it use". This interface computes one such label from the
/// program text. Keep it cheap and repeatable: the same program must always land in the same pigeonhole.</para>
/// </remarks>
public interface IProgramDescriptor
{
    /// <summary>Gets the unique descriptor name used as the archive dimension key.</summary>
    string Name { get; }

    /// <summary>Computes the descriptor value for one candidate program.</summary>
    /// <param name="genome">The candidate to measure.</param>
    /// <returns>A finite value; never NaN or infinity.</returns>
    double Compute(ProgramGenome genome);
}
