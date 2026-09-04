using AiDotNet.Evolution.Programs;

namespace AiDotNet.Interfaces;

/// <summary>A program descriptor that measures against other programs and can be pointed at a new set of them.</summary>
/// <remarks>
/// <para>
/// Some descriptors are absolute — how long a program is, how many tokens it uses — and mean the same thing on the
/// first candidate and the ten-thousandth. Others are relative: how unlike the rest of the population a candidate
/// is only means something with respect to some population. A relative descriptor measured once against a fixed
/// reference set stays comparable, which is why that is the default, but it saturates: once the search has moved
/// away from the reference, every candidate reads "far from the reference" and the axis stops telling them apart,
/// which is exactly when diversity pressure is worth having.
/// </para>
/// <para>
/// Rebasing is the answer, and the danger it introduces is inconsistency. If new candidates are measured against a
/// new reference while archived elites keep coordinates taken against the old one, the archive fills with readings
/// from different rulers: an elite filed early sits beside one filed late, and the two cells cannot meaningfully be
/// compared. The map looks fine and means nothing. So a rebase is only half the operation — the other half is
/// re-measuring what is already archived, which <c>MapElitesArchive.Remeasure</c> does.
/// </para>
/// <para>
/// Implementations must be immutable: <see cref="Rebase"/> returns a new descriptor rather than changing this one,
/// because the old descriptor may still be in use while the new one is being prepared, and because a descriptor's
/// version hash is supposed to identify the exact reading it produces.
/// </para>
/// <para><b>For Beginners:</b> Implement this only for a descriptor that compares a program with other programs. It
/// lets the search point that comparison at a fresh set of programs part-way through, so the measurement keeps
/// discriminating instead of flattening out.</para>
/// </remarks>
public interface IRebasableProgramDescriptor : IVersionedProgramDescriptor
{
    /// <summary>Returns an equivalent descriptor measuring against a new reference set.</summary>
    /// <param name="references">The programs the descriptor should compare candidates with from now on.</param>
    /// <returns>A new descriptor with the same name and reading, against the supplied references.</returns>
    /// <exception cref="System.ArgumentNullException"><paramref name="references"/> is <see langword="null"/>.</exception>
    IRebasableProgramDescriptor Rebase(IReadOnlyList<ProgramGenome> references);
}
