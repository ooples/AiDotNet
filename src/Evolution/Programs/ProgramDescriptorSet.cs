using System.Collections.ObjectModel;
using AiDotNet.Interfaces;
using AiDotNet.Validation;

namespace AiDotNet.Evolution.Programs;

/// <summary>A validated, ordered collection of program descriptors evaluated together.</summary>
/// <remarks>
/// <para>
/// An archive dimension is identified by name, so two descriptors that answer to the same name would silently
/// overwrite each other's coordinate. This set rejects duplicate names at construction rather than at the first
/// evaluation, and <see cref="Compute"/> returns one dictionary keyed by those names, ready to hand to
/// <see cref="EvolutionTaskResult.Completed"/>. Every value is checked for finiteness, because a NaN coordinate
/// would be rejected deep inside the archive with a much less helpful message.
/// </para>
/// <para>
/// <see cref="VersionHash"/> folds in every descriptor name and type, so adding, removing, or renaming a dimension
/// changes the hash and a checkpoint written under the old set is refused rather than being reinterpreted against
/// a differently shaped grid. Name and type alone do not describe how a descriptor was configured, so a descriptor
/// that carries configuration - a diversity descriptor and its reference programs, for instance - also implements
/// <see cref="IVersionedProgramDescriptor"/> and contributes its own version hash here. Without that, swapping a
/// reference set would leave this hash identical and a resumed run would silently re-bin every restored elite
/// against coordinates it never produced. A descriptor that does not implement the interface is recorded as
/// unversioned, which is correct only when its type and name really do determine its output.
/// </para>
/// <para><b>For Beginners:</b> This is simply the list of pigeonhole axes your run uses, bundled together with a
/// safety check that no two axes share a name. Build one with, say, a length descriptor and a token-count
/// descriptor, then call <see cref="Compute"/> on a candidate to get both coordinates at once. The
/// <see cref="Names"/> property tells you which archive dimensions to declare so the two line up.</para>
/// </remarks>
public sealed class ProgramDescriptorSet
{
    private readonly ReadOnlyCollection<IProgramDescriptor> _descriptors;
    private readonly ReadOnlyCollection<string> _names;

    /// <summary>Initializes a descriptor set.</summary>
    /// <param name="descriptors">The descriptors to evaluate, in order; names must be unique.</param>
    /// <exception cref="ArgumentNullException"><paramref name="descriptors"/> is <c>null</c>, or an entry is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A descriptor has a blank name or two descriptors share a name.</exception>
    public ProgramDescriptorSet(IEnumerable<IProgramDescriptor> descriptors)
    {
        Guard.NotNull(descriptors);

        var ordered = new List<IProgramDescriptor>();
        var names = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (IProgramDescriptor descriptor in descriptors)
        {
            if (descriptor is null) throw new ArgumentNullException(nameof(descriptors), "Descriptors cannot be null.");
            if (string.IsNullOrWhiteSpace(descriptor.Name))
                throw new ArgumentException("Descriptor names cannot be empty or white space.", nameof(descriptors));
            string name = descriptor.Name.Trim();
            if (!seen.Add(name))
                throw new ArgumentException($"Descriptor name '{name}' is used more than once.", nameof(descriptors));
            ordered.Add(descriptor);
            names.Add(name);
        }

        _descriptors = new ReadOnlyCollection<IProgramDescriptor>(ordered);
        _names = new ReadOnlyCollection<string>(names);
        VersionHash = BuildVersionHash(ordered, names);
    }

    /// <summary>Initializes a descriptor set from an explicit argument list.</summary>
    /// <param name="descriptors">The descriptors to evaluate, in order; names must be unique.</param>
    /// <exception cref="ArgumentNullException"><paramref name="descriptors"/> is <c>null</c>, or an entry is <c>null</c>.</exception>
    /// <exception cref="ArgumentException">A descriptor has a blank name or two descriptors share a name.</exception>
    public ProgramDescriptorSet(params IProgramDescriptor[] descriptors)
        : this((IEnumerable<IProgramDescriptor>)(descriptors ?? throw new ArgumentNullException(nameof(descriptors))))
    {
    }

    /// <summary>Gets the descriptors in evaluation order.</summary>
    public IReadOnlyList<IProgramDescriptor> Descriptors => _descriptors;

    /// <summary>Gets the descriptor names, which are the archive dimension keys this set fills.</summary>
    public IReadOnlyList<string> Names => _names;

    /// <summary>Gets a version hash covering every descriptor name, implementation type, and own configuration.</summary>
    public string VersionHash { get; }

    /// <summary>Gets the number of descriptors in the set.</summary>
    public int Count => _descriptors.Count;

    /// <summary>Creates an empty set, for tasks whose evaluator supplies every descriptor itself.</summary>
    /// <returns>A set with no descriptors.</returns>
    public static ProgramDescriptorSet Empty() => new(Array.Empty<IProgramDescriptor>());

    /// <summary>Creates the standard set of built-in program descriptors.</summary>
    /// <param name="referenceSources">
    /// The fixed reference programs for the diversity dimension; pass an empty sequence to omit that dimension.
    /// </param>
    /// <returns>
    /// A set containing a length descriptor, a token-complexity descriptor, and — when references were supplied —
    /// a diversity descriptor.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="referenceSources"/> is <c>null</c>.</exception>
    public static ProgramDescriptorSet CreateDefault(IEnumerable<string> referenceSources)
    {
        Guard.NotNull(referenceSources);
        var references = new List<string>(referenceSources);
        var descriptors = new List<IProgramDescriptor>
        {
            new ProgramLengthDescriptor(),
            new ProgramTokenComplexityDescriptor()
        };

        if (references.Count > 0) descriptors.Add(new ProgramDiversityDescriptor(references));
        return new ProgramDescriptorSet(descriptors);
    }

    /// <summary>Computes every descriptor for one candidate program.</summary>
    /// <param name="genome">The candidate to measure.</param>
    /// <returns>A dictionary keyed by descriptor name; empty when the set has no descriptors.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="genome"/> is <c>null</c>.</exception>
    /// <exception cref="InvalidOperationException">A descriptor returned a value that is not finite.</exception>
    public IReadOnlyDictionary<string, double> Compute(ProgramGenome genome)
    {
        Guard.NotNull(genome);
        var values = new Dictionary<string, double>(StringComparer.Ordinal);
        for (int index = 0; index < _descriptors.Count; index++)
        {
            double value = _descriptors[index].Compute(genome);
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new InvalidOperationException(
                    $"Descriptor '{_names[index]}' returned a value that is not finite.");
            }

            values[_names[index]] = value;
        }

        return values;
    }

    private static string BuildVersionHash(List<IProgramDescriptor> descriptors, List<string> names)
    {
        var components = new List<string> { "program-descriptor-set-v2" };
        for (int index = 0; index < descriptors.Count; index++)
        {
            components.Add(names[index]);
            Type type = descriptors[index].GetType();
            components.Add(type.FullName ?? type.Name);
            components.Add(descriptors[index] is IVersionedProgramDescriptor versioned
                ? versioned.VersionHash ?? string.Empty
                : "unversioned");
        }

        return "program-descriptors-" + EvolutionHash.Combine(components);
    }
}
