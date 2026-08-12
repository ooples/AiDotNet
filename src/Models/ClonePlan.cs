using System;
using System.Collections.Generic;
using System.Reflection;

namespace AiDotNet.Models;

/// <summary>
/// Describes how to reproduce one type: which members are configuration, and how each is carried.
/// </summary>
/// <remarks>
/// <para>
/// A plan is produced at compile time by the clone plan generator and stored in
/// <see cref="CloneRegistry"/>. Nothing here is decided at clone time, so a clone cannot depend on
/// the state of the object it is copying — which is what makes the result reviewable and testable
/// rather than emergent.
/// </para>
/// <para>
/// <b>Why a plan rather than generated copy code:</b> a Roslyn generator can only add members to a
/// <c>partial</c> type, and none of the 594 options classes are partial. Emitting a plan instead
/// keeps every one of them untouched while still deciding correctness at compile time, and means a
/// class written by a user works without them declaring anything at all.
/// </para>
/// </remarks>
public sealed class ClonePlan
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ClonePlan"/> class.
    /// </summary>
    /// <param name="type">The type this plan reproduces.</param>
    /// <param name="entries">The configuration members, in a stable order.</param>
    /// <exception cref="ArgumentNullException">Thrown when an argument is null.</exception>
    public ClonePlan(
        Type type,
        IReadOnlyList<ClonePlanEntry> entries,
        IReadOnlyList<string>? constructorParameters = null,
        IReadOnlyList<IReadOnlyList<string>>? constructorCandidates = null)
    {
        Type = type ?? throw new ArgumentNullException(nameof(type));
        Entries = entries ?? throw new ArgumentNullException(nameof(entries));
        ConstructorParameters = constructorParameters ?? Array.Empty<string>();
        ConstructorCandidates = constructorCandidates
            ?? (ConstructorParameters.Count > 0
                ? new[] { ConstructorParameters }
                : Array.Empty<IReadOnlyList<string>>());
    }

    /// <summary>
    /// Gets every constructor the type can be rebuilt through, widest first.
    /// </summary>
    /// <remarks>
    /// <para>
    /// More than one is normal, and recording only the widest was wrong. Around fifty models in this
    /// library take an ONNX model path in one constructor and an optimizer in another; a model built
    /// natively has no path stored, so rebuilding it through the ONNX constructor passes null and
    /// throws. Which constructor is right is a property of the INSTANCE, not of the type, and cannot
    /// be decided when the plan is generated.
    /// </para>
    /// <para>
    /// So the choice is deferred: every satisfiable constructor is recorded, and
    /// <c>CloneEngine</c> picks the one whose required arguments the instance actually holds. The
    /// mode is read off the state the object already carries rather than recorded separately.
    /// </para>
    /// </remarks>
    public IReadOnlyList<IReadOnlyList<string>> ConstructorCandidates { get; }

    /// <summary>Gets the type this plan reproduces.</summary>
    public Type Type { get; }

    /// <summary>
    /// Gets the configuration property names feeding this type's constructor, in parameter order.
    /// </summary>
    /// <value>Empty when the type is reconstructed without arguments, as options classes are.</value>
    /// <remarks>
    /// <para>
    /// Layers and models take arguments, so reconstructing them means calling a real constructor
    /// rather than allocating and assigning. Recording which carried property feeds each parameter
    /// is what makes that automatic for the author: they write an ordinary constructor and store
    /// its arguments in same-named properties, and nothing else.
    /// </para>
    /// <para>
    /// The list is only ever populated when the generator proved that EVERY parameter maps to a
    /// carried property. That proof is what makes reconstruction correct by construction rather
    /// than by check: if every input to the constructor is carried, the constructor is a pure
    /// function of carried configuration, so the rebuilt object is structurally identical. A
    /// parameter it cannot map is a build error naming that parameter, not a silent omission.
    /// </para>
    /// </remarks>
    public IReadOnlyList<string> ConstructorParameters { get; }

    /// <summary>Gets the configuration members carried by a clone, in a stable order.</summary>
    /// <remarks>
    /// Includes members declared on base types. Missing an inherited member is precisely how 71
    /// copy constructors came to drop <c>ModelOptions.Seed</c>, so the plan is built from the full
    /// inheritance chain rather than from a single type's declarations.
    /// </remarks>
    public IReadOnlyList<ClonePlanEntry> Entries { get; }
}

/// <summary>
/// One configuration member and the manner in which a clone carries it.
/// </summary>
public sealed class ClonePlanEntry
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ClonePlanEntry"/> class.
    /// </summary>
    /// <param name="property">The property to carry.</param>
    /// <param name="copy">How to carry it.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="property"/> is null.</exception>
    public ClonePlanEntry(PropertyInfo property, CloneCopyKind copy)
    {
        Property = property ?? throw new ArgumentNullException(nameof(property));
        Copy = copy;
    }

    /// <summary>Gets the property carried by a clone.</summary>
    public PropertyInfo Property { get; }

    /// <summary>Gets the manner in which the value is carried.</summary>
    public CloneCopyKind Copy { get; }
}

/// <summary>
/// How a single configuration value is carried to a clone.
/// </summary>
public enum CloneCopyKind
{
    /// <summary>
    /// Assign the same value. Correct for numbers, strings, enums, and for immutable or stateless
    /// objects such as activation functions and kernels, where sharing one instance between the
    /// original and the clone changes nothing observable.
    /// </summary>
    /// <remarks>
    /// Activation functions, kernels and schedules are supplied by callers as delegates and
    /// interfaces, and they <i>are</i> configuration: a clone that dropped them would behave
    /// differently while looking correct. Carrying them by reference is both correct and cheap.
    /// </remarks>
    ByReference = 0,

    /// <summary>
    /// Duplicate the container so the two instances do not write through the same buffer.
    /// </summary>
    /// <remarks>
    /// A bare assignment of a list or an array leaves the clone and the original sharing storage,
    /// so mutating one silently reconfigures the other. That is invisible to a property-by-property
    /// equality check and is why the round-trip tests also assert that mutating a clone cannot
    /// affect its original.
    /// </remarks>
    Deep = 1,
}
