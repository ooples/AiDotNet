using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace AiDotNet.Models;

/// <summary>
/// Holds the compile-time clone plan for every discovered type, and builds one on demand for types
/// the generator never saw.
/// </summary>
/// <remarks>
/// <para>
/// The clone plan generator registers a plan for every type it discovers, so correctness for those
/// is settled at compile time and enforced by the analyzer. Types the generator never saw — one
/// defined in a consumer's own assembly, or produced at runtime — fall back to reflection, which is
/// slower on first use but never simply fails.
/// </para>
/// <para>
/// The layering matters: a compile-time plan is checkable, a runtime plan is not, and mixing them
/// silently would leave nobody able to say which guarantees applied. <see cref="IsVerified"/>
/// reports which of the two produced a given plan.
/// </para>
/// </remarks>
public static class CloneRegistry
{
    private static readonly ConcurrentDictionary<Type, ClonePlan> Generated = new();
    private static readonly ConcurrentDictionary<Type, ClonePlan> Reflected = new();

    /// <summary>
    /// Registers a compile-time plan. Called by generated code.
    /// </summary>
    /// <param name="plan">The plan to register.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="plan"/> is null.</exception>
    /// <remarks>
    /// Any reflection plan already cached for this type is evicted. A reflected plan is a fallback
    /// for a type nothing registered, so the moment one IS registered the fallback is stale -- and
    /// <c>GetOrAdd</c> would otherwise keep serving it for the life of the process, silently
    /// preferring a plan with no constructor over the compile-time one that has it.
    /// </remarks>
    public static void Register(ClonePlan plan)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        Generated[plan.Type] = plan;

        Reflected.TryRemove(plan.Type, out _);
        if (plan.Type.IsGenericTypeDefinition)
        {
            // Closed forms were cached against the open form's absence, so they are stale too.
            foreach (var closed in Reflected.Keys)
            {
                if (closed.IsGenericType && !closed.IsGenericTypeDefinition
                    && closed.GetGenericTypeDefinition() == plan.Type)
                {
                    Reflected.TryRemove(closed, out _);
                }
            }
        }
    }

    /// <summary>
    /// Gets a value indicating whether a type's plan was produced at compile time, and so is covered
    /// by the analyzer and the generated round-trip test.
    /// </summary>
    /// <param name="type">The type to query.</param>
    /// <returns><see langword="true"/> when the plan is generated; <see langword="false"/> when it is reflected.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="type"/> is null.</exception>
    public static bool IsVerified(Type type)
    {
        if (type is null) throw new ArgumentNullException(nameof(type));

        EnsureGeneratedPlansLoaded();

        if (Generated.ContainsKey(type)) return true;

        // A closed generic whose open form is registered is still compile-time decided: the
        // property set and the copy kinds came from the generator, and only the PropertyInfo
        // handles were re-bound. Reporting it as unverified would understate the guarantee exactly
        // as badly as the reverse would overstate it.
        return type.IsGenericType
            && !type.IsGenericTypeDefinition
            && Generated.ContainsKey(type.GetGenericTypeDefinition());
    }

    /// <summary>
    /// Gets every type with a compile-time plan. Used by the generated round-trip tests.
    /// </summary>
    /// <returns>The verified types.</returns>
    public static IEnumerable<Type> VerifiedTypes() => Generated.Keys;

    /// <summary>
    /// Gets the plan for a type, building one by reflection if the generator never saw it.
    /// </summary>
    /// <param name="type">The type to plan for.</param>
    /// <returns>The plan.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="type"/> is null.</exception>
    public static ClonePlan GetPlan(Type type)
    {
        if (type is null) throw new ArgumentNullException(nameof(type));

        EnsureGeneratedPlansLoaded();

        if (Generated.TryGetValue(type, out var generated)) return generated;

        // A generic type is registered under its open form -- typeof(Foo<>) -- because that is the
        // only handle the generator can name. Without this, every closed Foo<double> missed its
        // generated plan and fell through to reflection, which is nearly every options and layer
        // type in the library: the compile-time guarantee existed but was not the one in force.
        if (type.IsGenericType && !type.IsGenericTypeDefinition)
        {
            var definition = type.GetGenericTypeDefinition();
            if (Generated.TryGetValue(definition, out var open))
            {
                return Reflected.GetOrAdd(type, t => Close(open, t));
            }
        }

        return Reflected.GetOrAdd(type, BuildByReflection);
    }

    /// <summary>
    /// Re-binds an open generic's plan against one of its closed forms.
    /// </summary>
    /// <param name="open">The plan registered for the generic type definition.</param>
    /// <param name="closed">The closed type to bind against.</param>
    /// <returns>A plan whose properties belong to <paramref name="closed"/>.</returns>
    /// <remarks>
    /// The entries carry a <see cref="PropertyInfo"/> obtained from the open definition, and such a
    /// handle cannot read or write an instance of a closed type. Only the property NAME and the
    /// copy kind survive re-binding; both were decided at compile time, so the result is still the
    /// generated decision rather than a rediscovered one.
    /// </remarks>
    private static ClonePlan Close(ClonePlan open, Type closed)
    {
        var entries = new List<ClonePlanEntry>(open.Entries.Count);

        foreach (var entry in open.Entries)
        {
            var property = closed.GetProperty(
                entry.Property.Name, BindingFlags.Public | BindingFlags.Instance);

            if (property is not null && property.CanRead && property.CanWrite)
            {
                entries.Add(new ClonePlanEntry(property, entry.Copy));
            }
        }

        // The recorded constructor travels with the plan. Dropping it here would be invisible and
        // total: every model and every layer is generic, so a plan that loses its constructor on
        // closing falls back to demanding a parameterless constructor -- which is precisely the
        // constructor these types do not have.
        return new ClonePlan(closed, entries, open.ConstructorParameters, open.ConstructorCandidates);
    }

    private static readonly object GeneratedGate = new();
    private static volatile bool _generatedLoaded;

    /// <summary>
    /// Runs the generated registrations once, on first use, and does not return until they are all
    /// present.
    /// </summary>
    /// <remarks>
    /// <para>
    /// THE FLAG IS SET AFTER THE WORK, NOT BEFORE. This was a single <c>Interlocked.Exchange</c> that
    /// claimed the flag and then ran <c>RegisterAll</c>, so a second caller arriving mid-registration
    /// saw "already loaded", missed a plan that was still on its way in, and fell through to
    /// <see cref="BuildByReflection"/>. That reflection plan then went into <c>Reflected</c> and
    /// STAYED there, because <c>GetOrAdd</c> keeps the first value it was given -- so the generated
    /// plan could never replace it, for the life of the process.
    /// </para>
    /// <para>
    /// The symptom was a type that cloned correctly when its test ran alone and failed when the suite
    /// ran, reporting "the clone plan recorded no constructor for it". That was true of the cached
    /// plan and false of the type, which is the worst kind of error message to be handed.
    /// </para>
    /// </remarks>
    /// <remarks>
    /// <para>
    /// Resolved by reflection rather than called directly so that this file still compiles when the
    /// generator produces nothing — during bootstrap, or in a consumer's assembly that references
    /// the library without running its generators. A direct call would make the runtime depend on
    /// generated output existing, which is exactly the kind of coupling that turns a missing
    /// generator into an unexplainable build failure.
    /// </para>
    /// <para>
    /// Absence is not an error: every type then falls back to a reflected plan, and
    /// <see cref="IsVerified"/> reports honestly that no compile-time plan was available.
    /// </para>
    /// </remarks>
    private static void EnsureGeneratedPlansLoaded()
    {
        if (_generatedLoaded) return;

        lock (GeneratedGate)
        {
            if (_generatedLoaded) return;

            var registrations = typeof(CloneRegistry).Assembly
                .GetType("AiDotNet.Generated.CloneRegistrations", throwOnError: false);

            registrations
                ?.GetMethod("RegisterAll", BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public)
                ?.Invoke(null, null);

            _generatedLoaded = true;
        }
    }

    /// <summary>
    /// Builds a plan for a type the generator never saw, applying the same rules it would.
    /// </summary>
    /// <param name="type">The type to plan for.</param>
    /// <returns>The reflected plan.</returns>
    /// <remarks>
    /// <para>
    /// The rule is that everything is configuration unless provably otherwise. A property is
    /// carried when it can be publicly read and written; a read-only, privately set, or computed
    /// property is skipped because it is constructor-owned or derived from the values that
    /// <i>are</i> carried, and re-deriving it is what keeps a clone consistent rather than merely
    /// equal.
    /// </para>
    /// <para>
    /// Deliberately <b>not</b> excluded by type shape: delegates and interfaces are carried, since
    /// activation functions, kernels and schedules arrive that way and are genuine configuration.
    /// Excluding them by shape would produce a clone that behaves differently while looking right.
    /// </para>
    /// </remarks>
    private static ClonePlan BuildByReflection(Type type)
    {
        var entries = new List<ClonePlanEntry>();
        var seen = new HashSet<string>(StringComparer.Ordinal);

        // Walk the inheritance chain explicitly. GetProperties() on a derived type does surface
        // inherited members, but walking the chain keeps the order stable from base to derived and
        // makes the inherited surface visible rather than implied -- the surface whose omission
        // dropped ModelOptions.Seed from 71 hand-written copy constructors.
        for (var current = type; current is not null && current != typeof(object); current = current.BaseType)
        {
            var declared = current.GetProperties(
                BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);

            foreach (var property in declared.OrderBy(p => p.Name, StringComparer.Ordinal))
            {
                if (!property.CanRead || property.SetMethod?.IsPublic != true) continue;
                if (property.GetIndexParameters().Length > 0) continue;
                if (IsExcluded(property)) continue;
                if (!seen.Add(property.Name)) continue;

                entries.Add(new ClonePlanEntry(property, CopyKindFor(property.PropertyType)));
            }
        }

        entries.Reverse();
        return new ClonePlan(
            type,
            entries,
            constructorParameters: null,
            constructorCandidates: BuildConstructorCandidates(type));
    }

    /// <summary>
    /// Derives the constructors a type can be rebuilt through, for a type the generator never saw.
    /// </summary>
    /// <param name="type">The type to plan for.</param>
    /// <returns>The candidates, widest first, or <see langword="null"/> when none were derived.</returns>
    /// <remarks>
    /// <para>
    /// A reflected plan used to carry properties and nothing else, so <c>CloneEngine.Construct</c>
    /// found no candidate and fell through to demanding a parameterless constructor. That made a
    /// model the generator cannot see -- one declared in a consumer's own assembly, or a distribution
    /// the generator skips -- cloneable only if it happened to have one, which is precisely the
    /// "write your own model and everything generic just works" promise failing at the assembly
    /// boundary. <c>GammaDistribution</c> and a test's own network subclass both died on it, with an
    /// error telling the author to store constructor arguments in members they had already stored
    /// them in.
    /// </para>
    /// <para>
    /// Only derived when there is NO parameterless constructor. Where one exists, allocate-and-assign
    /// is the path that has always run and the one every options object and layer relies on; taking a
    /// derived constructor instead would re-route thousands of working clones through new code to fix
    /// a case that is not broken. This fills the hole and touches nothing else.
    /// </para>
    /// <para>
    /// The rule is the generator's rule, applied at runtime: a parameter is supplied when a property
    /// or field holds it -- matched by name, by name with a leading underscore, or by the suffix rule
    /// that lets <c>_bayesOptions</c> supply <c>options</c> -- and its type fits. A parameter nothing
    /// supplies falls back to its own declared default, and a REQUIRED parameter nothing supplies
    /// disqualifies that constructor rather than being handed null.
    /// </para>
    /// </remarks>
    private static IReadOnlyList<IReadOnlyList<string>>? BuildConstructorCandidates(Type type)
    {
        const BindingFlags Flags =
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;

        var parameterless = type.GetConstructor(Flags, binder: null, Type.EmptyTypes, modifiers: null);
        if (parameterless is not null) return null;

        var candidates = new List<IReadOnlyList<string>>();

        // Widest first, matching what the plan promises. The engine still picks the first one this
        // INSTANCE can satisfy, so a narrow constructor wins whenever the wide one wants something
        // the object never stored.
        foreach (var constructor in type.GetConstructors(Flags)
            .OrderByDescending(c => c.GetParameters().Length))
        {
            var parameters = constructor.GetParameters();
            if (parameters.Length == 0) continue;

            var members = new string[parameters.Length];
            var recordable = true;

            for (int i = 0; i < parameters.Length; i++)
            {
                var member = FindSupplyingMember(type, parameters[i]);
                if (member is not null) { members[i] = member; continue; }
                if (parameters[i].HasDefaultValue) { members[i] = CloneEngine.UseDefault; continue; }

                recordable = false;
                break;
            }

            if (recordable) candidates.Add(members);
        }

        return candidates.Count > 0 ? candidates : null;
    }

    /// <summary>
    /// Finds the member holding the value a constructor parameter was built from.
    /// </summary>
    /// <param name="type">The type being planned for.</param>
    /// <param name="parameter">The constructor parameter to supply.</param>
    /// <returns>The member's name, or <see langword="null"/> when nothing holds it.</returns>
    /// <remarks>
    /// An exact name beats a suffix match wherever both exist, so a type holding both <c>_options</c>
    /// and <c>_bayesOptions</c> supplies <c>options</c> from the one actually named after it. The type
    /// must fit as well as the name: a field called <c>_seed</c> holding a random generator does not
    /// supply an <c>int seed</c>, and matching on the name alone would pass it and throw inside the
    /// constructor rather than declining the candidate here.
    /// </remarks>
    private static string? FindSupplyingMember(Type type, ParameterInfo parameter)
    {
        const BindingFlags Flags =
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly;

        if (parameter.Name is not { } name) return null;

        string? bySuffix = null;

        for (var current = type; current is not null && current != typeof(object); current = current.BaseType)
        {
            foreach (var property in current.GetProperties(Flags))
            {
                if (!property.CanRead || property.GetIndexParameters().Length > 0) continue;

                switch (Supplies(property.Name, name, property.PropertyType, parameter.ParameterType))
                {
                    case MemberMatch.Exact: return property.Name;
                    case MemberMatch.Suffix: bySuffix ??= property.Name; break;
                }
            }

            foreach (var field in current.GetFields(Flags))
            {
                switch (Supplies(field.Name, name, field.FieldType, parameter.ParameterType))
                {
                    case MemberMatch.Exact: return field.Name;
                    case MemberMatch.Suffix: bySuffix ??= field.Name; break;
                }
            }
        }

        return bySuffix ?? FindUniqueByType(type, parameter.ParameterType);
    }

    /// <summary>
    /// Finds the one member of a parameter's exact type, when there is exactly one.
    /// </summary>
    /// <param name="type">The type being planned for.</param>
    /// <param name="parameterType">The constructor parameter's type.</param>
    /// <returns>That member's name, or <see langword="null"/> when there is not exactly one.</returns>
    /// <remarks>
    /// <para>
    /// The same rule <c>ClonePlanGenerator.FindUniqueByType</c> applies at compile time, and it is
    /// here for the same reason: a constructor parameter is routinely stored under a name no rule
    /// guesses. A subclass taking <c>arch</c> and handing it to a base that keeps it in
    /// <c>Architecture</c> stores the value perfectly well; "Architecture" simply does not end in
    /// "arch". Declining there would make a model unrebuildable over a naming choice.
    /// </para>
    /// <para>
    /// EXACTLY ONE, and by exact type, both as the generator has it. Two members of a type would bind
    /// in declaration order and could silently swap one for the other. Primitives and enums are
    /// excluded because a lone int matching a lone int parameter is a coincidence, not a
    /// correspondence -- keeping the runtime rule and the compile-time rule the same one.
    /// </para>
    /// </remarks>
    private static string? FindUniqueByType(Type type, Type parameterType)
    {
        if (parameterType.IsPrimitive || parameterType.IsEnum
            || parameterType == typeof(string) || parameterType == typeof(decimal)
            || parameterType == typeof(DateTime) || parameterType == typeof(object))
        {
            return null;
        }

        const BindingFlags Flags =
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly;

        string? found = null;

        for (var current = type; current is not null && current != typeof(object); current = current.BaseType)
        {
            foreach (var property in current.GetProperties(Flags))
            {
                if (!property.CanRead || property.GetIndexParameters().Length > 0) continue;
                if (property.PropertyType != parameterType) continue;
                if (found is not null) return null;

                found = property.Name;
            }

            foreach (var field in current.GetFields(Flags))
            {
                if (field.FieldType != parameterType) continue;
                if (found is not null) return null;

                found = field.Name;
            }
        }

        return found;
    }

    private enum MemberMatch { None, Suffix, Exact }

    /// <summary>Decides whether a member can supply a constructor parameter.</summary>
    private static MemberMatch Supplies(string member, string parameter, Type memberType, Type parameterType)
    {
        if (!parameterType.IsAssignableFrom(memberType)) return MemberMatch.None;

        var trimmed = member.StartsWith("_", StringComparison.Ordinal) ? member.Substring(1) : member;
        if (string.Equals(trimmed, parameter, StringComparison.OrdinalIgnoreCase)) return MemberMatch.Exact;

        return trimmed.Length > parameter.Length
            && trimmed.EndsWith(parameter, StringComparison.OrdinalIgnoreCase)
                ? MemberMatch.Suffix
                : MemberMatch.None;
    }

    /// <summary>
    /// Determines whether a property is explicitly excluded from configuration.
    /// </summary>
    /// <param name="property">The property to test.</param>
    /// <returns><see langword="true"/> when the property carries an exclusion attribute.</returns>
    /// <remarks>
    /// Matched by name so that the runtime does not have to reference the attribute assembly, and
    /// so a consumer can define their own equivalents. Keeping the escape hatch small and named
    /// makes every exclusion greppable, which is the point of having one.
    /// </remarks>
    private static bool IsExcluded(PropertyInfo property)
        => property.GetCustomAttributes(inherit: true)
            .Select(a => a.GetType().Name)
            .Any(n => n is "NotConfigurationAttribute" or "ExternalResourceAttribute");

    /// <summary>
    /// Chooses how a value of the given type is carried.
    /// </summary>
    /// <param name="type">The property type.</param>
    /// <returns>The copy kind.</returns>
    /// <remarks>
    /// Mutable containers are duplicated so the two instances cannot write through one buffer.
    /// A string is a reference type but immutable, so sharing it is safe and copying it would be
    /// waste; the same reasoning covers activation functions and other stateless strategy objects.
    /// </remarks>
    private static CloneCopyKind CopyKindFor(Type type)
    {
        if (type == typeof(string)) return CloneCopyKind.ByReference;
        if (type.IsArray) return CloneCopyKind.Deep;

        if (type.IsGenericType)
        {
            var definition = type.GetGenericTypeDefinition();
            if (definition == typeof(List<>)
                || definition == typeof(Dictionary<,>)
                || definition == typeof(HashSet<>)
                || definition == typeof(IList<>)
                || definition == typeof(ICollection<>))
            {
                return CloneCopyKind.Deep;
            }
        }

        return CloneCopyKind.ByReference;
    }
}
