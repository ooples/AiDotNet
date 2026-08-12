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
    public static void Register(ClonePlan plan)
    {
        if (plan is null) throw new ArgumentNullException(nameof(plan));
        Generated[plan.Type] = plan;
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
        return new ClonePlan(closed, entries, open.ConstructorParameters);
    }

    private static int _generatedLoaded;

    /// <summary>
    /// Runs the generated registrations once, on first use.
    /// </summary>
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
        if (System.Threading.Interlocked.Exchange(ref _generatedLoaded, 1) != 0) return;

        var registrations = typeof(CloneRegistry).Assembly
            .GetType("AiDotNet.Generated.CloneRegistrations", throwOnError: false);

        registrations
            ?.GetMethod("RegisterAll", BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public)
            ?.Invoke(null, null);
    }

    /// <summary>
    /// Builds a plan for a type the generator never saw, applying the same rules it would.
    /// </summary>
    /// <param name="type">The type to plan for.</param>
    /// <returns>The reflected plan.</returns>
    /// <remarks>
    /// <para>
    /// The rule is that everything is configuration unless provably otherwise. A property is
    /// carried when it can be both read and written; a read-only or computed property is skipped
    /// because it is derived from the values that <i>are</i> carried, and re-deriving it is what
    /// keeps a clone consistent rather than merely equal.
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
                if (!property.CanRead || !property.CanWrite) continue;
                if (property.GetIndexParameters().Length > 0) continue;
                if (IsExcluded(property)) continue;
                if (!seen.Add(property.Name)) continue;

                entries.Add(new ClonePlanEntry(property, CopyKindFor(property.PropertyType)));
            }
        }

        entries.Reverse();
        return new ClonePlan(type, entries);
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
