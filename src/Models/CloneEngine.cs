using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

namespace AiDotNet.Models;

/// <summary>
/// Executes a <see cref="ClonePlan"/>: reconstructs an instance and carries its configuration.
/// </summary>
/// <remarks>
/// <para>
/// This is the single implementation of "copy the configuration" in the library. Everything that
/// clones routes through it, so a property cannot be carried correctly in one place and dropped in
/// another — which is the failure that 1802 hand-written clone paths made possible.
/// </para>
/// <para>
/// <b>Reconstruction rather than field copying.</b> A fresh instance is created and the plan's
/// entries are applied to it. Anything not in the plan is therefore whatever the constructor
/// produced, not a stale value carried over — so a derived or cached property is re-derived rather
/// than duplicated. scikit-learn's <c>clone()</c> works this way for the same reason; the
/// difference here is that the plan is generated and checked at compile time instead of relying on
/// a constructor convention verified only at test time.
/// </para>
/// </remarks>
public static class CloneEngine
{
    /// <summary>
    /// Creates a configuration copy of <paramref name="source"/>.
    /// </summary>
    /// <param name="source">The instance to copy.</param>
    /// <returns>A new instance of the same runtime type carrying the same configuration.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when the runtime type cannot be constructed without arguments.
    /// </exception>
    public static object CopyConfiguration(object source)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));

        var type = source.GetType();
        var clone = Construct(type);
        var pending = new List<(ClonePlanEntry Entry, object? Value)>();

        foreach (var entry in CloneRegistry.GetPlan(type).Entries)
        {
            object? value;
            try
            {
                value = entry.Property.GetValue(source);
            }
            catch (TargetInvocationException ex)
            {
                // A getter that computes rather than returns. MultilayerPerceptronOptions.Optimizer
                // builds a default model on first read, so cloning triggers that construction and
                // any failure inside it surfaces here with a stack trace pointing at ModelHelper,
                // giving no sign that cloning caused it. Naming the type and property converts an
                // unrelated-looking exception into one that says where to look.
                throw new InvalidOperationException(
                    $"Reading {type.Name}.{entry.Property.Name} while cloning threw "
                    + $"{ex.InnerException?.GetType().Name ?? ex.GetType().Name}: "
                    + $"{ex.InnerException?.Message ?? ex.Message}. A property whose getter computes "
                    + "or lazily constructs is not configuration; mark it [NotConfiguration] so a "
                    + "clone re-derives it instead of reading it.",
                    ex.InnerException ?? ex);
            }

            pending.Add((entry, value));
        }

        Assign(type, clone, pending);
        return clone;
    }

    /// <summary>
    /// Applies values in repeated passes until none remain or no pass makes progress.
    /// </summary>
    /// <param name="type">The type being cloned, for error messages.</param>
    /// <param name="clone">The instance being populated.</param>
    /// <param name="pending">The values to apply.</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when a pass assigns nothing and values remain, naming each stuck property.
    /// </exception>
    /// <remarks>
    /// <para>
    /// A setter may validate against ANOTHER property, which makes a single ordered pass unsound:
    /// <c>TabTransformerOptions.NumHeads</c> requires that it divide <c>EmbeddingDimension</c>, so
    /// assigning it before the dimension is carried checks it against the constructor default
    /// instead. No fixed order fixes this in general, since two properties can each constrain the
    /// other.
    /// </para>
    /// <para>
    /// Retrying works because the original object is internally consistent: some order satisfies
    /// every constraint, and repeating until nothing more succeeds finds one without the engine
    /// needing to know what the constraints are. A pass that assigns nothing while values remain is
    /// a genuine circular constraint rather than a missed ordering, and is reported as such.
    /// </para>
    /// </remarks>
    private static void Assign(Type type, object clone, List<(ClonePlanEntry Entry, object? Value)> pending)
    {
        var failures = new Dictionary<string, string>(StringComparer.Ordinal);

        while (pending.Count > 0)
        {
            var remaining = new List<(ClonePlanEntry Entry, object? Value)>();
            failures.Clear();

            foreach (var (entry, value) in pending)
            {
                try
                {
                    entry.Property.SetValue(
                        clone, entry.Copy == CloneCopyKind.Deep ? Duplicate(value) : value);
                }
                catch (TargetInvocationException ex)
                {
                    remaining.Add((entry, value));
                    failures[entry.Property.Name] =
                        ex.InnerException?.Message ?? ex.Message;
                }
            }

            if (remaining.Count == pending.Count)
            {
                throw new InvalidOperationException(
                    $"Cloning {type.Name} could not assign "
                    + string.Join(", ", failures.Select(f => $"{f.Key} ({f.Value})"))
                    + ". Each setter rejected a value the original already holds, and no assignment "
                    + "order satisfies them, so the constraints between these properties are "
                    + "circular.");
            }

            pending = remaining;
        }
    }

    /// <summary>
    /// Creates an instance without invoking configuration logic.
    /// </summary>
    /// <param name="type">The type to construct.</param>
    /// <returns>The new instance.</returns>
    /// <exception cref="InvalidOperationException">Thrown when no argument-less construction exists.</exception>
    /// <remarks>
    /// A non-public parameterless constructor is accepted deliberately: a type may reasonably keep
    /// one private so that callers use a factory, and that is a statement about how the type should
    /// be *used*, not a reason a clone cannot reproduce it.
    /// </remarks>
    private static object Construct(Type type)
    {
        var constructor = type.GetConstructor(
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
            binder: null,
            Type.EmptyTypes,
            modifiers: null);

        if (constructor is null)
        {
            throw new InvalidOperationException(
                $"{type.Name} cannot be cloned because it has no parameterless constructor. "
                + "Add one (it may be private), or override the clone behaviour on the type.");
        }

        return constructor.Invoke(null);
    }

    /// <summary>
    /// Duplicates a mutable container so the copy and the original do not share storage.
    /// </summary>
    /// <param name="value">The value to duplicate.</param>
    /// <returns>A duplicate, or the original when it is null or not a recognised container.</returns>
    /// <remarks>
    /// <para>
    /// The copy is one level deep, which matches what the plan promises. A list of mutable objects
    /// yields a new list holding the same elements: the two instances can no longer add or remove
    /// independently of one another, which is the sharing bug this addresses, while the elements
    /// themselves stay shared. Elements needing their own copies are configuration in their own
    /// right and get their own plans.
    /// </para>
    /// <para>
    /// A null container stays null rather than becoming an empty one. "Not configured" and
    /// "configured to be empty" are different states, and a clone must not quietly convert one into
    /// the other.
    /// </para>
    /// </remarks>
    private static object? Duplicate(object? value)
    {
        switch (value)
        {
            case null:
                return null;

            case Array array:
                return array.Clone();

            case IDictionary dictionary:
                return CopyInto(dictionary, Activator.CreateInstance(dictionary.GetType()));

            case IList list:
                return CopyInto(list, Activator.CreateInstance(list.GetType()));
        }

        // A set is neither IList nor IDictionary, so it is reached through its own Add.
        var type = value.GetType();
        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(HashSet<>))
        {
            var copy = Activator.CreateInstance(type, value);
            if (copy is not null) return copy;
        }

        return value;
    }

    private static object? CopyInto(IDictionary source, object? target)
    {
        if (target is not IDictionary typed) return source;

        foreach (DictionaryEntry entry in source)
        {
            typed[entry.Key] = entry.Value;
        }

        return typed;
    }

    private static object? CopyInto(IList source, object? target)
    {
        if (target is not IList typed) return source;

        foreach (var item in source)
        {
            typed.Add(item);
        }

        return typed;
    }
}
