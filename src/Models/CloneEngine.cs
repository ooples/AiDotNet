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
    /// Stands in a recorded constructor for "pass this parameter's declared default".
    /// </summary>
    /// <remarks>
    /// Not a member name -- no C# member can be called this -- so it cannot collide with one. The
    /// same literal is spelled out in <c>ClonePlanGenerator</c>, which lives in the analyzer assembly
    /// and cannot be referenced from here; changing it there requires changing it here.
    /// </remarks>
    internal const string UseDefault = "=default";

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
        var plan = CloneRegistry.GetPlan(type);
        var clone = Construct(type, plan, source);
        var pending = new List<(ClonePlanEntry Entry, object? Value)>();

        foreach (var entry in plan.Entries)
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
    private static object Construct(Type type, ClonePlan plan, object source)
    {
        // A type with recorded constructor parameters is rebuilt by CALLING that constructor with
        // its carried configuration, not by allocating and assigning. That matters because a
        // constructor derives things from its arguments -- weight buffers sized from InputSize,
        // sub-layers built from a depth setting -- and re-deriving them is what keeps a clone
        // consistent. Copying those structures instead would carry a stale derived value forward.
        // The generator only records parameters when it proved every one is supplied by a member of
        // the type -- a property, or the private field the constructor stored it in -- so this cannot
        // be partially satisfied: either the constructor is a pure function of state the instance
        // still holds, or nothing was recorded and the parameterless path below applies.
        // Each recorded constructor is tried in order, and the first one the INSTANCE can actually
        // satisfy wins. "Satisfy" means no required parameter -- one with no default -- would receive
        // null. That is what distinguishes a model loaded from an ONNX file, which has its path
        // stored, from one trained natively, which does not: taking the widest constructor
        // unconditionally passed null for onnxModelPath and made 51 models throw on clone.
        foreach (var candidate in plan.ConstructorCandidates)
        {
            var arguments = new object?[candidate.Count];
            var readable = true;

            for (int i = 0; i < arguments.Length; i++)
            {
                // The sentinel means the generator found nothing storing this OPTIONAL parameter, so
                // the constructor's own default is what it gets -- the same value the hand-written
                // override left it at. Type.Missing is how reflection spells that.
                if (candidate[i] == UseDefault) { arguments[i] = Type.Missing; continue; }

                if (!TryReadMember(type, candidate[i], source, out arguments[i])) { readable = false; break; }

                arguments[i] = DuplicateSubModel(arguments[i]);
            }

            if (!readable) continue;

            // Matched on parameter NAMES, not on how many there are. Overloads of equal arity are
            // ordinary -- a model taking (options, regularization) beside one taking
            // (options, lossFunction) -- and picking by count alone would pass each value to
            // whichever overload reflection happened to return first.
            // ARITY FIRST, NAMES ONLY TO BREAK A TIE. The plan records members in constructor-parameter
            // ORDER, so position already carries the mapping. Re-deriving it from names here could not
            // work for a member the generator sourced by TYPE rather than by name: BayesianRegression
            // takes bayesianOptions and stores it in _bayesOptions, which FindUniqueByType matched on
            // the type alone and no name rule can reproduce. The engine then rejected a member the
            // generator had certified, and reported "every member read, so the constructor could not
            // be matched by name" -- true, and beside the point.
            //
            // Arity is safe to rely on because the generator refuses to record two constructors of the
            // same arity: an ambiguous overload set is left unrecorded rather than guessed at. Names
            // still decide when several constructors share an arity in some future shape.
            var byArity = type.GetConstructors(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                .Where(c => c.GetParameters().Length == arguments.Length)
                .ToList();

            var withArgs = byArity.FirstOrDefault(c =>
            {
                var parameters = c.GetParameters();
                for (int i = 0; i < parameters.Length; i++)
                {
                    if (!NamesTheSameValue(parameters[i].Name, candidate[i])) return false;
                }

                return true;
            }) ?? (byArity.Count == 1 ? byArity[0] : null);

            if (withArgs is null) continue;

            var parameterInfos = withArgs.GetParameters();
            var satisfied = true;

            for (int i = 0; i < parameterInfos.Length; i++)
            {
                // A required parameter handed null is the signature of the wrong constructor for
                // this instance -- the value it wants was never stored because this object was not
                // built that way. An optional one is fine: its default is what it would have got.
                if (arguments[i] is null && !parameterInfos[i].HasDefaultValue)
                {
                    satisfied = false;
                    break;
                }

                // A member may hold the argument MORE GENERALLY than the constructor takes it: a
                // time series model keeps its ARModelOptions in the base's Options property, and its
                // own hand-written clone downcast on the way back in. The plan may now source such a
                // member, so the runtime value is what decides whether this constructor really fits
                // -- without this the call reaches Invoke and throws instead of moving on to the
                // next candidate, which is the whole point of recording more than one.
                if (arguments[i] is not null
                    && !ReferenceEquals(arguments[i], Type.Missing)
                    && !parameterInfos[i].ParameterType.IsInstanceOfType(arguments[i]))
                {
                    satisfied = false;
                    break;
                }
            }

            // OptionalParamBinding is what turns a Type.Missing slot into the declared default.
            // Without it the call throws, and it throws for every model with an unstored optional
            // parameter -- which is 307 of them.
            if (satisfied)
            {
                return withArgs.Invoke(
                    BindingFlags.OptionalParamBinding, binder: null, arguments, culture: null);
            }
        }

        var constructor = type.GetConstructor(
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance,
            binder: null,
            Type.EmptyTypes,
            modifiers: null);

        if (constructor is null)
        {
            // SAY WHAT ACTUALLY HAPPENED. This used to read "has no parameterless constructor. Add
            // one", which names a fix that is almost never the right one: reaching here usually means
            // the plan HAD a recorded constructor and this instance could not satisfy it, and adding
            // a parameterless constructor would paper over that by producing a default-configured
            // clone. The message cost a full investigation once; the candidates and the reason each
            // one declined are what a reader actually needs.
            var detail = plan.ConstructorCandidates.Count == 0
                ? "the clone plan recorded no constructor for it (see ADN0059)"
                : "none of its recorded constructors could be satisfied by this instance: "
                  + string.Join("; ", plan.ConstructorCandidates.Select(c => DescribeCandidate(type, c, source)));

            throw new InvalidOperationException(
                $"{type.Name} cannot be rebuilt: {detail}. It also has no parameterless constructor to "
                + "fall back on. Store each constructor argument in a member named after it so the "
                + "generator can replay the constructor.");
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
    /// <summary>
    /// Clones a constructor argument that is itself a model or a layer.
    /// </summary>
    /// <param name="value">The argument value read from the source.</param>
    /// <returns>Its clone, or the value unchanged when it is not something that clones itself.</returns>
    /// <remarks>
    /// <para>
    /// A constructor argument used to be handed straight across, which meant a rebuilt model SHARED
    /// its sub-modules with the one it was copied from. Nothing looks wrong at the moment of cloning
    /// -- the payload writes the same weights back into the same tensors -- and the damage appears
    /// later, when training the copy also trains the original. That is what roughly 180 hand-written
    /// <c>Clone</c> overrides in the diffusion family are working around when they call
    /// <c>_unet.Clone()</c> and <c>_vae.Clone()</c> before handing them to the constructor: the base
    /// could not do it, so each model did it again.
    /// </para>
    /// <para>
    /// Cheap, because these clones are copy-on-write: a layer's <c>Clone</c> shares the parent's
    /// weight tensors by reference and only materialises on write. So this buys independence without
    /// buying a second copy of a foundation-scale model, which is the reason it can be applied to
    /// every argument rather than to a hand-picked list of which sub-modules "count".
    /// </para>
    /// <para>
    /// Matched on the library's own <c>ICloneable&lt;T&gt;</c> rather than a list of base types, so a
    /// consumer's own module is duplicated on the same terms as one of ours. Anything that does not
    /// declare itself cloneable -- options, primitives, a shared frozen resource -- is passed across
    /// untouched, exactly as before.
    /// </para>
    /// </remarks>
    private static object? DuplicateSubModel(object? value)
    {
        if (value is null) return null;

        var cloneable = value.GetType().GetInterfaces().FirstOrDefault(i =>
            i.IsGenericType
            && i.GetGenericTypeDefinition().Name == "ICloneable`1"
            && i.Namespace == "AiDotNet.Interfaces");

        if (cloneable?.GetMethod("Clone", Type.EmptyTypes) is not { } clone) return value;

        // A sub-module that cannot clone itself is not a reason to abandon the rebuild -- the shared
        // reference is what happened before this existed, so falling back to it is no worse than the
        // behaviour this replaces, and the alternative is refusing to clone the parent at all.
        try
        {
            return clone.Invoke(value, null) ?? value;
        }
        catch (TargetInvocationException)
        {
            return value;
        }
    }

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

    /// <summary>
    /// Reads the value a recorded constructor parameter was built from.
    /// </summary>
    /// <param name="type">The runtime type being rebuilt.</param>
    /// <param name="member">The member name the plan recorded.</param>
    /// <param name="source">The instance being cloned.</param>
    /// <param name="value">Receives the value, or null when no such member exists.</param>
    /// <returns><see langword="true"/> when the member was found and read.</returns>
    /// <remarks>
    /// Private fields are in scope. A constructor argument that is not also exposed as a property is
    /// the normal case for a model -- a diffusion model's U-Net lives in <c>_unet</c> and nowhere
    /// else -- and refusing to read it would mean the only rebuildable models are the ones that
    /// happen to re-expose everything they were built from.
    /// </remarks>
    /// <summary>Explains, for one recorded constructor, why this instance could not satisfy it.</summary>
    /// <param name="type">The type being rebuilt.</param>
    /// <param name="candidate">The recorded member names, in constructor-parameter order.</param>
    /// <param name="source">The instance being cloned.</param>
    /// <returns>A phrase naming the first member that blocked it.</returns>
    private static string DescribeCandidate(Type type, IReadOnlyList<string> candidate, object source)
    {
        var names = string.Join(", ", candidate);

        for (int i = 0; i < candidate.Count; i++)
        {
            if (candidate[i] == UseDefault) continue;

            object? value;
            try
            {
                if (!TryReadMember(type, candidate[i], source, out value))
                {
                    return $"[{names}] -- '{candidate[i]}' is not readable on this type";
                }
            }
            catch (Exception ex)
            {
                var inner = ex.InnerException ?? ex;
                return $"[{names}] -- reading '{candidate[i]}' threw {inner.GetType().Name}: {inner.Message}";
            }

            if (value is null)
            {
                return $"[{names}] -- '{candidate[i]}' is null on this instance";
            }
        }

        return $"[{names}] -- every member read, so the constructor could not be matched by name";
    }

    private static bool TryReadMember(Type type, string member, object source, out object? value)
    {
        const BindingFlags Flags =
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;

        for (var current = type; current is not null; current = current.BaseType)
        {
            var property = current.GetProperty(member, Flags | BindingFlags.DeclaredOnly);
            if (property is not null && property.CanRead)
            {
                value = property.GetValue(source);
                return true;
            }

            var field = current.GetField(member, Flags | BindingFlags.DeclaredOnly);
            if (field is not null)
            {
                value = field.GetValue(source);
                return true;
            }
        }

        value = null;
        return false;
    }

    /// <summary>
    /// Determines whether a constructor parameter and a recorded member name denote the same value.
    /// </summary>
    /// <param name="parameter">The constructor parameter's name.</param>
    /// <param name="member">The member name the plan recorded.</param>
    /// <returns><see langword="true"/> when they correspond.</returns>
    /// <remarks>
    /// The recorded name is the member that holds the value, which is usually the parameter with a
    /// leading underscore. Comparing the two raw would reject <c>_unet</c> against <c>unet</c> and
    /// silently drop back to demanding a parameterless constructor, so the underscore is stripped
    /// before comparing.
    /// </remarks>
    private static bool NamesTheSameValue(string? parameter, string member)
    {
        if (parameter is null) return false;

        // The sentinel stands for the parameter's own default, so it matches whatever it sits against.
        if (member == UseDefault) return true;

        var trimmed = member.StartsWith("_", StringComparison.Ordinal) ? member.Substring(1) : member;
        if (string.Equals(parameter, trimmed, StringComparison.OrdinalIgnoreCase)) return true;

        // THE SUFFIX RULE, because the generator uses it when it sources the member. FindByNameSuffix
        // accepts a member whose name ENDS with the parameter's -- BayesianRegression keeps its
        // options in _bayesOptions, AttentiveNAS keeps its searchSpace in _nasSearchSpace -- and
        // records it in the plan. Matching only on equality here made the engine reject a member the
        // generator had just certified, so the plan was right and the rebuild refused it: "every
        // member read, so the constructor could not be matched by name". Two rules for one question
        // is one rule too many; this is the same one.
        // Decoration at EITHER end, for the same reason: the generator now also sources a member
        // whose name STARTS with the parameter's (StackingClassifier keeps its `finalEstimator`
        // factory in _finalEstimatorFactory). Accepting only the suffix here would recreate exactly
        // the split this comment warns about, with the plan certifying a member the rebuild refuses.
        return trimmed.Length > parameter.Length
            && (trimmed.EndsWith(parameter, StringComparison.OrdinalIgnoreCase)
                || trimmed.StartsWith(parameter, StringComparison.OrdinalIgnoreCase));
    }

}
