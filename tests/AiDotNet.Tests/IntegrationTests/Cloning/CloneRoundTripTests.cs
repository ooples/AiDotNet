using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.AutoML;
using AiDotNet.Models;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests.Cloning;

/// <summary>
/// Proves that a clone carries every configuration property, and that the two instances are
/// genuinely independent afterwards.
/// </summary>
/// <remarks>
/// <para>
/// Two assertions, because there are two distinct failure modes and neither one catches the other.
/// A property that is never carried shows up as unequal values. A property carried by sharing a
/// mutable container shows up as <i>equal</i> values that change together — invisible to a
/// property-by-property comparison, and exactly the bug where configuring a clone silently
/// reconfigures the original.
/// </para>
/// <para>
/// Every property is set to a value distinguishable from its default first. Comparing two freshly
/// constructed objects proves nothing: their properties already agree, so a clone that carried
/// nothing at all would pass.
/// </para>
/// </remarks>
public class CloneRoundTripTests
{
    private readonly ITestOutputHelper _output;

    /// <summary>Initializes a new instance of the <see cref="CloneRoundTripTests"/> class.</summary>
    /// <param name="output">Sink for the coverage summary.</param>
    public CloneRoundTripTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void SerializationShell_DeclinesValueInvalidConstructorCandidate()
    {
        var original = new AutoMLEnsembleModel<double>();

        var clone = Assert.IsType<AutoMLEnsembleModel<double>>(
            CloneEngine.CopyConfiguration(original));

        Assert.Empty(clone.Members);
        Assert.Empty(clone.Weights);
        Assert.Equal(original.PredictionType, clone.PredictionType);
    }

    /// <summary>
    /// Round-trips every type holding a compile-time clone plan.
    /// </summary>
    /// <returns>A task representing the test.</returns>
    /// <remarks>
    /// Data-driven over the registry rather than one test method per type, so the suite gains a
    /// single method rather than thousands — the full suite already runs 4731 tests and times out
    /// under load. A failure still names the type and the property, so diagnosis is unaffected.
    /// </remarks>
    [Fact(Timeout = 600000)]
    public async Task EveryPlannedType_RoundTripsAndStaysIndependent()
    {
        await Task.Yield();

        // Touch the registry so the generated registrations load before enumeration.
        _ = CloneRegistry.GetPlan(typeof(CloneRoundTripTests));

        var types = CloneRegistry.VerifiedTypes().ToList();
        Assert.True(types.Count > 0, "No generated clone plans were registered.");

        var failures = new List<string>();
        int cloned = 0, skipped = 0;

        foreach (var registered in types)
        {
            if (registered.IsAbstract)
            {
                skipped++;
                continue;
            }

            // The registry keys an open generic as typeof(Foo<>), which is what lets a closed
            // instantiation resolve through it -- but Activator cannot instantiate an unbound
            // generic. Nearly every options and layer type in this library is generic over its
            // numeric type, so skipping them left two thirds of the planned types unexercised
            // while the run still reported success.
            var type = Close(registered);
            if (type is null)
            {
                skipped++;
                continue;
            }

            object original;
            try
            {
                original = Activator.CreateInstance(type)!;
            }
            catch (Exception ex) when (ex is MissingMethodException or TargetInvocationException
                or ArgumentException or NotSupportedException)
            {
                skipped++;
                continue;
            }

            // Looked up by the CLOSED type: a PropertyInfo obtained from an open generic type
            // definition cannot read or write an instance of a closed one. Until GetPlan resolves
            // an open-generic registration against the closed type it is asked about, a closed
            // generic falls through to the reflected plan -- so this exercises the engine and the
            // fallback, but not yet the generated plan for these types.
            var plan = CloneRegistry.GetPlan(type);
            var populated = Populate(original, plan);

            object clone;
            try
            {
                clone = CloneEngine.CopyConfiguration(original);
            }
            catch (Exception ex)
            {
                Exception cause = ex;
                while (cause.InnerException is not null) cause = cause.InnerException;
                failures.Add($"{type.Name}: clone threw {cause.GetType().Name}: {cause.Message}");
                continue;
            }

            cloned++;
            CheckCarried(type, plan, original, clone, populated, failures);
            CheckIndependent(type, plan, original, clone, populated, failures);
        }

        _output.WriteLine($"planned types : {types.Count}");
        _output.WriteLine($"cloned        : {cloned}");
        _output.WriteLine($"skipped       : {skipped} (abstract, open generic, or not constructible)");
        _output.WriteLine($"failures      : {failures.Count}");

        foreach (var failure in failures.Take(40))
        {
            _output.WriteLine("  " + failure);
        }

        Assert.True(
            failures.Count == 0,
            $"{failures.Count} of {cloned} cloned types failed. First: "
            + string.Join(" | ", failures.Take(5)));
    }

    /// <summary>
    /// Closes an open generic with a concrete numeric argument so it can be instantiated.
    /// </summary>
    /// <param name="type">A registered type, open or closed.</param>
    /// <returns>A constructible type, or null when no argument satisfies its constraints.</returns>
    /// <remarks>
    /// <c>double</c> is used because these types are generic over their numeric type and are
    /// overwhelmingly exercised at <c>double</c> in practice, so the closed form under test is the
    /// one users actually construct. A type whose constraints reject it is reported as skipped
    /// rather than quietly passed.
    /// </remarks>
    private static Type? Close(Type type)
    {
        if (!type.ContainsGenericParameters) return type;

        var parameters = type.GetGenericArguments();

        // Arguments are chosen by parameter NAME, not filled uniformly. These types are generic
        // over a numeric type plus the shapes it operates on, so <double, double, double> is not
        // merely unusual -- it is a combination the library explicitly rejects, and constructing it
        // produced an exception from deep inside ModelHelper that looked like an engine fault.
        var candidates = new List<Type[]>();
        var byName = new Type[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            byName[i] = parameters[i].Name switch
            {
                "TInput" => typeof(Matrix<double>),
                "TOutput" => typeof(Vector<double>),
                _ => typeof(double),
            };
        }

        candidates.Add(byName);

        // The supported pairings, in the order the library lists them, for parameters whose names
        // carry no hint.
        if (parameters.Length == 3)
        {
            candidates.Add(new[] { typeof(double), typeof(Matrix<double>), typeof(Vector<double>) });
            candidates.Add(new[] { typeof(double), typeof(Tensor<double>), typeof(Tensor<double>) });
            candidates.Add(new[] { typeof(double), typeof(Vector<double>), typeof(Vector<double>) });
        }
        else if (parameters.Length == 1)
        {
            candidates.Add(new[] { typeof(double) });
        }

        foreach (var arguments in candidates)
        {
            if (arguments.Length != parameters.Length) continue;

            try
            {
                return type.MakeGenericType(arguments);
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
            {
                // Constraints reject this shape; try the next.
            }
        }

        return null;
    }

    /// <summary>Asserts that each populated property survived the clone.</summary>
    private static void CheckCarried(
        Type type, ClonePlan plan, object original, object clone,
        ISet<string> populated, ICollection<string> failures)
    {
        foreach (var entry in plan.Entries.Where(e => populated.Contains(e.Property.Name)))
        {
            object? before, after;
            try
            {
                before = entry.Property.GetValue(original);
                after = entry.Property.GetValue(clone);
            }
            catch (Exception)
            {
                continue;
            }

            if (!ValuesMatch(before, after))
            {
                failures.Add($"{type.Name}.{entry.Property.Name}: not carried ({Describe(before)} -> {Describe(after)})");
            }
        }
    }

    /// <summary>
    /// Asserts that mutating the clone's containers cannot reach the original.
    /// </summary>
    /// <remarks>
    /// This is the assertion a property-by-property comparison cannot make. Two properties holding
    /// the same list are equal on every check and still wrong.
    /// </remarks>
    private static void CheckIndependent(
        Type type, ClonePlan plan, object original, object clone,
        ISet<string> populated, ICollection<string> failures)
    {
        foreach (var entry in plan.Entries.Where(e => e.Copy == CloneCopyKind.Deep))
        {
            if (!populated.Contains(entry.Property.Name)) continue;

            if (entry.Property.GetValue(original) is not { } before) continue;
            if (entry.Property.GetValue(clone) is not { } after) continue;

            if (ReferenceEquals(before, after))
            {
                failures.Add(
                    $"{type.Name}.{entry.Property.Name}: clone shares the original's instance, "
                    + "so mutating one reconfigures the other");
            }
        }
    }

    /// <summary>
    /// Sets every plan property to a value distinguishable from its default.
    /// </summary>
    /// <returns>The names actually populated; only those can be meaningfully asserted on.</returns>
    private static ISet<string> Populate(object target, ClonePlan plan)
    {
        var populated = new HashSet<string>(StringComparer.Ordinal);

        foreach (var entry in plan.Entries)
        {
            // The probe models what a creator can configure. Reflecting through a private setter
            // can manufacture states the public API and every constructor reject -- for example a
            // 39-voxel grid paired with eleven pooling blocks. Such a derived property remains in
            // the plan so legitimate internal state can be carried, but it is not independently
            // mutated by this public-configuration census.
            if (entry.Property.SetMethod?.IsPublic != true) continue;

            object? current;
            try
            {
                current = entry.Property.GetValue(target);
            }
            catch (Exception)
            {
                // A getter that computes rather than returns; excluded from the probe rather than
                // allowed to abort the whole run before it reports anything.
                continue;
            }

            var value = SampleValue(entry.Property.PropertyType, current);
            if (value is null) continue;

            try
            {
                entry.Property.SetValue(target, value);
                populated.Add(entry.Property.Name);
            }
            catch (Exception ex) when (ex is TargetInvocationException or ArgumentException)
            {
                // A validating setter rejecting a sample is a property this test cannot exercise,
                // not a clone defect. Excluded from the assertions rather than reported as one.
            }
        }

        return populated;
    }

    /// <summary>Produces a value of the given type that differs from <paramref name="current"/>.</summary>
    private static object? SampleValue(Type type, object? current)
    {
        var underlying = Nullable.GetUnderlyingType(type) ?? type;

        if (underlying.IsEnum)
        {
            var values = Enum.GetValues(underlying);
            foreach (var candidate in values)
            {
                if (!Equals(candidate, current)) return candidate;
            }

            return null;
        }

        if (underlying == typeof(bool)) return !(current as bool? ?? false);
        if (underlying == typeof(string)) return "clone-round-trip-probe";
        if (underlying == typeof(int)) return (current as int? ?? 0) + 7;
        if (underlying == typeof(long)) return (current as long? ?? 0L) + 7L;
        if (underlying == typeof(double)) return (current as double? ?? 0d) + 0.375d;
        if (underlying == typeof(float)) return (current as float? ?? 0f) + 0.375f;
        if (underlying == typeof(decimal)) return (current as decimal? ?? 0m) + 0.375m;

        if (underlying.IsArray && underlying.GetElementType() is { } element && element.IsValueType)
        {
            var array = Array.CreateInstance(element, 2);
            var sample = SampleValue(element, null);
            if (sample is null) return null;
            array.SetValue(sample, 0);
            array.SetValue(sample, 1);
            return array;
        }

        if (underlying.IsGenericType && underlying.GetGenericTypeDefinition() == typeof(List<>))
        {
            var element2 = underlying.GetGenericArguments()[0];
            var sample = SampleValue(element2, null);
            if (sample is null) return null;

            var list = (IList)Activator.CreateInstance(underlying)!;
            list.Add(sample);
            return list;
        }

        // Reference types with no obvious sample are left alone rather than guessed at. They are
        // still covered by the carried-check when a default happens to be non-null.
        return null;
    }

    private static bool ValuesMatch(object? a, object? b)
    {
        if (a is null || b is null) return ReferenceEquals(a, b);
        if (a is IEnumerable ea and not string && b is IEnumerable eb and not string)
        {
            return ea.Cast<object?>().SequenceEqual(eb.Cast<object?>());
        }

        return Equals(a, b);
    }

    private static string Describe(object? value) => value switch
    {
        null => "null",
        string s => $"\"{s}\"",
        IEnumerable e and not string => "[" + string.Join(",", e.Cast<object?>().Take(4)) + "]",
        _ => value.ToString() ?? "?",
    };
}
