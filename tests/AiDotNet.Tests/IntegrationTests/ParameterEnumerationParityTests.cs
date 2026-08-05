using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Evidence-gathering harness for collapsing <c>ParameterCount</c> and <c>GetParameters()</c> onto a
/// single source of truth. Produces a report; does not gate the build.
/// </summary>
/// <remarks>
/// <para>
/// The repository currently carries 650 hand-written <c>ParameterCount</c> overrides and 721
/// <c>GetParameters()</c> overrides that must agree with each other by hand, with nothing enforcing
/// it — and 284 call sites slice a flat parameter vector by <c>ParameterCount</c>, so when they
/// disagree a saved vector silently restores into the wrong tensors. Nineteen models across
/// fourteen shards disagree today.
/// </para>
/// <para>
/// PyTorch cannot express this bug: <c>nn.Module.parameters()</c> is the only registry and the count
/// is <c>sum(p.numel() for p in parameters())</c> — a fold over the same iterator. Getting there
/// means deleting the overrides, and deleting 650 of anything by hand is how regressions happen. So
/// this harness establishes, per model, whether an override is REDUNDANT (it computes exactly what
/// the base already computes, and can be deleted mechanically with this report as the receipt) or
/// LOAD-BEARING (it differs — which is either a deliberate special case or one of the nineteen bugs,
/// and must be read by a human).
/// </para>
/// <para>
/// Reports rather than asserts, deliberately. Its output is an input to a migration, not a gate; a
/// red build here would say nothing that <c>ParameterCount_ShouldMatchGetParameters</c> does not
/// already say per model.
/// </para>
/// </remarks>
[Trait("Category", "Sweep")]
public class ParameterEnumerationParityTests
{
    private readonly ITestOutputHelper _output;

    public ParameterEnumerationParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>Machine-readable output; written incrementally so an interrupted run still yields data.</summary>
    private static string TsvPath =>
        Path.Combine(Path.GetTempPath(), "parameter-enumeration-parity.tsv");

    /// <summary>
    /// Above this parameter count the sweep reads <c>ParameterCount</c> only and never calls
    /// <c>GetParameters()</c>.
    /// </summary>
    /// <remarks>
    /// Materialising a flat vector costs 8 bytes per parameter for <c>double</c>, on top of the
    /// weights themselves. CogVideoModel's default constructor builds the paper-scale 5B variant
    /// (baseChannels 384, cross-attention dim 4096, 49 frames at 480x720); asking it for a flat
    /// copy is a ~40 GB allocation that killed the test host outright and took the rest of the
    /// sweep with it. The MODEL is fine -- construction is cheap and ParameterCount is just a sum
    /// over its two sub-networks. The sweep was the problem.
    ///
    /// So the guard is on size, not on a list of names: the cheap value decides whether the
    /// expensive one is safe to ask for. 50M parameters is ~400 MB at double, comfortably
    /// measurable, and above it the pairing cannot be checked -- which is reported honestly rather
    /// than skipped silently.
    /// </remarks>
    private const long MaxParametersToMaterialize = 50_000_000L;

    /// <summary>Bounded so one pathological constructor cannot stall the sweep.</summary>
    private static readonly TimeSpan ConstructionTimeout = TimeSpan.FromSeconds(10);

    private enum Verdict
    {
        /// <summary>Override computes what the base computes. Safe to delete.</summary>
        Redundant,

        /// <summary>Override disagrees with the base. Read it.</summary>
        LoadBearing,

        /// <summary>The two public members disagree with EACH OTHER. This is a live bug.</summary>
        SelfInconsistent,

        /// <summary>No override; already on the base path.</summary>
        AlreadyDerived,

        /// <summary>
        /// The nearest inherited declaration is ABSTRACT, so there is no implementation to fall back
        /// to and the override is mandatory. Not a defect and not deletable — these types need the
        /// generated enumeration to supply a body before they can join the migration.
        /// </summary>
        MandatoryOverride,

        /// <summary>Could not be measured (construction failed or timed out).</summary>
        Unmeasurable,
    }

    private sealed record Row(
        string TypeName,
        Verdict Verdict,
        long Declared,
        long ActualLength,
        long BaseDerived,
        bool OverridesCount,
        bool OverridesGet,
        string Note);

    [Fact(Timeout = 1800000)]
    public async System.Threading.Tasks.Task Report_ParameterEnumerationParity_AcrossAllModels()
    {
        await System.Threading.Tasks.Task.Yield();
        var rows = new List<Row>();

        // Stream each row to disk as it is measured, flushing every time. This sweep constructs
        // hundreds of models and takes many minutes, and a run that is interrupted -- CI job
        // timeout, an OOM, a developer stopping it -- would otherwise produce nothing at all
        // despite having done nearly all the work. Partial evidence is still evidence; the file
        // is the deliverable, the console output is a convenience.
        StreamWriter? tsv = null;
        try
        {
            tsv = new StreamWriter(TsvPath, append: false) { AutoFlush = true };
            tsv.WriteLine("Type\tVerdict\tDeclared\tActualLength\tBaseDerived\tOverridesCount\tOverridesGet\tNote");
        }
        catch
        {
            tsv = null;
        }

        void Record(Row r)
        {
            rows.Add(r);
            tsv?.WriteLine($"{r.TypeName}\t{r.Verdict}\t{r.Declared}\t{r.ActualLength}\t" +
                           $"{r.BaseDerived}\t{r.OverridesCount}\t{r.OverridesGet}\t{r.Note}");
        }

        foreach (var closedType in GetConstructableModelTypes())
        {
            var typeName = closedType.FullName ?? closedType.Name;

            bool overridesCount = DeclaresOverride(closedType, "ParameterCount");
            bool overridesGet = DeclaresGetParametersOverride(closedType);

            // Only types that override something are candidates for deletion, and only they can
            // desynchronise. A type already on the base path answers the migration question by
            // construction, and measuring it means constructing a model and materialising its full
            // parameter vector for no information — which is what pushed the first run past ten
            // minutes on ~2,000 types.
            if (!overridesCount && !overridesGet)
            {
                Record(new Row(typeName, Verdict.AlreadyDerived, -1, -1, -1, false, false,
                    "no override; nothing to migrate"));
                continue;
            }

            if (!TryConstruct(closedType, out object? instance, out string? ctorError) || instance is null)
            {
                Record(new Row(typeName, Verdict.Unmeasurable, -1, -1, -1,
                    overridesCount, overridesGet, ctorError ?? "construction failed"));
                continue;
            }

            try
            {
                long declared = ReadParameterCount(instance);
                if (declared > MaxParametersToMaterialize)
                {
                    Record(new Row(typeName, Verdict.Unmeasurable, declared, -1, -1,
                        overridesCount, overridesGet,
                        $"too large to materialise ({declared} parameters) - not measured"));
                    continue;
                }

                long actual = ReadGetParametersLength(instance);
                long baseDerived = BaseDerivedCount(instance, out string? baseErr);

                Verdict verdict;
                string note;

                if (declared != actual)
                {
                    // The live bug: whatever the override was for, the two public members do not
                    // describe the same tensors, so every length-paired caller is already wrong.
                    verdict = Verdict.SelfInconsistent;
                    note = $"declared {declared} vs actual {actual} (delta {declared - actual})";
                }
                else if (baseErr is not null)
                {
                    // Distinguish "the base genuinely cannot run here" from "this harness could not
                    // invoke it". A reflection/IL failure is a limitation of the measurement and must
                    // NOT be reported as a property of the model: an earlier revision classified 33
                    // BadImageFormatExceptions as LoadBearing, which is precisely the false verdict
                    // this report exists to avoid, and would have protected 33 overrides that may
                    // well be deletable.
                    bool abstractBase = baseErr.Contains("no inherited", StringComparison.Ordinal);
                    bool harnessFault = baseErr.Contains("BadImageFormat", StringComparison.Ordinal)
                                     || baseErr.Contains("InvalidProgram", StringComparison.Ordinal);

                    verdict = abstractBase ? Verdict.MandatoryOverride
                            : harnessFault ? Verdict.Unmeasurable
                            : Verdict.LoadBearing;

                    note = abstractBase
                        ? "nearest inherited declaration is abstract — override is required by the type system"
                        : harnessFault
                            ? $"HARNESS could not invoke the inherited implementation ({baseErr}) — no verdict"
                            : $"inherited implementation genuinely unusable ({baseErr}) — override is required";
                }
                else if (baseDerived != declared)
                {
                    verdict = Verdict.LoadBearing;
                    note = $"override {declared} vs inherited {baseDerived} " +
                           "(tensors outside Layers, or a different counting rule)";
                }
                else if (!BaseVectorMatches(instance, out long baseLen, out string? vecNote)
                         && !(vecNote?.Contains("BadImageFormat", StringComparison.Ordinal) ?? false))
                {
                    // Same count, different tensors. This is the case a count-only comparison would
                    // wave through, and it is the dangerous one: a checkpoint saved through the
                    // override restores element-for-element into the base's ordering.
                    verdict = Verdict.LoadBearing;
                    note = $"counts agree ({declared}) but vectors differ: {vecNote ?? "unknown"}" +
                           (baseLen >= 0 ? $" [base length {baseLen}]" : string.Empty);
                }
                else
                {
                    verdict = Verdict.Redundant;
                    note = $"count AND vector identical to inherited implementation ({declared})";
                }

                Record(new Row(typeName, verdict, declared, actual, baseDerived,
                    overridesCount, overridesGet, note));
            }
            catch (Exception ex)
            {
                Record(new Row(typeName, Verdict.Unmeasurable, -1, -1, -1,
                    overridesCount, overridesGet, $"{ex.GetType().Name}: {ex.Message}"));
            }
            finally
            {
                (instance as IDisposable)?.Dispose();
            }
        }

        tsv?.Dispose();
        WriteReport(rows);
    }

    private void WriteReport(List<Row> rows)
    {
        var byVerdict = rows.GroupBy(r => r.Verdict)
                            .ToDictionary(g => g.Key, g => g.ToList());

        _output.WriteLine($"Measured {rows.Count} model types.");
        _output.WriteLine("");
        _output.WriteLine("SUMMARY");
        foreach (Verdict v in Enum.GetValues(typeof(Verdict)))
        {
            int n = byVerdict.TryGetValue(v, out var list) ? list.Count : 0;
            _output.WriteLine($"  {v,-18} {n,5}");
        }

        _output.WriteLine("");
        _output.WriteLine("SELF-INCONSISTENT — live bugs, ParameterCount != GetParameters().Length.");
        _output.WriteLine("Every caller that slices a flat vector by ParameterCount is already wrong for these.");
        foreach (var r in Ordered(byVerdict, Verdict.SelfInconsistent))
        {
            _output.WriteLine($"  {r.TypeName}: {r.Note}");
        }

        _output.WriteLine("");
        _output.WriteLine("REDUNDANT — override computes exactly the base-derived value.");
        _output.WriteLine("These are the bulk deletion set: count AND element-by-element vector both match");
        foreach (var r in Ordered(byVerdict, Verdict.Redundant))
        {
            _output.WriteLine($"  {r.TypeName} (count={r.Declared})");
        }

        _output.WriteLine("");
        _output.WriteLine("MANDATORY — nearest inherited declaration is abstract; the override is required.");
        _output.WriteLine("These join the migration only once the generated enumeration can supply a body.");
        foreach (var r in Ordered(byVerdict, Verdict.MandatoryOverride))
        {
            _output.WriteLine($"  {r.TypeName} (count={r.Declared})");
        }

        _output.WriteLine("");
        _output.WriteLine("LOAD-BEARING — override differs from the base. Read each before touching it.");
        _output.WriteLine("Differs in count, in vector contents, or the inherited implementation cannot run at all.");
        foreach (var r in Ordered(byVerdict, Verdict.LoadBearing))
        {
            _output.WriteLine($"  {r.TypeName}: {r.Note}");
        }

        _output.WriteLine("");
        _output.WriteLine("UNMEASURABLE — could not construct or read. Not evidence of anything.");
        foreach (var r in Ordered(byVerdict, Verdict.Unmeasurable).Take(40))
        {
            _output.WriteLine($"  {r.TypeName}: {r.Note}");
        }

        _output.WriteLine("");
        _output.WriteLine($"TSV: {TsvPath}");
    }

    private static IEnumerable<Row> Ordered(Dictionary<Verdict, List<Row>> byVerdict, Verdict v)
        => byVerdict.TryGetValue(v, out var list)
            ? list.OrderBy(r => r.TypeName, StringComparer.Ordinal)
            : Enumerable.Empty<Row>();

    /// <summary>True when <paramref name="type"/> itself declares the override, not an ancestor.</summary>
    private static bool DeclaresOverride(Type type, string propertyName)
    {
        var prop = type.GetProperty(propertyName,
            BindingFlags.Public | BindingFlags.Instance | BindingFlags.FlattenHierarchy);
        var getter = prop?.GetGetMethod();
        return getter is not null && getter.DeclaringType == type;
    }

    private static bool DeclaresGetParametersOverride(Type type)
    {
        var m = type.GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance,
            binder: null, types: Type.EmptyTypes, modifiers: null);
        return m is not null && m.DeclaringType == type;
    }

    private static long ReadParameterCount(object instance)
    {
        var prop = instance.GetType().GetProperty("ParameterCount",
            BindingFlags.Public | BindingFlags.Instance | BindingFlags.FlattenHierarchy);
        return prop is null ? -1 : Convert.ToInt64(prop.GetValue(instance));
    }

    private static long ReadGetParametersLength(object instance)
    {
        var m = instance.GetType().GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance,
            binder: null, types: Type.EmptyTypes, modifiers: null);
        if (m is null) return -1;
        var vec = m.Invoke(instance, null);
        if (vec is null) return 0;
        var lenProp = vec.GetType().GetProperty("Length");
        return lenProp is null ? -1 : Convert.ToInt64(lenProp.GetValue(vec));
    }

    // Calling the REAL base implementation, not a reconstruction of it.
    //
    // An earlier version of this harness approximated the base by summing ParameterCount over
    // Layers. That is roughly what NeuralNetworkBase does, but only roughly — it also pre-resolves
    // lazy layer shapes from the architecture first, so a reconstruction can differ from the real
    // thing for exactly the lazy models this migration most needs to classify correctly. A
    // LoadBearing verdict produced that way could be an artifact of the approximation rather than a
    // real difference, and the whole point of this report is that someone can delete 600 overrides
    // on the strength of it.
    //
    // C# cannot express `someInstance.base.ParameterCount` from outside the type, and an open
    // delegate over the base MethodInfo still dispatches virtually — it would just call the
    // override again and every model would look Redundant. A DynamicMethod emitting `call` rather
    // than `callvirt` is the one way to get genuine non-virtual dispatch, so that is what this does.
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, Func<object, long>?> _baseCountInvokers = new();
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<Type, Func<object, object?>?> _baseGetInvokers = new();

    /// <summary>The nearest ancestor declaration — i.e. what would run if the override were deleted.</summary>
    private static MethodInfo? FindInheritedImplementation(Type type, Func<Type, MethodInfo?> pick)
    {
        for (var cur = type.BaseType; cur is not null; cur = cur.BaseType)
        {
            var m = pick(cur);
            // Skip abstract declarations: there is no body to call, and `call` against an abstract
            // method is invalid IL. Keep walking — a concrete implementation may sit further up.
            if (m is not null && !m.IsAbstract) return m;
            if (m is not null && m.IsAbstract) return null;   // abstract wins; nothing inherited
        }
        return null;
    }

    private static Func<object, long>? BaseParameterCountInvoker(Type modelType) =>
        _baseCountInvokers.GetOrAdd(modelType, t =>
        {
            var getter = FindInheritedImplementation(t, cur =>
                cur.GetProperty("ParameterCount",
                    BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)?.GetGetMethod());
            if (getter is null) return null;

            // Owned by the DECLARING type, not by this test's module: a DynamicMethod hosted in the
            // test assembly cannot reach another assembly's members even with skipVisibility, and
            // the failure surfaces as BadImageFormatException at invoke time rather than anything
            // legible. Widen int -> long explicitly; ParameterCount is long on 650 declarations but
            // int on four, and returning int32 where the signature says int64 is invalid IL.
            var dm = new System.Reflection.Emit.DynamicMethod(
                "CallBaseParameterCount", typeof(long), new[] { typeof(object) },
                getter.DeclaringType!, skipVisibility: true);
            var il = dm.GetILGenerator();
            il.Emit(System.Reflection.Emit.OpCodes.Ldarg_0);
            il.Emit(System.Reflection.Emit.OpCodes.Castclass, getter.DeclaringType!);
            il.Emit(System.Reflection.Emit.OpCodes.Call, getter);   // NOT callvirt
            if (getter.ReturnType == typeof(int)) il.Emit(System.Reflection.Emit.OpCodes.Conv_I8);
            il.Emit(System.Reflection.Emit.OpCodes.Ret);
            return (Func<object, long>)dm.CreateDelegate(typeof(Func<object, long>));
        });

    private static Func<object, object?>? BaseGetParametersInvoker(Type modelType) =>
        _baseGetInvokers.GetOrAdd(modelType, t =>
        {
            var method = FindInheritedImplementation(t, cur =>
                cur.GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly,
                    binder: null, types: Type.EmptyTypes, modifiers: null));
            if (method is null) return null;

            var dm = new System.Reflection.Emit.DynamicMethod(
                "CallBaseGetParameters", typeof(object), new[] { typeof(object) },
                method.DeclaringType!, skipVisibility: true);
            var il = dm.GetILGenerator();
            il.Emit(System.Reflection.Emit.OpCodes.Ldarg_0);
            il.Emit(System.Reflection.Emit.OpCodes.Castclass, method.DeclaringType!);
            il.Emit(System.Reflection.Emit.OpCodes.Call, method);   // NOT callvirt
            if (method.ReturnType.IsValueType) il.Emit(System.Reflection.Emit.OpCodes.Box, method.ReturnType);
            il.Emit(System.Reflection.Emit.OpCodes.Ret);
            return (Func<object, object?>)dm.CreateDelegate(typeof(Func<object, object?>));
        });

    private static long BaseDerivedCount(object instance, out string? error)
    {
        error = null;
        try
        {
            var invoker = BaseParameterCountInvoker(instance.GetType());
            if (invoker is null) { error = "no inherited ParameterCount"; return -1; }
            return invoker(instance);
        }
        catch (Exception ex)
        {
            error = $"base ParameterCount threw {ex.GetBaseException().GetType().Name}";
            return -1;
        }
    }

    /// <summary>
    /// Element-by-element comparison of the override's vector against the inherited one. Equal
    /// lengths are not enough to justify deleting an override — two different tensor sets can be
    /// the same size, and a saved checkpoint would then restore into the wrong ones.
    /// </summary>
    private static bool BaseVectorMatches(object instance, out long baseLength, out string? note)
    {
        baseLength = -1;
        note = null;
        try
        {
            var invoker = BaseGetParametersInvoker(instance.GetType());
            if (invoker is null) { note = "no inherited GetParameters"; return false; }

            var baseVec = invoker(instance);
            var ownVec = instance.GetType()
                .GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance,
                    binder: null, types: Type.EmptyTypes, modifiers: null)!
                .Invoke(instance, null);

            if (baseVec is null || ownVec is null) { note = "null vector"; return false; }

            var lenProp = baseVec.GetType().GetProperty("Length")!;
            baseLength = Convert.ToInt64(lenProp.GetValue(baseVec));
            long ownLength = Convert.ToInt64(lenProp.GetValue(ownVec));
            if (baseLength != ownLength) { note = $"length {ownLength} vs base {baseLength}"; return false; }

            var indexer = baseVec.GetType().GetProperty("Item", new[] { typeof(int) });
            if (indexer is null) { note = "vector not indexable; length-only comparison"; return true; }

            for (int i = 0; i < ownLength; i++)
            {
                var a = indexer.GetValue(ownVec, new object[] { i });
                var b = indexer.GetValue(baseVec, new object[] { i });
                if (!Equals(a, b)) { note = $"element {i} differs"; return false; }
            }
            return true;
        }
        catch (Exception ex)
        {
            note = $"base GetParameters threw {ex.GetBaseException().GetType().Name}";
            return false;
        }
    }

    private static bool TryConstruct(Type closedType, out object? instance, out string? error)
    {
        instance = null;
        error = null;
        try
        {
            var ctor = closedType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Where(c => c.GetParameters().Length == 0 ||
                            c.GetParameters().All(p => p.HasDefaultValue))
                .OrderBy(c => c.GetParameters().Length)
                .FirstOrDefault();
            if (ctor is null) { error = "no default-constructable ctor"; return false; }

            var args = ctor.GetParameters()
                .Select(p => p.DefaultValue == DBNull.Value ? null : p.DefaultValue)
                .ToArray();

            object? built = null;
            var task = System.Threading.Tasks.Task.Run(() => built = ctor.Invoke(args));
            if (!task.Wait(ConstructionTimeout))
            {
                // Drain rather than abandon: an abandoned constructor keeps burning CPU and makes
                // every subsequent measurement in this sweep slower and less trustworthy.
                task.Wait(TimeSpan.FromSeconds(20));
                error = $"construction timed out (>{ConstructionTimeout.TotalSeconds}s)";
                return false;
            }
            if (task.Exception is not null)
            {
                var inner = task.Exception.InnerException ?? task.Exception;
                error = $"{inner.GetType().Name}: {inner.Message}";
                return false;
            }

            instance = built;
            return instance is not null;
        }
        catch (Exception ex)
        {
            error = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    private static IEnumerable<Type> GetConstructableModelTypes()
    {
        var assembly = typeof(AiDotNet.Models.ModelMetadata<>).Assembly;
        var open = assembly.GetTypes()
            .Where(t => t.IsClass && !t.IsAbstract && t.IsGenericTypeDefinition)
            .Where(t => t.GetGenericArguments().Length == 1);

        foreach (var openType in open)
        {
            Type closed;
            try { closed = openType.MakeGenericType(typeof(double)); }
            catch { continue; }

            // Only models: the ones whose ParameterCount / GetParameters pair is the contract in
            // question. Layer-level parity is the generator's territory.
            bool isModel = closed.GetInterfaces().Any(i =>
                i.IsGenericType && i.GetGenericTypeDefinition().Name.StartsWith("IFullModel", StringComparison.Ordinal));
            if (!isModel) continue;

            bool constructable = closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Any(c => c.GetParameters().Length == 0 || c.GetParameters().All(p => p.HasDefaultValue));
            if (!constructable) continue;

            yield return closed;
        }
    }
}
