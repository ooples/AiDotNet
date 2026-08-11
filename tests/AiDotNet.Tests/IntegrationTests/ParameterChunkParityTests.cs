using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Enforces that <c>GetParameterChunks()</c> enumerates the SAME persistent state that
/// <c>ParameterCount</c> counts and <c>GetParameters()</c> returns.
/// </summary>
/// <remarks>
/// <para>
/// The framework bases and generator now expose one stable manifest for count, flat values,
/// chunks, and restore. Role-aware chunks keep optimizer semantics separate: trainable tensors and
/// non-trainable buffers occupy the same checkpoint order, but only trainable roles are handed to
/// an optimizer. Any disagreement here is therefore a product contract failure, not a report-only
/// migration metric.
/// </para>
/// <para>
/// Chunk lengths themselves are references, but ASKING for them is not free:
/// <c>GetParameterChunks()</c> calls <c>ResolveLazyLayerShapes()</c> first, which allocates every
/// deferred weight tensor. An earlier version of this comment claimed the enumeration was safe on
/// the multi-billion-parameter video models; it is not, and that assumption OOM-killed both a local
/// run and a CI runner. Models that are too large, or not sized yet, are skipped and counted --
/// see the guards in the loop for why a size threshold alone cannot catch the second case.
/// </para>
/// </remarks>
[Collection("ParameterSweeps")]
[Trait("Category", "Sweep")]
public class ParameterChunkParityTests
{
    private readonly ITestOutputHelper _output;

    public ParameterChunkParityTests(ITestOutputHelper output) => _output = output;

    /// <summary>Above this, the flat vector is never requested (see ParameterCountContractTests).</summary>
    private const long MaxParametersToMaterialize = 50_000_000L;

    private static readonly TimeSpan ConstructionTimeout = TimeSpan.FromSeconds(10);

    [Fact(Timeout = 1800000)]
    public async System.Threading.Tasks.Task ChunkSum_ShouldMatchParameterCount()
    {
        await System.Threading.Tasks.Task.Yield();
#if !NET10_0_OR_GREATER
        return;
#endif

        var divergent = new List<string>();
        var noChunks = new List<string>();
        int compared = 0, unmeasurable = 0, tooLarge = 0, unsized = 0;
        // THE GC CADENCE COUNTS CONSTRUCTIONS, NOT SUCCESSFUL COMPARISONS. Keying it off
        // `compared` meant that while `compared` was still 0 -- every model that is
        // unmeasurable, too large, unsized, or has no chunk API leaves it there -- `0 % 50 == 0`
        // held, forcing a blocking full collection plus finalizer wait for EVERY model at the
        // start of the sweep. ParameterCountContractTests already does it this way.
        int constructed = 0;

        var logPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "chunk-parity.txt");
        // DISPOSED ON EVERY PATH, AND A FAILURE TO OPEN IS REPORTED. The manual dispose ran only
        // on the normal path, so a throw from the enumeration below -- Assembly.GetTypes() raises
        // ReflectionTypeLoadException, and this sweep calls it -- leaked the handle and left the
        // partial log the comment above calls "the deliverable" unflushed. The empty catch was the
        // other half: when the file could not be opened at all, every later write silently went
        // nowhere and the run looked like one that simply found nothing.
        System.IO.StreamWriter? log = null;
        try
        {
            log = new System.IO.StreamWriter(logPath, append: false) { AutoFlush = true };
        }
        catch (Exception ex)
        {
            _output.WriteLine($"NOTE: could not open {logPath} ({ex.GetType().Name}: {ex.Message}); " +
                              "the console output below is the only record of this run.");
        }

        using var _logHandle = log;

        var modelTypes = GetConstructableModelTypes().ToArray();
        var measurements = await ParameterSweepProcess.MeasureAllAsync(
            modelTypes, includeChunks: true, MaxParametersToMaterialize, ConstructionTimeout);

        foreach (var result in measurements)
        {
            var closedType = result.ModelType;
            var typeName = closedType.FullName ?? closedType.Name;
            log?.WriteLine($"[measuring] {typeName}");
            constructed++;
            if (constructed % 50 == 0) { GC.Collect(); GC.WaitForPendingFinalizers(); }

            var measurement = result.Measurement;
            switch (measurement.Status)
            {
                case "too-large":
                    tooLarge++;
                    log?.WriteLine($"TOO-LARGE {typeName}: ParameterCount={measurement.Declared}");
                    continue;
                case "deferred":
                case "unmaterialized":
                    unsized++;
                    log?.WriteLine($"UNSIZED {typeName}: readiness={measurement.Readiness}");
                    continue;
                case "no-chunks":
                    noChunks.Add($"{typeName}: no GetParameterChunks (declared={measurement.Declared})");
                    continue;
                case "ok":
                    break;
                default:
                    unmeasurable++;
                    _output.WriteLine($"UNMEASURABLE {typeName}: {measurement.Status} {measurement.Error}");
                    continue;
            }

            compared++;
            bool ambiguousZero = measurement.Declared == 0 && measurement.Flat == 0 &&
                                 measurement.ChunkSum == 0 && measurement.Readiness != "ParameterFree";
            if (measurement.ChunkSum != measurement.Declared ||
                (measurement.Flat >= 0 && measurement.Flat != measurement.ChunkSum) || ambiguousZero)
            {
                var row = $"{typeName}: ParameterCount={measurement.Declared}, " +
                          $"chunkSum={measurement.ChunkSum} ({measurement.ChunkCount} chunks), " +
                          $"GetParameters().Length={(measurement.Flat < 0 ? "n/a" : measurement.Flat.ToString())}, " +
                          $"readiness={measurement.Readiness}";
                divergent.Add(row);
                log?.WriteLine("DIVERGENT " + row);
            }
        }

        _output.WriteLine($"Compared {compared} models; {noChunks.Count} expose no chunk API; " +
                          $"{tooLarge} too large to enumerate; {unsized} not sized yet; " +
                          $"{unmeasurable} unmeasurable; {divergent.Count} divergent.");
        foreach (var n in noChunks.Take(40)) _output.WriteLine("  NO-CHUNKS " + n);
        foreach (var d in divergent) _output.WriteLine("  DIVERGENT " + d);

        Assert.True(compared > 0,
            "The chunk-parity sweep compared NOTHING. It holds a 30-minute slot, so a run that " +
            "measured nothing is an infrastructure failure rather than a clean report. " +
            $"Skipped: {noChunks.Count} with no chunk API, {tooLarge} too large to enumerate, " +
            $"{unsized} not sized yet, {unmeasurable} unmeasurable.");
        Assert.True(noChunks.Count == 0,
            "Every materialized model must inherit or provide the canonical chunk API.\n" +
            string.Join("\n", noChunks));
        Assert.True(divergent.Count == 0,
            "ParameterCount, GetParameters, and GetParameterChunks must fold the same state " +
            "manifest.\n" + string.Join("\n", divergent));
    }

    private static bool ReadBool(object instance, string propertyName)
    {
        var prop = instance.GetType().GetProperty(propertyName,
            BindingFlags.Public | BindingFlags.Instance | BindingFlags.FlattenHierarchy);
        if (prop is null) return false;
        try { return prop.GetValue(instance) is true; }
        catch { return false; }
    }

    private static long ReadLong(object instance, string propertyName)
    {
        var prop = instance.GetType().GetProperty(propertyName,
            BindingFlags.Public | BindingFlags.Instance | BindingFlags.FlattenHierarchy);
        if (prop is null) return -1;
        try { return Convert.ToInt64(prop.GetValue(instance)); }
        catch { return -1; }
    }

    private static long ReadVectorLength(object instance)
    {
        var m = instance.GetType().GetMethod("GetParameters", BindingFlags.Public | BindingFlags.Instance,
            null, Type.EmptyTypes, null);
        if (m is null) return -1;
        var vec = m.Invoke(instance, null);
        if (vec is null) return 0;
        var lenProp = vec.GetType().GetProperty("Length");
        return lenProp is null ? -1 : Convert.ToInt64(lenProp.GetValue(vec));
    }

    private static bool TryConstruct(Type closedType, out object? instance)
    {
        instance = null;
        try
        {
            var ctor = closedType.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Where(c => c.GetParameters().Length == 0 || c.GetParameters().All(p => p.HasDefaultValue))
                .OrderBy(c => c.GetParameters().Length)
                .FirstOrDefault();
            if (ctor is null) return false;

            var args = ctor.GetParameters()
                .Select(p => p.DefaultValue == DBNull.Value ? null : p.DefaultValue)
                .ToArray();

            object? built = null;
            var task = System.Threading.Tasks.Task.Run(() => built = ctor.Invoke(args));
            if (!task.Wait(ConstructionTimeout)) { task.Wait(TimeSpan.FromSeconds(20)); return false; }
            if (task.Exception is not null) return false;

            instance = built;
            return instance is not null;
        }
        catch { return false; }
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

            bool isModel = closed.GetInterfaces().Any(i =>
                i.IsGenericType &&
                i.GetGenericTypeDefinition().Name.StartsWith("IFullModel", StringComparison.Ordinal));
            if (!isModel) continue;

            bool constructable = closed.GetConstructors(BindingFlags.Public | BindingFlags.Instance)
                .Any(c => c.GetParameters().Length == 0 || c.GetParameters().All(p => p.HasDefaultValue));
            if (!constructable) continue;

            yield return closed;
        }
    }
}
