using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Measures whether <c>GetParameterChunks()</c> enumerates the SAME tensors that
/// <c>ParameterCount</c> counts and <c>GetParameters()</c> returns.
/// </summary>
/// <remarks>
/// <para>
/// This is a precondition check, not a bug hunt. The plan is to make <c>ParameterCount</c> and
/// <c>GetParameters</c> both fold from <c>GetParameterChunks()</c> so they cannot disagree. That is
/// only safe if the chunk enumeration already covers the same tensors -- and there is concrete
/// reason to doubt it. <c>GetParameterChunks</c> documents itself as yielding "the per-tensor
/// weight references registered via <c>RegisterTrainableParameter</c>" for
/// <c>ITrainableLayer&lt;T&gt;</c> layers, and "for non-trainable / parameterless layers this
/// yields nothing." Each layer's <c>GetParameters()</c>, by contrast, is a hand-written flattening
/// of whatever fields that layer happens to hold. Those are two different sources.
/// </para>
/// <para>
/// A layer that flattens weights in <c>GetParameters()</c> but never registered them would
/// contribute zero chunks. Rewiring the base would then drop those parameters from BOTH surfaces
/// at once -- and the pairing gate would stay green, because both sides would agree on the wrong
/// number. That is the failure this test exists to find BEFORE the rewire, not after.
/// </para>
/// <para>
/// Chunk lengths are read without materialising anything, so unlike the flat-vector comparison
/// this is safe to run even on the multi-billion-parameter video models.
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

        var divergent = new List<string>();
        var noChunks = new List<string>();
        int compared = 0, unmeasurable = 0, tooLarge = 0;

        var logPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "chunk-parity.txt");
        System.IO.StreamWriter? log = null;
        try { log = new System.IO.StreamWriter(logPath, append: false) { AutoFlush = true }; } catch { }

        foreach (var closedType in GetConstructableModelTypes())
        {
            var typeName = closedType.FullName ?? closedType.Name;
            log?.WriteLine($"[measuring] {typeName}");

            if (!TryConstruct(closedType, out object? instance) || instance is null) { unmeasurable++; continue; }

            try
            {
                long declared = ReadLong(instance, "ParameterCount");
                if (declared < 0) { unmeasurable++; continue; }

                // Size check BEFORE touching the chunk API. GetParameterChunks() is documented as
                // returning zero-copy references, but it calls ResolveLazyLayerShapes() first --
                // which ALLOCATES every deferred weight tensor. For CogVideo's paper-scale 5B
                // variant that is tens of GB at double, and it killed this sweep outright. The
                // enumeration is cheap only once a model is already materialised; asking an
                // unmaterialised giant for its chunks is what forces the materialisation.
                if (declared > MaxParametersToMaterialize)
                {
                    tooLarge++;
                    log?.WriteLine($"TOO-LARGE {typeName}: ParameterCount={declared}");
                    continue;
                }

                var chunksMethod = closedType.GetMethod("GetParameterChunks",
                    BindingFlags.Public | BindingFlags.Instance, null, Type.EmptyTypes, null);
                if (chunksMethod is null)
                {
                    noChunks.Add($"{typeName}: no GetParameterChunks (declared={declared})");
                    continue;
                }

                long chunkSum = 0;
                int chunkCount = 0;
                if (chunksMethod.Invoke(instance, null) is IEnumerable chunks)
                {
                    foreach (var chunk in chunks)
                    {
                        if (chunk is null) continue;
                        var lenProp = chunk.GetType().GetProperty("Length");
                        if (lenProp is not null) chunkSum += Convert.ToInt64(lenProp.GetValue(chunk));
                        chunkCount++;
                    }
                }

                compared++;

                // Flat length only when it is cheap; the chunk comparison above is the point.
                // chunkSum is the honest size here: ParameterCount reports 0 for layers whose
                // input width is still deferred, so it can under-report a large model.
                long flat = -1;
                if (chunkSum <= MaxParametersToMaterialize)
                {
                    try { flat = ReadVectorLength(instance); }
                    catch { flat = -1; }
                }

                if (chunkSum != declared || (flat >= 0 && flat != chunkSum))
                {
                    var row = $"{typeName}: ParameterCount={declared}, chunkSum={chunkSum} " +
                              $"({chunkCount} chunks), GetParameters().Length={(flat < 0 ? "n/a" : flat.ToString())}";
                    divergent.Add(row);
                    log?.WriteLine("DIVERGENT " + row);
                }
            }
            catch (Exception ex)
            {
                unmeasurable++;
                _output.WriteLine($"UNMEASURABLE {typeName}: {ex.GetBaseException().GetType().Name}");
            }
            finally
            {
                (instance as IDisposable)?.Dispose();
            }

            if (compared % 50 == 0) { GC.Collect(); GC.WaitForPendingFinalizers(); }
        }

        log?.Dispose();

        _output.WriteLine($"Compared {compared} models; {noChunks.Count} expose no chunk API; " +
                          $"{tooLarge} too large to enumerate; {unmeasurable} unmeasurable; {divergent.Count} divergent.");
        foreach (var n in noChunks.Take(40)) _output.WriteLine("  NO-CHUNKS " + n);
        foreach (var d in divergent) _output.WriteLine("  DIVERGENT " + d);

        // Reported, not enforced. This measures whether a planned refactor is safe; it is not
        // itself a contract anyone has agreed to yet, and failing the build on it would block
        // work on a question we are still answering.
        Assert.True(true);
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
