#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;
using AiDotNet.Tensors.Engines;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GpuEngineSwapCollection
{
    // AiDotNetEngine.Current is process-wide: nothing else may run while a test swaps it to the GPU.
    public const string Name = "GpuEngineSwap";
}

/// <summary>
/// Every model-family network, predicted on the GPU engine: GPU Predict runs its eager forward in a TensorScope and
/// keeps whatever the model still reaches afterwards (<see cref="ReachableTensorKeeper"/>). A model that holds a tensor
/// the walk cannot see (a static cache, a delegate's capture) would find it freed on the next call; this fails on any
/// such tensor, and on a second call that cannot read its output.
/// </summary>
[Collection(GpuEngineSwapCollection.Name)]
[Trait("Category", "GPU")]
public class GpuPredictScopeSweepTests
{
    private static readonly Lazy<DirectGpuTensorEngine?> s_gpu = new(() =>
    {
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable) return gpu;
            gpu.Dispose();
        }
        catch (Exception) { /* no usable GPU backend: every case returns */ }
        return null;
    });

    private readonly ITestOutputHelper _output;

    public GpuPredictScopeSweepTests(ITestOutputHelper output) => _output = output;

    public static IEnumerable<object[]> ModelFamilies() =>
        typeof(GpuPredictScopeSweepTests).Assembly.GetTypes()
            // The model-family fixtures: other subclasses of the base are probes of one harness feature, not models.
            .Where(t => !t.IsAbstract && t.GetConstructor(Type.EmptyTypes) is not null && DerivesFromModelTestBase(t)
                && t.Namespace?.StartsWith("AiDotNet.Tests.ModelFamilyTests", StringComparison.Ordinal) == true)
            .Select(t => t.FullName!)
            .OrderBy(n => n, StringComparer.Ordinal)
            .Select(n => new object[] { n });

    [Theory(Timeout = 300000)]
    [MemberData(nameof(ModelFamilies))]
    public async Task GpuPredict_KeepsEveryTensorTheModelHolds(string fixtureType)
    {
        await Task.Yield();
        if (s_gpu.Value is not { } gpu)
        {
            _output.WriteLine("No GPU - nothing to test.");
            return;
        }

        var fixture = (IAsyncLifetime)Activator.CreateInstance(typeof(GpuPredictScopeSweepTests).Assembly.GetType(fixtureType)!)!;
        await fixture.InitializeAsync();
        var previous = AiDotNetEngine.Current;
        object? network = null;
        try
        {
            var type = fixture.GetType();
            const BindingFlags flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
            network = type.GetMethod("CreateNetwork", flags, null, Type.EmptyTypes, null)!.Invoke(fixture, null)!;
            var shape = (int[])type.GetProperty("EffectiveInputShape", flags)!.GetValue(fixture)!;
            var makeInput = type.GetMethod("CreateConstantTensor", flags, null, new[] { typeof(int[]), typeof(double) }, null)!;
            var predict = network.GetType().GetMethod("Predict", new[] { makeInput.ReturnType })!;
            var input1 = makeInput.Invoke(fixture, new object[] { shape, 0.1 });
            var input2 = makeInput.Invoke(fixture, new object[] { shape, 0.9 });

            AiDotNetEngine.Current = gpu;
            ReadAll(predict.Invoke(network, new[] { input1 }));
            var held = predict.Invoke(network, new[] { input2 });
            ReadAll(predict.Invoke(network, new[] { input1 }));
            ReadAll(held);

            var released = ReleasedReachableTensors(network);
            Assert.True(released.Count == 0,
                $"{type.Name}: {released.Count} tensors the model holds were released by a Predict's scope, e.g. {string.Join("; ", released.Take(3))}");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
            (network as IDisposable)?.Dispose();
            await fixture.DisposeAsync();
        }
    }

    /// <summary>The shapes of the tensors reachable from <paramref name="model"/> whose device storage was released.</summary>
    internal static List<string> ReleasedReachableTensors(object model)
    {
        var released = new List<string>();
        ReachableTensorKeeper.VisitReachable(model, (tensor, _) =>
        {
            var type = tensor.GetType();
            var vector = type.GetProperty("DataVector", BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.GetValue(tensor)!;
            if (!AiDotNet.Tensors.Helpers.HostSync.IsReleased(vector)) return;
            var shape = type.GetField("_shape", BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(tensor) as int[]
                ?? (type.GetProperty("Shape")?.GetValue(tensor) as IEnumerable<int>)?.ToArray() ?? Array.Empty<int>();
            released.Add($"{type.Name}[{string.Join(",", shape)}]");
        });
        return released;
    }

    // Reads the result on the host: throws the released-intermediate error if the scope freed it.
    private static void ReadAll(object? tensor) => tensor?.GetType().GetMethod("ToArray", Type.EmptyTypes)!.Invoke(tensor, null);

    private static bool DerivesFromModelTestBase(Type type)
    {
        for (var t = type.BaseType; t is not null; t = t.BaseType)
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(NeuralNetworkModelTestBase<>)) return true;
        return false;
    }
}
#endif
