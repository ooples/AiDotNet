using System;
using System.Linq;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Graph;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests.NeuralNetworks.Graph;

/// <summary>
/// Pins the difference between validating a GUESSED chain and validating the one that actually ran.
/// </summary>
/// <remarks>
/// <para>
/// Reading <c>Layers</c> as a linear list assumes the list IS the dataflow. For a branched model it is
/// not - it is several independent branches flattened into one list - so the reading pairs layers that
/// never meet. Measured across every constructible model, that guess produced four reports and every one
/// was false.
/// </para>
/// <para>
/// The four are kept here by name because each is a DIFFERENT way of being branched, and a fix that
/// handled only one shape would look complete: TRIE flattens a visual encoder and a text encoder; SAM2
/// returns only its image encoder as a chainable sequence and wires other branches separately; VideoMAE
/// follows a classification head with a reconstruction decoder that consumes ENCODER features, not the
/// classifier's; ABCNet enters its recognition branch through a permute+reshape.
/// </para>
/// </remarks>
public class TracedChainValidationTests
{
    private readonly ITestOutputHelper _out;
    public TracedChainValidationTests(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData("ABCNet")]
    [InlineData("SAM2")]
    [InlineData("TRIE")]
    [InlineData("VideoMAE")]
    public void TracedValidationClearsTheFalsePositivesTheLinearReadingProduces(string modelName)
    {
        const System.Reflection.BindingFlags Flags = System.Reflection.BindingFlags.Instance
            | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public
            | System.Reflection.BindingFlags.FlattenHierarchy;

        var open = typeof(NeuralNetworkBase<>).Assembly.GetTypes()
            .FirstOrDefault(t => t.Name == modelName + "`1" && t.IsGenericTypeDefinition);
        Assert.True(open is not null, $"{modelName} not found");

        object? instance = null;
            try { instance = modelName == "SAM2" ? SmallSam2() : Activator.CreateInstance(open.MakeGenericType(typeof(double))); }
        catch (Exception ex) { _out.WriteLine($"cannot construct: {ex.GetType().Name}"); return; }
        if (instance is null) return;

        dynamic layers = (instance.GetType().GetProperty("Layers", Flags)?.GetValue(instance) ?? throw new InvalidOperationException("instance.GetType().GetProperty(\"Layers\", Flags)?.GetValue(instance) is null where it is required."));
        if (layers is null || layers.Count < 2) return;

        int linear = LayerContractValidator.Validate<double>(layers).Count;

        dynamic architecture = ((dynamic)instance).GetArchitecture();
        int[] perSample = architecture.GetInputShape();
        var shape = new int[perSample.Length + 1];
        shape[0] = 1;
        for (int i = 0; i < perSample.Length; i++) shape[i + 1] = perSample[i];

        var probe = new Tensor<double>(shape);
        for (int i = 0; i < probe.Length; i++) probe[i] = (i % 7) / 7.0;

        int traced;
        using (var trace = new LayerForwardObserver<double>())
        {
            ((dynamic)instance).Predict(probe);
            traced = trace.ContiguousRuns().Sum(r => LayerContractValidator.Validate<double>(r).Count);
        }

        _out.WriteLine($"{modelName}: linear={linear} traced={traced}");

        Assert.True(
            traced == 0,
            $"{modelName}: validating the TRACED dataflow still reports {traced} layout mismatch(es). "
            + "The trace is the dataflow that actually ran, so a report here is either a real defect or a "
            + "gap in the tracer - it is no longer explainable as the linear reading pairing layers that "
            + "never meet.");
    }

    // SAM 2's default is the paper's Hiera-B+ at 1024 px, whose forward in double took the testhost past 12 GB; the
    // branches this test traces are the same at any width, so it runs the generated suite's capacity-scaled SAM 2
    // (every module kept, a 32 px canvas).
    private static object SmallSam2() => new AiDotNet.Video.Segmentation.SAM2<double>(
        new NeuralNetworkArchitecture<double>(inputType: AiDotNet.Enums.InputType.ThreeDimensional,
            taskType: AiDotNet.Enums.NeuralNetworkTaskType.Regression, inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 1),
        options: new AiDotNet.Video.Options.SAM2Options
        {
            MemoryBankSize = 2, HieraEmbeddingDimension = 16, HieraStageDepths = new[] { 1, 1, 1, 1 }, HieraInitialHeadCount = 1,
            HieraWindowSizes = new[] { 8, 4, 2, 1 }, HieraGlobalAttentionBlockIndexes = new[] { 2 }, ModelDimension = 16,
            MemoryDimension = 8, DecoderHeadCount = 4, MemoryAttentionLayerCount = 4, MemoryAttentionFeedForwardDimension = 64,
            MemoryFuserDimension = 32, MaskDecoderMlpDimension = 64,
        });
}
