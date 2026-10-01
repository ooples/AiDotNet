using System.Reflection;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Diffusion;
using AiDotNet.Enums;
using AiDotNet.Models.Parameters;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tokenization;
using Xunit;

namespace AiDotNet.Tests.UnitTests.VisionLanguage;

public sealed class JointVisionLanguageStateTests
{
    public JointVisionLanguageStateTests() => TestModuleInitializer.EnsureInitialized();

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeVision_PreservesBatchAndTokenAxes(bool batched)
    {
        using var model = new SmallLlava();
        var image = Values(batched ? new[] { 2, 3, 4, 4 } : new[] { 3, 4, 4 });
        var features = model.ExtractVisualFeatures(image);
        Assert.Equal(batched ? new[] { 2, 5, 8 } : new[] { 5, 8 }, features.Shape.ToArray());
    }

    [Fact]
    public void JoiningVisualAndTextStates_PreservesBothActualGradients()
    {
        using var model = new SmallLlava();
        var visual = Values(new[] { 2, 8 });
        var text = Values(new[] { 3, 8 });
        var join = typeof(LLaVANeuralNetwork<float>).GetMethod("ConcatenateSequences", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(join);
        using var tape = new GradientTape<float>();
        var joined = Assert.IsType<Tensor<float>>(join.Invoke(model, new object[] { visual, text }));
        var loss = model.SquaredSum(joined);
        var gradients = tape.ComputeGradients(loss, new[] { visual, text });
        foreach (var input in new[] { visual, text })
        {
            Assert.True(gradients.TryGetValue(input, out var gradient));
            Assert.NotNull(gradient);
            Assert.Equal(input.ToArray().Select(value => value * 2), gradient.ToArray());
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DiffusionCollector_UsesDeclaredLiveStorage_AndDoesNotReAddFrozenChildren(bool ownerVisitedFirst)
    {
        using var layer = new DenseLayer<float>(2);
        layer.Forward(Values(new[] { 1, 3 }));
        var live = Values(new[] { 2, 2 });
        var owner = new DeclaredOwner(live, layer);
        object root = ownerVisitedFirst ? new object[] { layer, owner } : new object[] { owner, layer };
        var parameters = Collect(root);
        Assert.Single(parameters);
        Assert.Same(live, parameters[0]);
    }

    internal static List<Tensor<float>> Collect(object root)
    {
        var collect = typeof(DiffusionModelBase<float>).GetMethod("CollectLayerParameters", BindingFlags.Static | BindingFlags.NonPublic);
        Assert.NotNull(collect);
        var parameters = new List<Tensor<float>>();
        collect.Invoke(null, new[] { root, parameters });
        return parameters;
    }

    internal static Tensor<float> Values(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = (i + 1) * 0.007f;
        return tensor;
    }

    private sealed class DeclaredOwner : IParameterChunkSource<float>
    {
        private readonly Tensor<float> _live;
        private readonly DenseLayer<float> _frozenChild;

        internal DeclaredOwner(Tensor<float> live, DenseLayer<float> frozenChild)
        {
            _live = live;
            _frozenChild = frozenChild;
        }

        public IEnumerable<ParameterChunk<float>> GetParameterStateChunks()
        {
            // The persistence payload is a snapshot, not the tensor that participated in forward.
            yield return new ParameterChunk<float>("weight", ParameterSlotRole.Trainable,
                new Tensor<float>(new[] { _live.Length }, new Vector<float>(_live.ToArray())), _live);
            yield return new ParameterChunk<float>("weight-alias", ParameterSlotRole.Alias, _live);
            foreach (var parameter in _frozenChild.GetTrainableParameters())
                yield return new ParameterChunk<float>("frozen-child", ParameterSlotRole.Frozen, parameter);
        }
    }

    internal sealed class SmallLlava : LLaVANeuralNetwork<float>
    {
        internal SmallLlava() : base(new NeuralNetworkArchitecture<float>(
                inputType: InputType.ThreeDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputDepth: 3, inputHeight: 4, inputWidth: 4, outputSize: 8),
            new AiDotNet.NeuralNetworks.Options.LLaVAOptions
            {
                ImageSize = 4, PatchSize = 2, VocabSize = 64, MaxSequenceLength = 64,
                EmbeddingDimension = 8, VisionDim = 8, VisionLayers = 1, NumLmLayers = 1, NumHeads = 2
            },
            tokenizer: LanguageModelTokenizerFactory.CreateForBackbone(
                LanguageModelBackbone.LLaMA, new[] { "red blue bright dark image" }, 64)) { }

        internal Tensor<float> SquaredSum(Tensor<float> input)
            => Engine.ReduceSum(Engine.TensorMultiply(input, input), new[] { 0, 1 }, keepDims: false);

        internal Tensor<float> WeightedSum(Tensor<float> input)
        {
            var weights = Values(input.Shape.ToArray());
            return Engine.ReduceSum(Engine.TensorMultiply(input, weights),
                Enumerable.Range(0, input.Rank).ToArray(), keepDims: false);
        }
    }
}
