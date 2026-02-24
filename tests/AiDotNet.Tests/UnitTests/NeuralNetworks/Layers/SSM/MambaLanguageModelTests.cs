using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Integration tests for <see cref="MambaLanguageModel{T}"/>.
/// Tests full forward-backward-parameter round-trips and multi-layer compositions.
/// </summary>
public class MambaLanguageModelTests
{
    [Fact]
    public void Constructor_ValidParameters_CreatesModel()
    {
        var model = new MambaLanguageModel<float>(
            vocabSize: 100, modelDimension: 32, numLayers: 2, stateDimension: 8);

        Assert.Equal(100, model.VocabSize);
        Assert.Equal(32, model.ModelDimension);
        Assert.Equal(2, model.NumLayers);
        Assert.Equal(8, model.StateDimension);
    }

    [Fact]
    public void Constructor_ThrowsWhenVocabSizeNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(vocabSize: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(vocabSize: 100, modelDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenNumLayersNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(vocabSize: 100, numLayers: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenStateDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(vocabSize: 100, stateDimension: 0));
    }

    [Fact]
    public void Forward_3D_ProducesCorrectOutputShape()
    {
        int batchSize = 2;
        int seqLen = 4;
        int vocabSize = 50;
        int modelDim = 32;

        var model = new MambaLanguageModel<float>(
            vocabSize, modelDim, numLayers: 2, stateDimension: 8, maxSeqLength: seqLen);

        var input = CreateOneHotInput(batchSize, seqLen, vocabSize);
        var output = model.Forward(input);

        Assert.Equal(new[] { batchSize, seqLen, vocabSize }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Forward_2D_ProducesCorrectOutputShape()
    {
        int seqLen = 4;
        int vocabSize = 50;
        int modelDim = 32;

        var model = new MambaLanguageModel<float>(
            vocabSize, modelDim, numLayers: 2, stateDimension: 8, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize).Reshape(seqLen, vocabSize);
        var output = model.Forward(input);

        Assert.Equal(new[] { seqLen, vocabSize }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backward_ProducesValidGradients()
    {
        int seqLen = 4;
        int vocabSize = 50;
        int modelDim = 32;

        var model = new MambaLanguageModel<float>(
            vocabSize, modelDim, numLayers: 2, stateDimension: 8, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output = model.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = model.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Backward_ThrowsWithoutForward()
    {
        var model = new MambaLanguageModel<float>(50, 32, 2, 8, maxSeqLength: 4);
        var grad = CreateRandomTensor(new[] { 1, 4, 50 });

        Assert.Throws<InvalidOperationException>(() => model.Backward(grad));
    }

    [Fact]
    public void FullTrainingStep_ForwardBackwardUpdate_NoErrors()
    {
        int seqLen = 4;
        int vocabSize = 30;
        int modelDim = 16;

        var model = new MambaLanguageModel<float>(
            vocabSize, modelDim, numLayers: 2, stateDimension: 4, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output = model.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        model.Backward(grad);
        model.UpdateParameters(0.001f);

        // Verify model still produces valid output after parameter update
        model.ResetState();
        var output2 = model.Forward(input);
        Assert.Equal(output.Shape, output2.Shape);
        Assert.False(ContainsNaN(output2));
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        var model = new MambaLanguageModel<float>(
            50, 32, numLayers: 2, stateDimension: 8, maxSeqLength: 4);

        var params1 = model.GetParameters();
        Assert.True(params1.Length > 0);
        Assert.Equal(model.ParameterCount, params1.Length);

        model.SetParameters(params1);
        var params2 = model.GetParameters();

        Assert.Equal(params1.Length, params2.Length);
        for (int i = 0; i < params1.Length; i++)
        {
            Assert.Equal(params1[i], params2[i]);
        }
    }

    [Fact]
    public void SetParameters_ThrowsOnWrongLength()
    {
        var model = new MambaLanguageModel<float>(50, 32, 2, 8, maxSeqLength: 4);
        Assert.Throws<ArgumentException>(() => model.SetParameters(new Vector<float>(10)));
    }

    [Fact]
    public void Forward_DeterministicWithSameParameters()
    {
        int seqLen = 4;
        int vocabSize = 30;
        int modelDim = 16;

        var model1 = new MambaLanguageModel<float>(
            vocabSize, modelDim, numLayers: 2, stateDimension: 4, maxSeqLength: seqLen);
        var model2 = new MambaLanguageModel<float>(
            vocabSize, modelDim, numLayers: 2, stateDimension: 4, maxSeqLength: seqLen);

        model2.SetParameters(model1.GetParameters());

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output1 = model1.Forward(input);
        var output2 = model2.Forward(input);

        var arr1 = output1.ToArray();
        var arr2 = output2.ToArray();
        for (int i = 0; i < arr1.Length; i++)
        {
            Assert.True(MathF.Abs(arr1[i] - arr2[i]) < 1e-5f,
                $"Mismatch at {i}: {arr1[i]:G6} vs {arr2[i]:G6}");
        }
    }

    [Fact]
    public void ResetState_AllowsReuse()
    {
        var model = new MambaLanguageModel<float>(30, 16, 2, 4, maxSeqLength: 4);
        var input = CreateOneHotInput(1, 4, 30);

        model.Forward(input);
        model.ResetState();

        var output = model.Forward(input);
        Assert.NotNull(output);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SupportsTraining_ReturnsTrue()
    {
        var model = new MambaLanguageModel<float>(30, 16, 2, 4, maxSeqLength: 4);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void GetMetadata_ContainsExpectedKeys()
    {
        var model = new MambaLanguageModel<float>(100, 64, 4, 16, maxSeqLength: 32);
        var metadata = model.GetMetadata();

        Assert.True(metadata.ContainsKey("VocabSize"));
        Assert.True(metadata.ContainsKey("ModelDimension"));
        Assert.True(metadata.ContainsKey("NumLayers"));
        Assert.True(metadata.ContainsKey("StateDimension"));
        Assert.Equal("100", metadata["VocabSize"]);
        Assert.Equal("64", metadata["ModelDimension"]);
        Assert.Equal("4", metadata["NumLayers"]);
    }

    [Fact]
    public void Forward_Double_ProducesValidOutput()
    {
        int seqLen = 4;
        int vocabSize = 30;

        var model = new MambaLanguageModel<double>(
            vocabSize, 16, numLayers: 2, stateDimension: 4, maxSeqLength: seqLen);

        var input = CreateOneHotDoubleInput(1, seqLen, vocabSize);
        var output = model.Forward(input);

        Assert.Equal(new[] { 1, seqLen, vocabSize }, output.Shape);
        Assert.False(ContainsNaNDouble(output));
    }

    [Fact]
    public void InitializeStateCache_CreatesCache()
    {
        var model = new MambaLanguageModel<float>(30, 16, 2, 4, maxSeqLength: 4);

        var cache = model.InitializeStateCache();

        Assert.NotNull(cache);
        Assert.NotNull(model.StateCache);
        Assert.Equal(0, cache.CachedLayerCount);
        Assert.False(cache.CompressionEnabled);
    }

    [Fact]
    public void InitializeStateCache_WithCompression_EnablesCompression()
    {
        var model = new MambaLanguageModel<float>(30, 16, 2, 4, maxSeqLength: 4);

        var cache = model.InitializeStateCache(enableCompression: true, compressionBitWidth: 8);

        Assert.True(cache.CompressionEnabled);
    }

    [Fact]
    public void GenerateStep_ProducesValidLogits()
    {
        int vocabSize = 20;
        var model = new MambaLanguageModel<float>(vocabSize, 16, 2, 4, maxSeqLength: 4);
        model.InitializeStateCache();

        // Create one-hot token
        var token = new Tensor<float>(new[] { vocabSize });
        token[5] = 1.0f; // Token index 5

        var logits = model.GenerateStep(token);

        Assert.Equal(new[] { vocabSize }, logits.Shape);
        Assert.False(ContainsNaN(logits));
    }

    [Fact]
    public void GenerateStep_MultipleSteps_ProducesValidLogits()
    {
        int vocabSize = 20;
        var model = new MambaLanguageModel<float>(vocabSize, 16, 2, 4, maxSeqLength: 8);
        model.InitializeStateCache();

        var random = new Random(42);
        for (int step = 0; step < 4; step++)
        {
            var token = new Tensor<float>(new[] { vocabSize });
            token[random.Next(vocabSize)] = 1.0f;

            var logits = model.GenerateStep(token);

            Assert.Equal(new[] { vocabSize }, logits.Shape);
            Assert.False(ContainsNaN(logits));
        }
    }

    [Fact]
    public void GenerateStep_ThrowsWithoutStateCache()
    {
        var model = new MambaLanguageModel<float>(20, 16, 2, 4, maxSeqLength: 4);
        var token = new Tensor<float>(new[] { 20 });
        token[0] = 1.0f;

        Assert.Throws<InvalidOperationException>(() => model.GenerateStep(token));
    }

    [Fact]
    public void ResetState_ClearsStateCache()
    {
        var model = new MambaLanguageModel<float>(20, 16, 2, 4, maxSeqLength: 4);
        var cache = model.InitializeStateCache();

        var token = new Tensor<float>(new[] { 20 });
        token[3] = 1.0f;
        model.GenerateStep(token);

        model.ResetState();

        // Cache should be reset (no cached layers)
        Assert.Equal(0, cache.CachedLayerCount);
    }

    [Fact]
    public void MultiLayerModel_ProducesNonTrivialOutput()
    {
        int seqLen = 4;
        int vocabSize = 20;
        int modelDim = 16;

        var model = new MambaLanguageModel<float>(
            vocabSize, modelDim, numLayers: 4, stateDimension: 4, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output = model.Forward(input);

        // Output should not be all zeros or all same value
        var arr = output.ToArray();
        bool hasVariation = false;
        for (int i = 1; i < arr.Length; i++)
        {
            if (MathF.Abs(arr[i] - arr[0]) > 1e-6f)
            {
                hasVariation = true;
                break;
            }
        }
        Assert.True(hasVariation, "Output should have variation across positions and vocab");
    }

    #region Helpers

    private static Tensor<float> CreateOneHotInput(int batchSize, int seqLen, int vocabSize, int seed = 42)
    {
        var tensor = new Tensor<float>(new[] { batchSize, seqLen, vocabSize });
        var random = new Random(seed);

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokenIdx = random.Next(vocabSize);
                tensor[new[] { b, s, tokenIdx }] = 1.0f;
            }
        }
        return tensor;
    }

    private static Tensor<double> CreateOneHotDoubleInput(int batchSize, int seqLen, int vocabSize, int seed = 42)
    {
        var tensor = new Tensor<double>(new[] { batchSize, seqLen, vocabSize });
        var random = new Random(seed);

        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokenIdx = random.Next(vocabSize);
                tensor[new[] { b, s, tokenIdx }] = 1.0;
            }
        }
        return tensor;
    }

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return tensor;
    }

    private static bool ContainsNaN(Tensor<float> tensor)
    {
        foreach (var value in tensor.ToArray())
        {
            if (float.IsNaN(value)) return true;
        }
        return false;
    }

    private static bool ContainsNaNDouble(Tensor<double> tensor)
    {
        foreach (var value in tensor.ToArray())
        {
            if (double.IsNaN(value)) return true;
        }
        return false;
    }

    #endregion
}
