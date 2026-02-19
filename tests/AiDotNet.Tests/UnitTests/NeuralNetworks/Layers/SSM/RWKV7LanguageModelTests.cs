using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Tests for <see cref="RWKV7Block{T}"/> and <see cref="RWKV7LanguageModel{T}"/>.
/// Validates RWKV-7 "Goose" architecture: WKV-7 dynamic state evolution, SiLU channel mixing,
/// group normalization, parallel training, sequential generation, and parameter management.
/// </summary>
public class RWKV7LanguageModelTests
{
    #region RWKV7Block Constructor Tests

    [Fact]
    public void Block_Constructor_ValidParameters_CreatesBlock()
    {
        var block = new RWKV7Block<float>(
            sequenceLength: 16, modelDimension: 32, numHeads: 4);

        Assert.Equal(32, block.ModelDimension);
        Assert.Equal(4, block.NumHeads);
        Assert.Equal(8, block.HeadDimension);
        Assert.True(block.ParameterCount > 0);
    }

    [Fact]
    public void Block_Constructor_ThrowsWhenSequenceLengthNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new RWKV7Block<float>(sequenceLength: 0, modelDimension: 32, numHeads: 4));
    }

    [Fact]
    public void Block_Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new RWKV7Block<float>(sequenceLength: 16, modelDimension: 0, numHeads: 4));
    }

    [Fact]
    public void Block_Constructor_ThrowsWhenNumHeadsNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new RWKV7Block<float>(sequenceLength: 16, modelDimension: 32, numHeads: 0));
    }

    [Fact]
    public void Block_Constructor_ThrowsWhenDimensionNotDivisibleByHeads()
    {
        Assert.Throws<ArgumentException>(() =>
            new RWKV7Block<float>(sequenceLength: 16, modelDimension: 33, numHeads: 4));
    }

    [Fact]
    public void Block_SupportsTraining_ReturnsTrue()
    {
        var block = new RWKV7Block<float>(16, 32, 4);
        Assert.True(block.SupportsTraining);
    }

    [Fact]
    public void Block_FFNDimension_MatchesMultiplier()
    {
        var block = new RWKV7Block<float>(16, 32, 4, ffnMultiplier: 3.5);
        Assert.Equal((int)(32 * 3.5), block.FFNDimension);
    }

    #endregion

    #region RWKV7Block Forward Tests

    [Fact]
    public void Block_Forward_3D_ProducesCorrectShape()
    {
        int batchSize = 2;
        int seqLen = 4;
        int modelDim = 32;

        var block = new RWKV7Block<float>(seqLen, modelDim, numHeads: 4);
        var input = CreateRandomTensor(new[] { batchSize, seqLen, modelDim });
        var output = block.Forward(input);

        Assert.Equal(new[] { batchSize, seqLen, modelDim }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Block_Forward_2D_ProducesCorrectShape()
    {
        int seqLen = 4;
        int modelDim = 32;

        var block = new RWKV7Block<float>(seqLen, modelDim, numHeads: 4);
        var input = CreateRandomTensor(new[] { seqLen, modelDim });
        var output = block.Forward(input);

        Assert.Equal(new[] { seqLen, modelDim }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Block_Forward_ProducesNonTrivialOutput()
    {
        var block = new RWKV7Block<float>(4, 16, 2);
        var input = CreateRandomTensor(new[] { 1, 4, 16 });
        var output = block.Forward(input);

        var arr = output.ToArray();
        bool hasVariation = false;
        for (int i = 1; i < Math.Min(arr.Length, 100); i++)
        {
            if (MathF.Abs(arr[i] - arr[0]) > 1e-8f)
            {
                hasVariation = true;
                break;
            }
        }
        Assert.True(hasVariation, "Block output should have variation");
    }

    #endregion

    #region RWKV7Block Backward Tests

    [Fact]
    public void Block_Backward_ProducesValidGradients()
    {
        var block = new RWKV7Block<float>(4, 32, 4);
        var input = CreateRandomTensor(new[] { 1, 4, 32 });
        var output = block.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = block.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Block_Backward_ThrowsWithoutForward()
    {
        var block = new RWKV7Block<float>(4, 32, 4);
        var grad = CreateRandomTensor(new[] { 1, 4, 32 });
        Assert.Throws<InvalidOperationException>(() => block.Backward(grad));
    }

    [Fact]
    public void Block_FullTrainingStep_NoErrors()
    {
        var block = new RWKV7Block<float>(4, 16, 2);
        var input = CreateRandomTensor(new[] { 1, 4, 16 });

        var output = block.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        block.Backward(grad);
        block.UpdateParameters(0.001f);

        block.ResetState();
        var output2 = block.Forward(input);
        Assert.Equal(output.Shape, output2.Shape);
        Assert.False(ContainsNaN(output2));
    }

    #endregion

    #region RWKV7Block Parameter Management

    [Fact]
    public void Block_GetSetParameters_RoundTrip()
    {
        var block = new RWKV7Block<float>(4, 32, 4);
        var params1 = block.GetParameters();
        Assert.True(params1.Length > 0);
        Assert.Equal(block.ParameterCount, params1.Length);

        block.SetParameters(params1);
        var params2 = block.GetParameters();

        for (int i = 0; i < params1.Length; i++)
            Assert.Equal(params1[i], params2[i]);
    }

    [Fact]
    public void Block_SetParameters_ThrowsOnWrongLength()
    {
        var block = new RWKV7Block<float>(4, 32, 4);
        Assert.Throws<ArgumentException>(() => block.SetParameters(new Vector<float>(10)));
    }

    [Fact]
    public void Block_GetMetadata_ContainsRWKV7()
    {
        var block = new RWKV7Block<float>(4, 32, 4);
        var metadata = block.GetMetadata();

        Assert.True(metadata.ContainsKey("Architecture"));
        Assert.Equal("RWKV-7", metadata["Architecture"]);
        Assert.Equal("32", metadata["ModelDimension"]);
        Assert.Equal("4", metadata["NumHeads"]);
    }

    #endregion

    #region RWKV7Block Recurrent State

    [Fact]
    public void Block_RecurrentState_InitiallyNull()
    {
        var block = new RWKV7Block<float>(4, 16, 2);
        Assert.Null(block.GetRecurrentState());
        Assert.Null(block.GetPreviousToken());
    }

    [Fact]
    public void Block_RecurrentState_PopulatedAfterForward()
    {
        var block = new RWKV7Block<float>(4, 16, 2);
        var input = CreateRandomTensor(new[] { 1, 4, 16 });
        block.Forward(input);

        Assert.NotNull(block.GetRecurrentState());
        Assert.NotNull(block.GetPreviousToken());
    }

    [Fact]
    public void Block_ResetState_ClearsRecurrentState()
    {
        var block = new RWKV7Block<float>(4, 16, 2);
        var input = CreateRandomTensor(new[] { 1, 4, 16 });
        block.Forward(input);

        block.ResetState();
        Assert.Null(block.GetRecurrentState());
        Assert.Null(block.GetPreviousToken());
    }

    #endregion

    #region RWKV7LanguageModel Constructor Tests

    [Fact]
    public void Model_Constructor_ValidParameters_CreatesModel()
    {
        var model = new RWKV7LanguageModel<float>(
            vocabSize: 100, modelDimension: 32, numLayers: 2, numHeads: 4);

        Assert.Equal(100, model.VocabSize);
        Assert.Equal(32, model.ModelDimension);
        Assert.Equal(2, model.NumLayers);
        Assert.Equal(4, model.NumHeads);
        Assert.Equal(3.5, model.FFNMultiplier);
    }

    [Fact]
    public void Model_Constructor_ThrowsWhenVocabSizeNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new RWKV7LanguageModel<float>(vocabSize: 0));
    }

    [Fact]
    public void Model_Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new RWKV7LanguageModel<float>(vocabSize: 100, modelDimension: 0));
    }

    [Fact]
    public void Model_Constructor_ThrowsWhenNumLayersNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new RWKV7LanguageModel<float>(vocabSize: 100, numLayers: 0));
    }

    [Fact]
    public void Model_Constructor_ThrowsWhenNumHeadsNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new RWKV7LanguageModel<float>(vocabSize: 100, numHeads: 0));
    }

    [Fact]
    public void Model_Constructor_ThrowsWhenDimensionNotDivisibleByHeads()
    {
        Assert.Throws<ArgumentException>(() =>
            new RWKV7LanguageModel<float>(vocabSize: 100, modelDimension: 33, numHeads: 4));
    }

    [Fact]
    public void Model_SupportsTraining_ReturnsTrue()
    {
        var model = new RWKV7LanguageModel<float>(30, 16, 2, 2);
        Assert.True(model.SupportsTraining);
    }

    #endregion

    #region RWKV7LanguageModel Forward Tests

    [Fact]
    public void Model_Forward_3D_ProducesCorrectOutputShape()
    {
        int batchSize = 2;
        int seqLen = 4;
        int vocabSize = 50;
        int modelDim = 32;

        var model = new RWKV7LanguageModel<float>(
            vocabSize, modelDim, numLayers: 2, numHeads: 4, maxSeqLength: seqLen);

        var input = CreateOneHotInput(batchSize, seqLen, vocabSize);
        var output = model.Forward(input);

        Assert.Equal(new[] { batchSize, seqLen, vocabSize }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Model_Forward_2D_ProducesCorrectOutputShape()
    {
        int seqLen = 4;
        int vocabSize = 50;
        int modelDim = 32;

        var model = new RWKV7LanguageModel<float>(
            vocabSize, modelDim, numLayers: 2, numHeads: 4, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize).Reshape(seqLen, vocabSize);
        var output = model.Forward(input);

        Assert.Equal(new[] { seqLen, vocabSize }, output.Shape);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Model_Forward_ProducesNonTrivialOutput()
    {
        int seqLen = 4;
        int vocabSize = 20;

        var model = new RWKV7LanguageModel<float>(
            vocabSize, 16, numLayers: 2, numHeads: 2, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output = model.Forward(input);

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

    #endregion

    #region RWKV7LanguageModel Backward Tests

    [Fact]
    public void Model_Backward_ProducesValidGradients()
    {
        int seqLen = 4;
        int vocabSize = 30;

        var model = new RWKV7LanguageModel<float>(
            vocabSize, 16, numLayers: 2, numHeads: 2, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output = model.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        var inputGrad = model.Backward(grad);

        Assert.Equal(input.Shape, inputGrad.Shape);
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Model_Backward_ThrowsWithoutForward()
    {
        var model = new RWKV7LanguageModel<float>(30, 16, 2, 2, maxSeqLength: 4);
        var grad = CreateRandomTensor(new[] { 1, 4, 30 });
        Assert.Throws<InvalidOperationException>(() => model.Backward(grad));
    }

    [Fact]
    public void Model_FullTrainingStep_ForwardBackwardUpdate_NoErrors()
    {
        int seqLen = 4;
        int vocabSize = 20;

        var model = new RWKV7LanguageModel<float>(
            vocabSize, 16, numLayers: 2, numHeads: 2, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output = model.Forward(input);
        var grad = CreateRandomTensor(output.Shape);
        model.Backward(grad);
        model.UpdateParameters(0.001f);

        model.ResetState();
        var output2 = model.Forward(input);
        Assert.Equal(output.Shape, output2.Shape);
        Assert.False(ContainsNaN(output2));
    }

    #endregion

    #region RWKV7LanguageModel Parameter Management

    [Fact]
    public void Model_GetParameters_SetParameters_RoundTrip()
    {
        var model = new RWKV7LanguageModel<float>(
            30, 16, numLayers: 2, numHeads: 2, maxSeqLength: 4);

        var params1 = model.GetParameters();
        Assert.True(params1.Length > 0);
        Assert.Equal(model.ParameterCount, params1.Length);

        model.SetParameters(params1);
        var params2 = model.GetParameters();

        Assert.Equal(params1.Length, params2.Length);
        for (int i = 0; i < params1.Length; i++)
            Assert.Equal(params1[i], params2[i]);
    }

    [Fact]
    public void Model_SetParameters_ThrowsOnWrongLength()
    {
        var model = new RWKV7LanguageModel<float>(30, 16, 2, 2, maxSeqLength: 4);
        Assert.Throws<ArgumentException>(() => model.SetParameters(new Vector<float>(10)));
    }

    [Fact]
    public void Model_Forward_DeterministicWithSameParameters()
    {
        int seqLen = 4;
        int vocabSize = 20;

        var model1 = new RWKV7LanguageModel<float>(
            vocabSize, 16, numLayers: 2, numHeads: 2, maxSeqLength: seqLen);
        var model2 = new RWKV7LanguageModel<float>(
            vocabSize, 16, numLayers: 2, numHeads: 2, maxSeqLength: seqLen);

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

    #endregion

    #region RWKV7LanguageModel Sequential Generation

    [Fact]
    public void Model_GenerateStep_ProducesValidLogits()
    {
        int vocabSize = 20;
        var model = new RWKV7LanguageModel<float>(vocabSize, 16, 2, 2, maxSeqLength: 8);
        model.InitializeGeneration();

        var token = new Tensor<float>(new[] { vocabSize });
        token[5] = 1.0f;

        var logits = model.GenerateStep(token);

        Assert.Equal(new[] { vocabSize }, logits.Shape);
        Assert.False(ContainsNaN(logits));
    }

    [Fact]
    public void Model_GenerateStep_MultipleSteps_ProducesValidLogits()
    {
        int vocabSize = 20;
        var model = new RWKV7LanguageModel<float>(vocabSize, 16, 2, 2, maxSeqLength: 8);
        model.InitializeGeneration();

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
    public void Model_GenerateStep_ThrowsWithoutInit()
    {
        var model = new RWKV7LanguageModel<float>(20, 16, 2, 2, maxSeqLength: 4);
        var token = new Tensor<float>(new[] { 20 });
        token[0] = 1.0f;

        Assert.Throws<InvalidOperationException>(() => model.GenerateStep(token));
    }

    [Fact]
    public void Model_IsGenerating_FalseByDefault()
    {
        var model = new RWKV7LanguageModel<float>(20, 16, 2, 2, maxSeqLength: 4);
        Assert.False(model.IsGenerating);
    }

    [Fact]
    public void Model_InitializeGeneration_SetsIsGenerating()
    {
        var model = new RWKV7LanguageModel<float>(20, 16, 2, 2, maxSeqLength: 4);
        model.InitializeGeneration();
        Assert.True(model.IsGenerating);
    }

    [Fact]
    public void Model_ResetState_ClearsGenerationMode()
    {
        var model = new RWKV7LanguageModel<float>(20, 16, 2, 2, maxSeqLength: 4);
        model.InitializeGeneration();

        var token = new Tensor<float>(new[] { 20 });
        token[3] = 1.0f;
        model.GenerateStep(token);

        model.ResetState();
        Assert.False(model.IsGenerating);
    }

    [Fact]
    public void Model_ResetState_AllowsReuse()
    {
        var model = new RWKV7LanguageModel<float>(20, 16, 2, 2, maxSeqLength: 4);
        var input = CreateOneHotInput(1, 4, 20);

        model.Forward(input);
        model.ResetState();

        var output = model.Forward(input);
        Assert.NotNull(output);
        Assert.False(ContainsNaN(output));
    }

    #endregion

    #region RWKV7LanguageModel Metadata

    [Fact]
    public void Model_GetMetadata_ContainsExpectedKeys()
    {
        var model = new RWKV7LanguageModel<float>(100, 64, 4, 8, maxSeqLength: 32);
        var metadata = model.GetMetadata();

        Assert.True(metadata.ContainsKey("VocabSize"));
        Assert.True(metadata.ContainsKey("ModelDimension"));
        Assert.True(metadata.ContainsKey("NumLayers"));
        Assert.True(metadata.ContainsKey("NumHeads"));
        Assert.True(metadata.ContainsKey("Architecture"));
        Assert.Equal("100", metadata["VocabSize"]);
        Assert.Equal("64", metadata["ModelDimension"]);
        Assert.Equal("4", metadata["NumLayers"]);
        Assert.Equal("8", metadata["NumHeads"]);
        Assert.Equal("RWKV-7-Goose", metadata["Architecture"]);
    }

    #endregion

    #region RWKV7LanguageModel Double Precision

    [Fact]
    public void Model_Forward_Double_ProducesValidOutput()
    {
        int seqLen = 4;
        int vocabSize = 20;

        var model = new RWKV7LanguageModel<double>(
            vocabSize, 16, numLayers: 2, numHeads: 2, maxSeqLength: seqLen);

        var input = CreateOneHotDoubleInput(1, seqLen, vocabSize);
        var output = model.Forward(input);

        Assert.Equal(new[] { 1, seqLen, vocabSize }, output.Shape);
        Assert.False(ContainsNaNDouble(output));
    }

    [Fact]
    public void Model_GenerateStep_Double_ProducesValidLogits()
    {
        int vocabSize = 20;
        var model = new RWKV7LanguageModel<double>(vocabSize, 16, 2, 2, maxSeqLength: 8);
        model.InitializeGeneration();

        var token = new Tensor<double>(new[] { vocabSize });
        token[5] = 1.0;

        var logits = model.GenerateStep(token);
        Assert.Equal(new[] { vocabSize }, logits.Shape);
        Assert.False(ContainsNaNDouble(logits));
    }

    #endregion

    #region RWKV7LanguageModel Multi-Layer

    [Fact]
    public void Model_MultiLayer_ProducesNonTrivialOutput()
    {
        int seqLen = 4;
        int vocabSize = 20;

        var model = new RWKV7LanguageModel<float>(
            vocabSize, 16, numLayers: 4, numHeads: 2, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output = model.Forward(input);

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
        Assert.True(hasVariation, "Multi-layer output should have variation");
    }

    #endregion

    #region Helpers

    private static Tensor<float> CreateOneHotInput(int batchSize, int seqLen, int vocabSize, int seed = 42)
    {
        var tensor = new Tensor<float>(new[] { batchSize, seqLen, vocabSize });
        var random = new Random(seed);

        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < seqLen; s++)
                tensor[new[] { b, s, random.Next(vocabSize) }] = 1.0f;

        return tensor;
    }

    private static Tensor<double> CreateOneHotDoubleInput(int batchSize, int seqLen, int vocabSize, int seed = 42)
    {
        var tensor = new Tensor<double>(new[] { batchSize, seqLen, vocabSize });
        var random = new Random(seed);

        for (int b = 0; b < batchSize; b++)
            for (int s = 0; s < seqLen; s++)
                tensor[new[] { b, s, random.Next(vocabSize) }] = 1.0;

        return tensor;
    }

    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var tensor = new Tensor<float>(shape);
        var random = new Random(seed);
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = (float)(random.NextDouble() * 2 - 1);
        return tensor;
    }

    private static bool ContainsNaN(Tensor<float> tensor)
    {
        foreach (var value in tensor.ToArray())
            if (float.IsNaN(value)) return true;
        return false;
    }

    private static bool ContainsNaNDouble(Tensor<double> tensor)
    {
        foreach (var value in tensor.ToArray())
            if (double.IsNaN(value)) return true;
        return false;
    }

    #endregion
}
