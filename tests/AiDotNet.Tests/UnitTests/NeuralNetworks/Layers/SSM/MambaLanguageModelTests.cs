using AiDotNet.Enums;
using AiDotNet.Models;
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
    private static NeuralNetworkArchitecture<float> CreateArch(int vocabSize = 100)
    {
        return new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.TextGeneration,
            inputSize: vocabSize,
            outputSize: vocabSize);
    }

    private static NeuralNetworkArchitecture<double> CreateDoubleArch(int vocabSize = 30)
    {
        return new NeuralNetworkArchitecture<double>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.TextGeneration,
            inputSize: vocabSize,
            outputSize: vocabSize);
    }

    [Fact]
    public void Constructor_ValidParameters_CreatesModel()
    {
        var model = new MambaLanguageModel<float>(
            CreateArch(),
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
            new MambaLanguageModel<float>(CreateArch(1), vocabSize: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(CreateArch(), vocabSize: 100, modelDimension: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenNumLayersNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(CreateArch(), vocabSize: 100, numLayers: 0));
    }

    [Fact]
    public void Constructor_ThrowsWhenStateDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(CreateArch(), vocabSize: 100, stateDimension: 0));
    }

    [Fact]
    public void Predict_3D_ProducesCorrectOutputShape()
    {
        int batchSize = 2;
        int seqLen = 4;
        int vocabSize = 50;
        int modelDim = 32;

        var model = new MambaLanguageModel<float>(
            CreateArch(vocabSize),
            vocabSize, modelDim, numLayers: 2, stateDimension: 8, maxSeqLength: seqLen);

        var input = CreateOneHotInput(batchSize, seqLen, vocabSize);
        var output = model.Predict(input);

        Assert.Equal(new[] { batchSize, seqLen, vocabSize }, output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Predict_2D_ProducesCorrectOutputShape()
    {
        int seqLen = 4;
        int vocabSize = 50;
        int modelDim = 32;

        var model = new MambaLanguageModel<float>(
            CreateArch(vocabSize),
            vocabSize, modelDim, numLayers: 2, stateDimension: 8, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize).Reshape(seqLen, vocabSize);
        var output = model.Predict(input);

        Assert.Equal(new[] { seqLen, vocabSize }, output.Shape.ToArray());
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void Backpropagate_ProducesValidGradients()
    {
        int seqLen = 4;
        int vocabSize = 50;
        int modelDim = 32;

        var model = new MambaLanguageModel<float>(
            CreateArch(vocabSize),
            vocabSize, modelDim, numLayers: 2, stateDimension: 8, maxSeqLength: seqLen);

        model.SetTrainingMode(true);
        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output = model.Predict(input);
        model.SetTrainingMode(true); // Re-enable after Predict set it to false
        var grad = CreateRandomTensor(output.Shape.ToArray());
        var inputGrad = model.Backpropagate(grad);

        Assert.Equal(input.Shape.ToArray(), inputGrad.Shape.ToArray());
        Assert.False(ContainsNaN(inputGrad));
    }

    [Fact]
    public void Backpropagate_ThrowsWithoutTrainingMode()
    {
        var model = new MambaLanguageModel<float>(
            CreateArch(50), 50, 32, 2, 8, maxSeqLength: 4);
        var grad = CreateRandomTensor(new[] { 1, 4, 50 });

        Assert.Throws<InvalidOperationException>(() => model.Backpropagate(grad));
    }

    [Fact]
    public void Train_ForwardBackwardUpdate_NoErrors()
    {
        int seqLen = 4;
        int vocabSize = 30;
        int modelDim = 16;

        var model = new MambaLanguageModel<float>(
            CreateArch(vocabSize),
            vocabSize, modelDim, numLayers: 2, stateDimension: 4, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var expected = CreateOneHotInput(1, seqLen, vocabSize, seed: 99);

        model.Train(input, expected);

        // Verify model still produces valid output after parameter update
        model.ResetState();
        var output2 = model.Predict(input);
        Assert.Equal(new[] { 1, seqLen, vocabSize }, output2.Shape.ToArray());
        Assert.False(ContainsNaN(output2));
    }

    [Fact]
    public void GetParameters_SetParameters_RoundTrip()
    {
        var model = new MambaLanguageModel<float>(
            CreateArch(50),
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
        var model = new MambaLanguageModel<float>(
            CreateArch(50), 50, 32, 2, 8, maxSeqLength: 4);
        Assert.Throws<ArgumentException>(() => model.SetParameters(new Vector<float>(10)));
    }

    [Fact]
    public void Predict_DeterministicWithSameParameters()
    {
        int seqLen = 4;
        int vocabSize = 30;
        int modelDim = 16;

        var model1 = new MambaLanguageModel<float>(
            CreateArch(vocabSize),
            vocabSize, modelDim, numLayers: 2, stateDimension: 4, maxSeqLength: seqLen);
        var model2 = new MambaLanguageModel<float>(
            CreateArch(vocabSize),
            vocabSize, modelDim, numLayers: 2, stateDimension: 4, maxSeqLength: seqLen);

        model2.SetParameters(model1.GetParameters());

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output1 = model1.Predict(input);
        var output2 = model2.Predict(input);

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
        var model = new MambaLanguageModel<float>(
            CreateArch(30), 30, 16, 2, 4, maxSeqLength: 4);
        var input = CreateOneHotInput(1, 4, 30);

        model.Predict(input);
        model.ResetState();

        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.False(ContainsNaN(output));
    }

    [Fact]
    public void SupportsTraining_ReturnsTrue()
    {
        var model = new MambaLanguageModel<float>(
            CreateArch(30), 30, 16, 2, 4, maxSeqLength: 4);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void GetModelMetadata_ContainsExpectedKeys()
    {
        var model = new MambaLanguageModel<float>(
            CreateArch(100), 100, 64, 4, 16, maxSeqLength: 32);
        var metadata = model.GetModelMetadata();


        Assert.True(metadata.AdditionalInfo.ContainsKey("VocabSize"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("ModelDimension"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("NumLayers"));
        Assert.True(metadata.AdditionalInfo.ContainsKey("StateDimension"));
        Assert.Equal(100, metadata.AdditionalInfo["VocabSize"]);
        Assert.Equal(64, metadata.AdditionalInfo["ModelDimension"]);
        Assert.Equal(4, metadata.AdditionalInfo["NumLayers"]);
    }

    [Fact]
    public void Predict_Double_ProducesValidOutput()
    {
        int seqLen = 4;
        int vocabSize = 30;

        var model = new MambaLanguageModel<double>(
            CreateDoubleArch(vocabSize),
            vocabSize, 16, numLayers: 2, stateDimension: 4, maxSeqLength: seqLen);

        var input = CreateOneHotDoubleInput(1, seqLen, vocabSize);
        var output = model.Predict(input);

        Assert.Equal(new[] { 1, seqLen, vocabSize }, output.Shape.ToArray());
        Assert.False(ContainsNaNDouble(output));
    }

    [Fact]
    public void MultiLayerModel_ProducesNonTrivialOutput()
    {
        int seqLen = 4;
        int vocabSize = 20;
        int modelDim = 16;

        var model = new MambaLanguageModel<float>(
            CreateArch(vocabSize),
            vocabSize, modelDim, numLayers: 4, stateDimension: 4, maxSeqLength: seqLen);

        var input = CreateOneHotInput(1, seqLen, vocabSize);
        var output = model.Predict(input);

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
