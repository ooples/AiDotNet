using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using Xunit;
using System;
using System.Threading.Tasks;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.Layers.SSM;

/// <summary>
/// Integration tests for <see cref="MambaLanguageModel{T}"/>.
/// Tests full forward-backward-parameter round-trips and multi-layer compositions.
/// </summary>
// Training and per-token Step tests run real Mamba forward/backward through the process-wide
// AiDotNetEngine.Current; serialize via the RealModelInference collection so parallel tests mutating that
// global engine cannot defeat the CpuEngine pin mid-run (documented GPU-autodetect flake).
[Collection("RealModelInference")]
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

    [Fact(Timeout = 120000)]
    public async Task Constructor_ValidParameters_CreatesModel()
    {
        var model = new MambaLanguageModel<float>(
            CreateArch(),
            vocabSize: 100, modelDimension: 32, numLayers: 2, stateDimension: 8);

        Assert.Equal(100, model.VocabSize);
        Assert.Equal(32, model.ModelDimension);
        Assert.Equal(2, model.NumLayers);
        Assert.Equal(8, model.StateDimension);
    }

    [Fact(Timeout = 120000)]
    public async Task Constructor_ThrowsWhenVocabSizeNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(CreateArch(1), vocabSize: 0));
    }

    [Fact(Timeout = 120000)]
    public async Task Constructor_ThrowsWhenModelDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(CreateArch(), vocabSize: 100, modelDimension: 0));
    }

    [Fact(Timeout = 120000)]
    public async Task Constructor_ThrowsWhenNumLayersNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(CreateArch(), vocabSize: 100, numLayers: 0));
    }

    [Fact(Timeout = 120000)]
    public async Task Constructor_ThrowsWhenStateDimensionNotPositive()
    {
        Assert.Throws<ArgumentException>(() =>
            new MambaLanguageModel<float>(CreateArch(), vocabSize: 100, stateDimension: 0));
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_3D_ProducesCorrectOutputShape()
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

    [Fact(Timeout = 120000)]
    public async Task Predict_2D_ProducesCorrectOutputShape()
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



    [Fact(Timeout = 120000)]
    public async Task Train_ForwardBackwardUpdate_NoErrors()
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

    [Fact(Timeout = 120000)]
    public async Task GetParameters_SetParameters_RoundTrip()
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

    [Fact(Timeout = 120000)]
    public async Task SetParameters_ThrowsOnWrongLength()
    {
        var model = new MambaLanguageModel<float>(
            CreateArch(50), 50, 32, 2, 8, maxSeqLength: 4);
        Assert.Throws<ArgumentException>(() => model.SetParameters(new Vector<float>(10)));
    }

    [Fact(Timeout = 120000)]
    public async Task Predict_DeterministicWithSameParameters()
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

    [Fact(Timeout = 120000)]
    public async Task ResetState_AllowsReuse()
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

    [Fact(Timeout = 120000)]
    public async Task SupportsTraining_ReturnsTrue()
    {
        var model = new MambaLanguageModel<float>(
            CreateArch(30), 30, 16, 2, 4, maxSeqLength: 4);
        Assert.True(model.SupportsTraining);
    }

    [Fact(Timeout = 120000)]
    public async Task GetModelMetadata_ContainsExpectedKeys()
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

    [Fact(Timeout = 120000)]
    public async Task Predict_Double_ProducesValidOutput()
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

    [Fact(Timeout = 120000)]
    public async Task MultiLayerModel_ProducesNonTrivialOutput()
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

    [Fact(Timeout = 120000)]
    public async Task Step_Incremental_MatchesFullSequenceForward()
    {
        // The KV-cache fast path (per-token Step) must be mathematically equivalent to the parallel
        // full-sequence selective scan (Gu & Dao 2023): feeding tokens one at a time while carrying the
        // recurrent state must reproduce Predict's logits at every position. seqLen > conv kernel (4)
        // ensures the causal-conv window is fully exercised.
        await Task.CompletedTask;
        int seqLen = 5;
        int vocabSize = 12;
        int modelDim = 16;

        var model = new MambaLanguageModel<double>(
            CreateDoubleArch(vocabSize),
            vocabSize, modelDim, numLayers: 2, stateDimension: 8, maxSeqLength: seqLen);

        var input = CreateOneHotDoubleInput(1, seqLen, vocabSize, seed: 7);

        // Full-sequence (parallel selective scan) reference.
        model.ResetState();
        var full = model.Predict(input); // [1, seqLen, vocab]

        // Incremental: feed one token at a time carrying KV-cache state.
        var state = model.CreateStepState(batchSize: 1);
        for (int t = 0; t < seqLen; t++)
        {
            var token = SliceTimeStep(input, t, vocabSize); // [1, 1, vocab]
            var stepLogits = model.Step(token, state);      // [1, 1, vocab]

            for (int v = 0; v < vocabSize; v++)
            {
                double expected = full[new[] { 0, t, v }];
                double actual = stepLogits[new[] { 0, 0, v }];
                Assert.True(System.Math.Abs(expected - actual) < 1e-8,
                    $"Incremental Step diverged from full Forward at t={t}, v={v}: full={expected:G9} vs step={actual:G9}");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Step_Incremental_MatchesFullSequenceForward_MultiLayer()
    {
        // Same equivalence guarantee with more blocks, confirming per-block state threads correctly.
        await Task.CompletedTask;
        int seqLen = 6;
        int vocabSize = 10;
        int modelDim = 16;

        var model = new MambaLanguageModel<double>(
            CreateDoubleArch(vocabSize),
            vocabSize, modelDim, numLayers: 4, stateDimension: 4, maxSeqLength: seqLen);

        var input = CreateOneHotDoubleInput(1, seqLen, vocabSize, seed: 13);

        model.ResetState();
        var full = model.Predict(input);

        var state = model.CreateStepState(batchSize: 1);
        for (int t = 0; t < seqLen; t++)
        {
            var stepLogits = model.Step(SliceTimeStep(input, t, vocabSize), state);
            for (int v = 0; v < vocabSize; v++)
            {
                double expected = full[new[] { 0, t, v }];
                double actual = stepLogits[new[] { 0, 0, v }];
                Assert.True(System.Math.Abs(expected - actual) < 1e-8,
                    $"Multi-layer Step diverged at t={t}, v={v}: full={expected:G9} vs step={actual:G9}");
            }
        }
    }

    [Fact(Timeout = 120000)]
    public async Task Training_ChangesParameters_AndReducesLoss()
    {
        // Learning invariant: training must actually move parameters and reduce loss. This guards the
        // regression where MambaBlock registered its trainable parameters only inside UpdateParameters
        // (never reached by the tape path) and omitted _aLog/_dParam — making every Train() a silent no-op.
        await Task.CompletedTask;
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            int seqLen = 5;
            int vocabSize = 20;
            int modelDim = 16;

            var model = new MambaLanguageModel<double>(
                CreateDoubleArch(vocabSize), vocabSize, modelDim, numLayers: 2, stateDimension: 4, maxSeqLength: seqLen);

            var input = CreateOneHotDoubleInput(1, seqLen, vocabSize, seed: 5);
            var target = CreateOneHotDoubleInput(1, seqLen, vocabSize, seed: 6);

            model.Predict(input); // warmup: materialize lazy params before snapshotting
            var before = model.GetParameters().ToArray();

            model.Train(input, target);
            var earlyLoss = Convert.ToDouble(model.GetLastLoss());
            for (var i = 0; i < 40; i++)
            {
                model.Train(input, target);
            }

            var lateLoss = Convert.ToDouble(model.GetLastLoss());
            var after = model.GetParameters().ToArray();

            var changed = false;
            for (var i = 0; i < before.Length; i++)
            {
                if (Math.Abs(before[i] - after[i]) > 1e-12)
                {
                    changed = true;
                    break;
                }
            }

            Assert.True(changed, "Mamba parameters did not change after training (tape-trainable registration regression).");
            Assert.True(lateLoss < earlyLoss, $"Mamba training did not reduce loss: early={earlyLoss:G6}, late={lateLoss:G6}");
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }

    #region Helpers

    private static Tensor<double> SliceTimeStep(Tensor<double> input, int t, int vocabSize)
    {
        var token = new Tensor<double>(new[] { 1, 1, vocabSize });
        for (int v = 0; v < vocabSize; v++)
        {
            token[new[] { 0, 0, v }] = input[new[] { 0, t, v }];
        }
        return token;
    }

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
