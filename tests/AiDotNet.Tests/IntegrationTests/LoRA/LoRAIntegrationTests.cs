using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.LoRA;

/// <summary>
/// Integration tests for the LoRA (Low-Rank Adaptation) module.
/// Tests cover LoRALayer, various adapters, and parameter efficiency.
/// </summary>
[Collection("NonParallelIntegration")]
public class LoRAIntegrationTests
{
    private const int InputSize = 64;
    private const int OutputSize = 32;
    private const int Rank = 8;
    private const double Alpha = 16.0;
    private const double LearningRate = 0.01;

    #region LoRALayer Tests

    [Fact(Timeout = 120000)]
    public async Task LoRALayer_Initialize_CreatesCorrectDimensions()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);

        Assert.Equal(Rank, layer.Rank);
        Assert.Equal(Alpha, Convert.ToDouble(layer.Alpha));
        Assert.Equal(Alpha / Rank, Convert.ToDouble(layer.Scaling), 6);

        // Parameter count = A (inputSize * rank) + B (rank * outputSize)
        int expectedParams = (InputSize * Rank) + (Rank * OutputSize);
        Assert.Equal(expectedParams, layer.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task LoRALayer_Initialize_InvalidRank_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoRALayer<double>(InputSize, OutputSize, 0, Alpha));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoRALayer<double>(InputSize, OutputSize, -1, Alpha));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoRALayer<double>(10, 10, 20, Alpha)); // rank > min(in, out)
    }

    [Fact(Timeout = 120000)]
    public async Task LoRALayer_Forward_ProducesCorrectOutputShape()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(OutputSize, output.Shape[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task LoRALayer_Forward_BatchInput_ProducesCorrectOutputShape()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);
        int batchSize = 4;
        var input = CreateTensor(batchSize, InputSize);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(OutputSize, output.Shape[1]);
    }



    [Fact(Timeout = 120000)]
    public async Task LoRALayer_MergeWeights_ProducesCorrectDimensions()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);

        var mergedWeights = layer.MergeWeights();

        Assert.Equal(InputSize, mergedWeights.Rows);
        Assert.Equal(OutputSize, mergedWeights.Columns);
    }

    [Fact(Timeout = 120000)]
    public async Task LoRALayer_GetSetParameters_RoundTrip()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);

        var originalParams = layer.GetParameters();
        var modifiedParams = originalParams.Clone();

        // Modify a parameter
        modifiedParams[0] = 999.0;

        layer.SetParameters(modifiedParams);
        var retrievedParams = layer.GetParameters();

        Assert.Equal(999.0, retrievedParams[0], 6);
    }


    [Fact(Timeout = 120000)]
    public async Task LoRALayer_SupportsJitCompilation_ReturnsTrueWhenInitialized()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);

        Assert.True(layer.SupportsJitCompilation);
    }

    #endregion

    #region StandardLoRAAdapter Tests

    [Fact(Timeout = 120000)]
    public async Task StandardLoRAAdapter_WrapsDenseLayer_Correctly()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize, (IActivationFunction<double>)new ReLUActivation<double>());
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
        Assert.Equal(Rank, adapter.Rank);
        Assert.Equal(Alpha, adapter.Alpha);
        Assert.True(adapter.IsBaseLayerFrozen);
    }

    [Fact(Timeout = 120000)]
    public async Task StandardLoRAAdapter_Forward_CombinesBaseAndLoRA()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize, (IActivationFunction<double>)new ReLUActivation<double>());
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = adapter.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(OutputSize, output.Shape[1]);

        // Verify output is finite
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]), "Output contains NaN");
            Assert.False(double.IsInfinity(output[i]), "Output contains Infinity");
        }
    }

    [Fact(Timeout = 120000)]
    public async Task StandardLoRAAdapter_FrozenBaseLayer_OnlyTrainsLoRA()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha, freezeBaseLayer: true);

        // Only LoRA parameters should be trainable
        int loraParamCount = (InputSize * Rank) + (Rank * OutputSize);
        Assert.Equal(loraParamCount, adapter.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task StandardLoRAAdapter_UnfrozenBaseLayer_TrainsBoth()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha, freezeBaseLayer: false);

        // Both base and LoRA parameters should be trainable
        int baseParamCount = baseLayer.ParameterCount;
        int loraParamCount = (InputSize * Rank) + (Rank * OutputSize);
        Assert.Equal(baseParamCount + loraParamCount, adapter.ParameterCount);
    }

    [Fact(Timeout = 120000)]
    public async Task StandardLoRAAdapter_MergeToOriginalLayer_ProducesFunctionalLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        // Get output from adapter
        var adapterOutput = adapter.Forward(input);

        // Merge and get output from merged layer
        var mergedLayer = adapter.MergeToOriginalLayer();
        var mergedOutput = mergedLayer.Forward(input);

        // Outputs should be similar (within numerical precision)
        Assert.Equal(adapterOutput.Length, mergedOutput.Length);
        for (int i = 0; i < adapterOutput.Length; i++)
        {
            Assert.Equal(adapterOutput[i], mergedOutput[i], 4);
        }
    }



    #endregion

    #region LoRA Adapter Variants Tests

    [Fact(Timeout = 120000)]
    public async Task QLoRAAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new QLoRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
        Assert.Equal(Rank, adapter.Rank);
    }

    [Fact(Timeout = 120000)]
    public async Task QLoRAAdapter_Forward_ProducesValidOutput()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new QLoRAAdapter<double>(baseLayer, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = adapter.Forward(input);

        Assert.Equal(OutputSize, output.Shape[1]);
        AssertTensorFinite(output);
    }

    [Fact(Timeout = 120000)]
    public async Task DoRAAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new DoRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
        Assert.Equal(Rank, adapter.Rank);
    }

    [Fact(Timeout = 120000)]
    public async Task DoRAAdapter_Forward_ProducesValidOutput()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new DoRAAdapter<double>(baseLayer, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = adapter.Forward(input);

        Assert.Equal(OutputSize, output.Shape[1]);
        AssertTensorFinite(output);
    }

    [Fact(Timeout = 120000)]
    public async Task AdaLoRAAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new AdaLoRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
        Assert.Equal(Rank, adapter.Rank);
    }

    [Fact(Timeout = 120000)]
    public async Task AdaLoRAAdapter_Forward_ProducesValidOutput()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new AdaLoRAAdapter<double>(baseLayer, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = adapter.Forward(input);

        Assert.Equal(OutputSize, output.Shape[1]);
        AssertTensorFinite(output);
    }

    [Fact(Timeout = 120000)]
    public async Task VeRAAdapter_Initialize_CorrectlyWrapsLayer()
    {
        // VeRA requires shared matrices to be initialized first
        VeRAAdapter<double>.InitializeSharedMatrices(InputSize, OutputSize, Rank);

        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new VeRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
    }

    [Fact(Timeout = 120000)]
    public async Task LoKrAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new LoKrAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
    }

    [Fact(Timeout = 120000)]
    public async Task LoHaAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new LoHaAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
    }

    #endregion

    #region Parameter Efficiency Tests

    [Fact(Timeout = 120000)]
    public async Task LoRA_ParameterReduction_SignificantForLargeLayers()
    {
        int largeInputSize = 1024;
        int largeOutputSize = 1024;
        int rank = 8;

        var baseLayer = new DenseLayer<double>(largeInputSize, largeOutputSize);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, rank, freezeBaseLayer: true);

        int fullParams = largeInputSize * largeOutputSize + largeOutputSize; // weights + bias
        int loraParams = adapter.ParameterCount;

        // LoRA should have significantly fewer parameters
        double reductionRatio = (double)loraParams / fullParams;
        Assert.True(reductionRatio < 0.05, $"LoRA should reduce parameters by >95%, but ratio was {reductionRatio:P2}");
    }

    [Fact(Timeout = 120000)]
    public async Task LoRA_ParameterCount_MatchesFormula()
    {
        int inputSize = 512;
        int outputSize = 256;
        int rank = 16;

        var layer = new LoRALayer<double>(inputSize, outputSize, rank);

        // Formula: A (inputSize * rank) + B (rank * outputSize)
        int expectedParams = (inputSize * rank) + (rank * outputSize);
        Assert.Equal(expectedParams, layer.ParameterCount);
    }

    #endregion

    #region Training Workflow Tests


    #endregion

    #region Edge Cases Tests

    [Fact(Timeout = 120000)]
    public async Task LoRALayer_VerySmallRank_StillWorks()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, rank: 1, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = layer.Forward(input);

        Assert.Equal(OutputSize, output.Shape[1]);
        AssertTensorFinite(output);
    }

    [Fact(Timeout = 120000)]
    public async Task LoRALayer_RankEqualsMinDimension_Works()
    {
        int small = 8;
        int large = 64;

        // Rank can equal the smaller dimension
        var layer = new LoRALayer<double>(large, small, rank: small, Alpha);
        var input = CreateTensor(1, large);

        var output = layer.Forward(input);

        Assert.Equal(small, output.Shape[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task LoRALayer_LargeBatch_HandlesCorrectly()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);
        int batchSize = 128;
        var input = CreateTensor(batchSize, InputSize);

        var output = layer.Forward(input);

        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(OutputSize, output.Shape[1]);
    }

    [Fact(Timeout = 120000)]
    public async Task StandardLoRAAdapter_NullBaseLayer_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new StandardLoRAAdapter<double>(null!, Rank, Alpha));
    }

    [Fact(Timeout = 120000)]
    public async Task LoRALayer_DefaultAlpha_EqualsRank()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, alpha: -1);

        Assert.Equal(Rank, Convert.ToDouble(layer.Alpha), 6);
        Assert.Equal(1.0, Convert.ToDouble(layer.Scaling), 6);
    }

    #endregion

    #region Multiple Adapter Variants Forward Pass Tests

    [Fact(Timeout = 120000)]
    public async Task AllAdapters_ForwardPass_ProducesValidOutput()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);

        // Initialize shared matrices for VeRA and DVoRA
        VeRAAdapter<double>.InitializeSharedMatrices(InputSize, OutputSize, Rank);
        DVoRAAdapter<double>.InitializeSharedMatrices(InputSize, OutputSize, Rank);

        // Note: MoRA requires square layers; LoRAXS requires SVD initialization with pretrained weights
        var adapters = new List<(string Name, LoRAAdapterBase<double> Adapter, int ExpectedOutputSize)>
        {
            ("Standard", new StandardLoRAAdapter<double>(new DenseLayer<double>(InputSize, OutputSize), Rank, Alpha), OutputSize),
            ("DoRA", new DoRAAdapter<double>(new DenseLayer<double>(InputSize, OutputSize), Rank, Alpha), OutputSize),
            ("VeRA", new VeRAAdapter<double>(new DenseLayer<double>(InputSize, OutputSize), Rank, Alpha), OutputSize),
            ("DVoRA", new DVoRAAdapter<double>(new DenseLayer<double>(InputSize, OutputSize), Rank, Alpha), OutputSize),
            ("LoKr", new LoKrAdapter<double>(new DenseLayer<double>(InputSize, OutputSize), Rank, Alpha), OutputSize),
            ("LoHa", new LoHaAdapter<double>(new DenseLayer<double>(InputSize, OutputSize), Rank, Alpha), OutputSize),
            ("LoRAFA", new LoRAFAAdapter<double>(new DenseLayer<double>(InputSize, OutputSize), Rank, Alpha), OutputSize),
            ("PiSSA", new PiSSAAdapter<double>(new DenseLayer<double>(InputSize, OutputSize), Rank, Alpha), OutputSize),
            ("MoRA", new MoRAAdapter<double>(new DenseLayer<double>(InputSize, InputSize), Rank, Alpha), InputSize), // MoRA requires square
        };

        foreach (var (name, adapter, expectedOutputSize) in adapters)
        {
            var testInput = CreateTensor(1, adapter.BaseLayer.GetInputShape()[0]);
            var output = adapter.Forward(testInput);

            Assert.Equal(expectedOutputSize, output.Shape[1]);
            AssertTensorFinite(output, $"{name} adapter output");
        }
    }

    #endregion

    #region Helper Methods

    private static Tensor<double> CreateTensor(int batchSize, int featureSize)
    {
        var data = new Vector<double>(batchSize * featureSize);
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < data.Length; i++)
        {
            data[i] = random.NextDouble() * 2 - 1; // Range [-1, 1]
        }

        return new Tensor<double>(new[] { batchSize, featureSize }, data);
    }

    private static void AssertTensorFinite(Tensor<double> tensor, string context = "Tensor")
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            Assert.False(double.IsNaN(tensor[i]), $"{context} contains NaN at index {i}");
            Assert.False(double.IsInfinity(tensor[i]), $"{context} contains Infinity at index {i}");
        }
    }

    #endregion
}
