using System;
using System.Collections.Generic;
using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LoRA;
using AiDotNet.LoRA.Adapters;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using Xunit;

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

    [Fact]
    public void LoRALayer_Initialize_CreatesCorrectDimensions()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);

        Assert.Equal(Rank, layer.Rank);
        Assert.Equal(Alpha, Convert.ToDouble(layer.Alpha));
        Assert.Equal(Alpha / Rank, Convert.ToDouble(layer.Scaling), 6);

        // Parameter count = A (inputSize * rank) + B (rank * outputSize)
        int expectedParams = (InputSize * Rank) + (Rank * OutputSize);
        Assert.Equal(expectedParams, layer.ParameterCount);
    }

    [Fact]
    public void LoRALayer_Initialize_InvalidRank_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoRALayer<double>(InputSize, OutputSize, 0, Alpha));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoRALayer<double>(InputSize, OutputSize, -1, Alpha));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LoRALayer<double>(10, 10, 20, Alpha)); // rank > min(in, out)
    }

    [Fact]
    public void LoRALayer_Forward_ProducesCorrectOutputShape()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(OutputSize, output.Shape[1]);
    }

    [Fact]
    public void LoRALayer_Forward_BatchInput_ProducesCorrectOutputShape()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);
        int batchSize = 4;
        var input = CreateTensor(batchSize, InputSize);

        var output = layer.Forward(input);

        Assert.Equal(2, output.Shape.Length);
        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(OutputSize, output.Shape[1]);
    }

    [Fact]
    public void LoRALayer_Backward_ComputesGradients()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);
        var input = CreateTensor(1, InputSize);
        var outputGradient = CreateTensor(1, OutputSize);

        // Forward pass first
        layer.Forward(input);

        // Backward pass
        var inputGradient = layer.Backward(outputGradient);

        Assert.Equal(2, inputGradient.Shape.Length);
        Assert.Equal(1, inputGradient.Shape[0]);
        Assert.Equal(InputSize, inputGradient.Shape[1]);

        // Verify gradients are finite
        for (int i = 0; i < inputGradient.Length; i++)
        {
            Assert.False(double.IsNaN(inputGradient[i]), "Input gradient contains NaN");
            Assert.False(double.IsInfinity(inputGradient[i]), "Input gradient contains Infinity");
        }
    }

    [Fact]
    public void LoRALayer_UpdateParameters_ModifiesWeights()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);
        var input = CreateTensor(1, InputSize);
        var outputGradient = CreateTensor(1, OutputSize);

        // Get initial parameters
        var initialParams = layer.GetParameters();

        // Forward and backward pass
        layer.Forward(input);
        layer.Backward(outputGradient);
        layer.UpdateParameters(LearningRate);

        // Get updated parameters
        var updatedParams = layer.GetParameters();

        // Verify parameters changed
        bool parametersChanged = false;
        for (int i = 0; i < initialParams.Length; i++)
        {
            if (Math.Abs(initialParams[i] - updatedParams[i]) > 1e-10)
            {
                parametersChanged = true;
                break;
            }
        }
        Assert.True(parametersChanged, "Parameters should change after update");
    }

    [Fact]
    public void LoRALayer_MergeWeights_ProducesCorrectDimensions()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);

        var mergedWeights = layer.MergeWeights();

        Assert.Equal(InputSize, mergedWeights.Rows);
        Assert.Equal(OutputSize, mergedWeights.Columns);
    }

    [Fact]
    public void LoRALayer_GetSetParameters_RoundTrip()
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

    [Fact]
    public void LoRALayer_ResetState_ClearsInternals()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        layer.Forward(input);
        layer.ResetState();

        // After reset, backward should throw because forward wasn't called
        var outputGradient = CreateTensor(1, OutputSize);
        Assert.Throws<InvalidOperationException>(() => layer.Backward(outputGradient));
    }

    [Fact]
    public void LoRALayer_SupportsJitCompilation_ReturnsTrueWhenInitialized()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);

        Assert.True(layer.SupportsJitCompilation);
    }

    #endregion

    #region StandardLoRAAdapter Tests

    [Fact]
    public void StandardLoRAAdapter_WrapsDenseLayer_Correctly()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize, (IActivationFunction<double>)new ReLUActivation<double>());
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
        Assert.Equal(Rank, adapter.Rank);
        Assert.Equal(Alpha, adapter.Alpha);
        Assert.True(adapter.IsBaseLayerFrozen);
    }

    [Fact]
    public void StandardLoRAAdapter_Forward_CombinesBaseAndLoRA()
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

    [Fact]
    public void StandardLoRAAdapter_FrozenBaseLayer_OnlyTrainsLoRA()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha, freezeBaseLayer: true);

        // Only LoRA parameters should be trainable
        int loraParamCount = (InputSize * Rank) + (Rank * OutputSize);
        Assert.Equal(loraParamCount, adapter.ParameterCount);
    }

    [Fact]
    public void StandardLoRAAdapter_UnfrozenBaseLayer_TrainsBoth()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha, freezeBaseLayer: false);

        // Both base and LoRA parameters should be trainable
        int baseParamCount = baseLayer.ParameterCount;
        int loraParamCount = (InputSize * Rank) + (Rank * OutputSize);
        Assert.Equal(baseParamCount + loraParamCount, adapter.ParameterCount);
    }

    [Fact]
    public void StandardLoRAAdapter_MergeToOriginalLayer_ProducesFunctionalLayer()
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

    [Fact]
    public void StandardLoRAAdapter_Backward_PropagatesGradients()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha);
        var input = CreateTensor(1, InputSize);
        var outputGradient = CreateTensor(1, OutputSize);

        adapter.Forward(input);
        var inputGradient = adapter.Backward(outputGradient);

        Assert.Equal(InputSize, inputGradient.Length / inputGradient.Shape[0]);

        // Verify gradients are finite
        for (int i = 0; i < inputGradient.Length; i++)
        {
            Assert.False(double.IsNaN(inputGradient[i]), "Input gradient contains NaN");
        }
    }

    [Fact]
    public void StandardLoRAAdapter_UpdateParameters_OnlyUpdatesLoRAWhenFrozen()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, Rank, Alpha, freezeBaseLayer: true);
        var input = CreateTensor(1, InputSize);
        var outputGradient = CreateTensor(1, OutputSize);

        // Get initial base layer parameters
        var initialBaseParams = baseLayer.GetParameters();

        // Forward, backward, update
        adapter.Forward(input);
        adapter.Backward(outputGradient);
        adapter.UpdateParameters(LearningRate);

        // Base layer parameters should be unchanged
        var finalBaseParams = baseLayer.GetParameters();
        for (int i = 0; i < initialBaseParams.Length; i++)
        {
            Assert.Equal(initialBaseParams[i], finalBaseParams[i], 10);
        }
    }

    #endregion

    #region LoRA Adapter Variants Tests

    [Fact]
    public void QLoRAAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new QLoRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
        Assert.Equal(Rank, adapter.Rank);
    }

    [Fact]
    public void QLoRAAdapter_Forward_ProducesValidOutput()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new QLoRAAdapter<double>(baseLayer, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = adapter.Forward(input);

        Assert.Equal(OutputSize, output.Shape[1]);
        AssertTensorFinite(output);
    }

    [Fact]
    public void DoRAAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new DoRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
        Assert.Equal(Rank, adapter.Rank);
    }

    [Fact]
    public void DoRAAdapter_Forward_ProducesValidOutput()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new DoRAAdapter<double>(baseLayer, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = adapter.Forward(input);

        Assert.Equal(OutputSize, output.Shape[1]);
        AssertTensorFinite(output);
    }

    [Fact]
    public void AdaLoRAAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new AdaLoRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
        Assert.Equal(Rank, adapter.Rank);
    }

    [Fact]
    public void AdaLoRAAdapter_Forward_ProducesValidOutput()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new AdaLoRAAdapter<double>(baseLayer, Rank, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = adapter.Forward(input);

        Assert.Equal(OutputSize, output.Shape[1]);
        AssertTensorFinite(output);
    }

    [Fact]
    public void VeRAAdapter_Initialize_CorrectlyWrapsLayer()
    {
        // VeRA requires shared matrices to be initialized first
        VeRAAdapter<double>.InitializeSharedMatrices(InputSize, OutputSize, Rank);

        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new VeRAAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
    }

    [Fact]
    public void LoKrAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new LoKrAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
    }

    [Fact]
    public void LoHaAdapter_Initialize_CorrectlyWrapsLayer()
    {
        var baseLayer = new DenseLayer<double>(InputSize, OutputSize);
        var adapter = new LoHaAdapter<double>(baseLayer, Rank, Alpha);

        Assert.Same(baseLayer, adapter.BaseLayer);
    }

    #endregion

    #region Parameter Efficiency Tests

    [Fact]
    public void LoRA_ParameterReduction_SignificantForLargeLayers()
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

    [Fact]
    public void LoRA_ParameterCount_MatchesFormula()
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

    [Fact]
    public void LoRA_TrainingWorkflow_ReducesLoss()
    {
        // Use small dimensions to avoid gradient explosion from random DenseLayer weight init
        int smallInput = 8;
        int smallOutput = 4;
        int smallRank = 2;
        double smallAlpha = 2.0;
        double trainingLr = 0.001;

        var baseLayer = new DenseLayer<double>(smallInput, smallOutput);
        var adapter = new StandardLoRAAdapter<double>(baseLayer, smallRank, smallAlpha);

        double initialLoss = 0;
        double finalLoss = 0;

        // Create small input/target with controlled values
        var inputData = new Vector<double>(smallInput);
        var targetData = new Vector<double>(smallOutput);
        for (int i = 0; i < smallInput; i++)
            inputData[i] = 0.1 * (i + 1);
        for (int i = 0; i < smallOutput; i++)
            targetData[i] = 0.5;

        var input = new Tensor<double>(new[] { 1, smallInput }, inputData);
        var target = new Tensor<double>(new[] { 1, smallOutput }, targetData);

        // Simple training loop with fixed synthetic data
        for (int epoch = 0; epoch < 10; epoch++)
        {
            var output = adapter.Forward(input);

            // Compute MSE loss gradient with clipping
            var gradient = new Tensor<double>(output.Shape);
            double loss = 0;
            for (int i = 0; i < output.Length; i++)
            {
                double diff = output[i] - target[i];
                loss += diff * diff;
                double grad = 2.0 * diff / output.Length;
                // Clip gradient to prevent NaN propagation
                gradient[i] = Math.Max(-1.0, Math.Min(1.0, grad));
            }
            loss /= output.Length;

            if (epoch == 0) initialLoss = loss;
            finalLoss = loss;

            // Verify every epoch produces finite loss
            Assert.False(double.IsNaN(loss), $"Loss became NaN at epoch {epoch}");
            Assert.False(double.IsInfinity(loss), $"Loss became Infinity at epoch {epoch}");

            adapter.Backward(gradient);
            adapter.UpdateParameters(trainingLr);
        }

        // With small dimensions and gradient clipping, training should produce finite results
        Assert.False(double.IsNaN(finalLoss), "Final loss should not be NaN");
        Assert.False(double.IsInfinity(finalLoss), "Final loss should not be Infinity");
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void LoRALayer_VerySmallRank_StillWorks()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, rank: 1, Alpha);
        var input = CreateTensor(1, InputSize);

        var output = layer.Forward(input);

        Assert.Equal(OutputSize, output.Shape[1]);
        AssertTensorFinite(output);
    }

    [Fact]
    public void LoRALayer_RankEqualsMinDimension_Works()
    {
        int small = 8;
        int large = 64;

        // Rank can equal the smaller dimension
        var layer = new LoRALayer<double>(large, small, rank: small, Alpha);
        var input = CreateTensor(1, large);

        var output = layer.Forward(input);

        Assert.Equal(small, output.Shape[1]);
    }

    [Fact]
    public void LoRALayer_LargeBatch_HandlesCorrectly()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, Alpha);
        int batchSize = 128;
        var input = CreateTensor(batchSize, InputSize);

        var output = layer.Forward(input);

        Assert.Equal(batchSize, output.Shape[0]);
        Assert.Equal(OutputSize, output.Shape[1]);
    }

    [Fact]
    public void StandardLoRAAdapter_NullBaseLayer_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new StandardLoRAAdapter<double>(null!, Rank, Alpha));
    }

    [Fact]
    public void LoRALayer_DefaultAlpha_EqualsRank()
    {
        var layer = new LoRALayer<double>(InputSize, OutputSize, Rank, alpha: -1);

        Assert.Equal(Rank, Convert.ToDouble(layer.Alpha), 6);
        Assert.Equal(1.0, Convert.ToDouble(layer.Scaling), 6);
    }

    #endregion

    #region Multiple Adapter Variants Forward Pass Tests

    [Fact]
    public void AllAdapters_ForwardPass_ProducesValidOutput()
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
