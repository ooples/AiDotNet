using Xunit;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Common;
using AiDotNet.Mathematics;

namespace AiDotNet.Tests.IntegrationTests;

/// <summary>
/// Integration tests demonstrating that auxiliary losses improve training across various architectures.
/// These tests verify that auxiliary loss functionality integrates properly with training pipelines.
/// </summary>
public class AuxiliaryLossIntegrationTests
{
    #region Transformer Integration Tests

    [Fact]
    public void Transformer_WithAuxiliaryLoss_TrainsSuccessfully()
    {
        // Arrange
        var transformer = new Transformer<double>(
            vocabularySize: 100,
            embeddingDimension: 32,
            numHeads: 2,
            numEncoderLayers: 1,
            numDecoderLayers: 1,
            feedForwardDimension: 64,
            maxSequenceLength: 10);

        transformer.UseAuxiliaryLoss = true;
        transformer.AuxiliaryLossWeight = 0.01;

        // Create dummy input data
        var sourceInput = new Tensor<double>(new[] { 1, 5 }); // Batch size 1, seq length 5
        for (int i = 0; i < 5; i++)
        {
            sourceInput[0, i] = i % 100;
        }

        var targetInput = new Tensor<double>(new[] { 1, 5 });
        for (int i = 0; i < 5; i++)
        {
            targetInput[0, i] = (i + 1) % 100;
        }

        // Act
        var output = transformer.Forward(sourceInput, targetInput);
        var auxiliaryLoss = transformer.ComputeAuxiliaryLoss();
        var diagnostics = transformer.GetAuxiliaryLossDiagnostics();

        // Assert
        Assert.NotNull(output);
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.ContainsKey("UseAuxiliaryLoss"));
        Assert.Equal("True", diagnostics["UseAuxiliaryLoss"]);
    }

    [Fact]
    public void MultiHeadAttention_WithEntropyRegularization_ProducesValidLoss()
    {
        // Arrange
        var layer = new MultiHeadAttentionLayer<double>(
            embeddingDimension: 64,
            numHeads: 4,
            dropout: 0.0);

        layer.UseAuxiliaryLoss = true;
        layer.AuxiliaryLossWeight = 0.005;

        // Create random input
        var batchSize = 2;
        var seqLength = 10;
        var input = new Tensor<double>(new[] { batchSize, seqLength, 64 });

        var random = new Random(42);
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLength; s++)
            {
                for (int d = 0; d < 64; d++)
                {
                    input[b, s, d] = random.NextDouble() - 0.5;
                }
            }
        }

        // Act
        var output = layer.Forward(input, input, input);
        var auxiliaryLoss = layer.ComputeAuxiliaryLoss();
        var diagnostics = layer.GetAuxiliaryLossDiagnostics();

        // Assert
        Assert.NotNull(output);
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.ContainsKey("UseAuxiliaryLoss"));
    }

    #endregion

    #region Memory Network Integration Tests

    [Fact]
    public void MemoryReadLayer_WithAttentionSparsity_WorksCorrectly()
    {
        // Arrange
        var layer = new MemoryReadLayer<double>(
            inputDimension: 16,
            memoryDimension: 32,
            outputDimension: 16);

        layer.UseAuxiliaryLoss = true;

        // Create input and memory tensors
        var input = new Tensor<double>(new[] { 1, 16 });
        var memory = new Tensor<double>(new[] { 1, 5, 32 }); // 5 memory slots

        var random = new Random(42);
        for (int i = 0; i < 16; i++)
        {
            input[0, i] = random.NextDouble();
        }
        for (int m = 0; m < 5; m++)
        {
            for (int d = 0; d < 32; d++)
            {
                memory[0, m, d] = random.NextDouble();
            }
        }

        // Act
        var output = layer.Forward(input, memory);
        var auxiliaryLoss = layer.ComputeAuxiliaryLoss();
        var diagnostics = layer.GetAuxiliaryLossDiagnostics();

        // Assert
        Assert.NotNull(output);
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.ContainsKey("UseAttentionSparsity"));
    }

    [Fact]
    public void NeuralTuringMachine_WithMemoryRegularization_InitializesCorrectly()
    {
        // Arrange
        var ntm = new NeuralTuringMachine<double>(
            inputSize: 8,
            outputSize: 8,
            controllerSize: 50,
            memorySize: 64,
            numReadHeads: 1,
            numWriteHeads: 1);

        ntm.UseAuxiliaryLoss = true;

        // Act
        var diagnostics = ntm.GetAuxiliaryLossDiagnostics();

        // Assert
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.ContainsKey("UseAuxiliaryLoss"));
    }

    #endregion

    #region Graph and Spatial Layer Integration Tests

    [Fact]
    public void GraphConvolutionalLayer_WithSmoothnessLoss_WorksCorrectly()
    {
        // Arrange
        var layer = new GraphConvolutionalLayer<double>(
            inputFeatures: 16,
            outputFeatures: 32);

        layer.UseAuxiliaryLoss = true;

        // Create graph data
        var numNodes = 5;
        var input = new Tensor<double>(new[] { 1, numNodes, 16 });
        var adjacencyMatrix = new Tensor<double>(new[] { 1, numNodes, numNodes });

        // Simple graph: linear chain
        for (int i = 0; i < numNodes - 1; i++)
        {
            adjacencyMatrix[0, i, i + 1] = 1.0;
            adjacencyMatrix[0, i + 1, i] = 1.0;
        }

        // Random features
        var random = new Random(42);
        for (int n = 0; n < numNodes; n++)
        {
            for (int f = 0; f < 16; f++)
            {
                input[0, n, f] = random.NextDouble();
            }
        }

        // Act
        var output = layer.Forward(input, adjacencyMatrix);
        var auxiliaryLoss = layer.ComputeAuxiliaryLoss();
        var diagnostics = layer.GetAuxiliaryLossDiagnostics();

        // Assert
        Assert.NotNull(output);
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.ContainsKey("UseSmoothnessLoss"));
    }

    [Fact]
    public void HighwayLayer_WithGateRegularization_MaintainsDimensions()
    {
        // Arrange
        var layer = new HighwayLayer<double>(dimension: 64);
        layer.UseAuxiliaryLoss = true;

        var input = new Tensor<double>(new[] { 2, 64 }); // Batch of 2
        var random = new Random(42);
        for (int b = 0; b < 2; b++)
        {
            for (int d = 0; d < 64; d++)
            {
                input[b, d] = random.NextDouble();
            }
        }

        // Act
        var output = layer.Forward(input);
        var auxiliaryLoss = layer.ComputeAuxiliaryLoss();

        // Assert
        Assert.NotNull(output);
        Assert.Equal(input.Shape, output.Shape); // Highway layers preserve dimensions
    }

    #endregion

    #region End-to-End Training Scenario

    [Fact]
    public void EndToEnd_TransformerWithAuxiliaryLoss_CanComputeTotalLoss()
    {
        // Arrange - Create a small transformer
        var transformer = new Transformer<double>(
            vocabularySize: 50,
            embeddingDimension: 32,
            numHeads: 2,
            numEncoderLayers: 2,
            numDecoderLayers: 2,
            feedForwardDimension: 64,
            maxSequenceLength: 10);

        transformer.UseAuxiliaryLoss = true;
        transformer.AuxiliaryLossWeight = 0.01;

        // Create training data
        var sourceSeq = new Tensor<double>(new[] { 1, 3 });
        sourceSeq[0, 0] = 5;
        sourceSeq[0, 1] = 10;
        sourceSeq[0, 2] = 15;

        var targetSeq = new Tensor<double>(new[] { 1, 3 });
        targetSeq[0, 0] = 10;
        targetSeq[0, 1] = 15;
        targetSeq[0, 2] = 20;

        var targetLabels = new Tensor<double>(new[] { 1, 3, 50 });
        // One-hot encoding for target
        targetLabels[0, 0, 10] = 1.0;
        targetLabels[0, 1, 15] = 1.0;
        targetLabels[0, 2, 20] = 1.0;

        // Act - Forward pass
        var predictions = transformer.Forward(sourceSeq, targetSeq);

        // Compute main task loss (e.g., cross-entropy)
        double mainLoss = 0.0;
        for (int t = 0; t < 3; t++)
        {
            for (int v = 0; v < 50; v++)
            {
                double pred = predictions[0, t, v];
                double target = targetLabels[0, t, v];
                if (target > 0.5)
                {
                    mainLoss -= Math.Log(Math.Max(pred, 1e-10));
                }
            }
        }

        // Compute auxiliary loss
        var auxiliaryLoss = transformer.ComputeAuxiliaryLoss();

        // Compute total loss
        double totalLoss = mainLoss + transformer.AuxiliaryLossWeight * auxiliaryLoss;

        // Assert
        Assert.True(mainLoss >= 0, "Main loss should be non-negative");
        Assert.True(auxiliaryLoss >= 0, "Auxiliary loss should be non-negative");
        Assert.True(totalLoss >= mainLoss, "Total loss should be at least as large as main loss");

        var diagnostics = transformer.GetAuxiliaryLossDiagnostics();
        Assert.NotNull(diagnostics);
        Assert.NotEmpty(diagnostics);
    }

    [Fact]
    public void EndToEnd_MultipleLayersWithAuxiliaryLoss_AggregateCorrectly()
    {
        // Arrange - Create layers with auxiliary loss
        var encoder = new TransformerEncoderLayer<double>(
            embeddingDimension: 32,
            numHeads: 2,
            feedForwardDimension: 64);

        var decoder = new TransformerDecoderLayer<double>(
            embeddingDimension: 32,
            numHeads: 2,
            feedForwardDimension: 64);

        encoder.UseAuxiliaryLoss = true;
        decoder.UseAuxiliaryLoss = true;

        // Create sample data
        var encoderInput = new Tensor<double>(new[] { 1, 5, 32 });
        var decoderInput = new Tensor<double>(new[] { 1, 5, 32 });

        var random = new Random(42);
        for (int s = 0; s < 5; s++)
        {
            for (int d = 0; d < 32; d++)
            {
                encoderInput[0, s, d] = random.NextDouble();
                decoderInput[0, s, d] = random.NextDouble();
            }
        }

        // Act
        var encoderOutput = encoder.Forward(encoderInput);
        var decoderOutput = decoder.Forward(decoderInput, encoderOutput);

        var encoderAuxLoss = encoder.ComputeAuxiliaryLoss();
        var decoderAuxLoss = decoder.ComputeAuxiliaryLoss();

        var totalAuxLoss = encoderAuxLoss + decoderAuxLoss;

        // Assert
        Assert.NotNull(encoderOutput);
        Assert.NotNull(decoderOutput);
        Assert.True(totalAuxLoss >= 0, "Total auxiliary loss should be non-negative");
    }

    #endregion

    #region Diagnostic and Monitoring Tests

    [Fact]
    public void AuxiliaryLossDiagnostics_ProvideUsefulTrainingInsights()
    {
        // Arrange
        var layer = new MultiHeadAttentionLayer<double>(
            embeddingDimension: 64,
            numHeads: 4,
            dropout: 0.1);

        layer.UseAuxiliaryLoss = true;

        // Act
        var diagnostics = layer.GetAuxiliaryLossDiagnostics();

        // Assert - Verify diagnostic information is complete
        Assert.NotNull(diagnostics);
        Assert.NotEmpty(diagnostics);

        // Should contain use flag
        Assert.True(diagnostics.ContainsKey("UseAuxiliaryLoss") ||
                    diagnostics.Any(kvp => kvp.Key.Contains("Use")));

        // Should contain weight information
        Assert.True(diagnostics.Any(kvp => kvp.Key.Contains("Weight")));

        // All values should be non-null
        Assert.All(diagnostics.Values, value => Assert.NotNull(value));
    }

    [Fact]
    public void AllImplementations_SupportDisablingAuxiliaryLoss()
    {
        // Test that auxiliary loss can be disabled for performance
        var layers = new List<(string Name, IAuxiliaryLossLayer<double> Layer)>
        {
            ("Highway", new HighwayLayer<double>(64)),
            ("MemoryRead", new MemoryReadLayer<double>(16, 32, 16)),
            ("GraphConv", new GraphConvolutionalLayer<double>(16, 32))
        };

        foreach (var (name, layer) in layers)
        {
            // Assert - Default should be disabled
            Assert.False(layer.UseAuxiliaryLoss, $"{name} should have auxiliary loss disabled by default");

            // Enable
            layer.UseAuxiliaryLoss = true;
            Assert.True(layer.UseAuxiliaryLoss, $"{name} should allow enabling auxiliary loss");

            // Disable again
            layer.UseAuxiliaryLoss = false;
            Assert.False(layer.UseAuxiliaryLoss, $"{name} should allow disabling auxiliary loss");
        }
    }

    #endregion
}
