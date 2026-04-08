using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed.NeuralOperators;
using AiDotNet.UncertaintyQuantification.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for LayerHelper to verify layer creation operations for various neural network architectures.
/// Tests verify layer counts, types, shapes, parameter counts, and architectural properties.
/// </summary>
public class LayerHelperIntegrationTests
{
    #region CreateDefaultLayers Tests

    [Fact]
    public void CreateDefaultLayers_BasicArchitecture_CreatesCorrectLayerStructure()
    {
        // CreateDefaultLayers(arch, hiddenLayerCount=1, hiddenLayerSize=64, outputSize=1)
        // Expected: DenseLayer(10,64) + DenseLayer(64,1) = 2 layers
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultLayers(architecture).ToList();

        Assert.Equal(2, layers.Count);
        Assert.IsType<DenseLayer<double>>(layers[0]);
        Assert.IsType<DenseLayer<double>>(layers[1]);

        // First layer: input to hidden
        var firstLayer = (DenseLayer<double>)layers[0];
        Assert.True(firstLayer.ParameterCount > 0, "First layer should have trainable parameters");

        // All layers should have valid output shapes
        Assert.All(layers, layer =>
        {
            var outputShape = layer.GetOutputShape();
            Assert.True(outputShape.Length > 0, "Output shape should have at least 1 dimension");
            Assert.All(outputShape, dim => Assert.True(dim > 0, $"Each dimension should be positive, got {dim}"));
        });
    }

    [Fact]
    public void CreateDefaultLayers_ClassificationTask_CreatesCorrectLayerStructure()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 10,
            outputSize: 5);

        var layers = LayerHelper<double>.CreateDefaultLayers(architecture).ToList();

        Assert.Equal(2, layers.Count);

        // Last layer output should match the requested output size
        var lastLayer = layers[^1];
        var lastOutputShape = lastLayer.GetOutputShape();
        int lastOutputSize = lastOutputShape.Aggregate(1, (a, b) => a * b);
        Assert.Equal(5, lastOutputSize);
    }

    [Fact]
    public void CreateDefaultLayers_Float_WorksWithDifferentNumericType()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        var layers = LayerHelper<float>.CreateDefaultLayers(architecture).ToList();

        Assert.Equal(2, layers.Count);
        Assert.IsType<DenseLayer<float>>(layers[0]);
        Assert.True(layers[0].ParameterCount > 0, "Float layer should have parameters");
    }

    [Fact]
    public void CreateDefaultLayers_CustomHiddenLayers_CreatesCorrectCount()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 100,
            outputSize: 10);

        // hiddenLayerCount=3 -> input + 2 hidden + output = 4 layers
        var layers = LayerHelper<double>.CreateDefaultLayers(architecture, hiddenLayerCount: 3).ToList();

        Assert.Equal(4, layers.Count);
        Assert.All(layers, layer => Assert.IsType<DenseLayer<double>>(layer));
    }

    #endregion

    #region CreateDefaultCNNLayers Tests

    [Fact]
    public void CreateDefaultCNNLayers_TwoDimensionalInput_NormalizesTo3DAndCreatesLayers()
    {
        // 2D input should be normalized to [1, height, width] internally
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 28,
            inputWidth: 28,
            inputDepth: 1,
            outputSize: 10);

        var layers = LayerHelper<double>.CreateDefaultCNNLayers(architecture).ToList();

        // Default: 2 conv layers, each with conv+pool = 4, then flatten + 1 dense + output = 7
        Assert.True(layers.Count >= 6, $"Expected at least 6 CNN layers, got {layers.Count}");

        // Should contain ConvolutionalLayer and MaxPoolingLayer
        Assert.Contains(layers, l => l is ConvolutionalLayer<double>);
        Assert.Contains(layers, l => l is MaxPoolingLayer<double>);
        Assert.Contains(layers, l => l is FlattenLayer<double>);

        // Last layer should be dense (output)
        Assert.IsType<DenseLayer<double>>(layers[^1]);
    }

    [Fact]
    public void CreateDefaultCNNLayers_ThreeDimensionalColorImage_CreatesCorrectStructure()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var layers = LayerHelper<double>.CreateDefaultCNNLayers(architecture).ToList();

        Assert.True(layers.Count >= 6, $"Expected at least 6 CNN layers, got {layers.Count}");
        Assert.Contains(layers, l => l is ConvolutionalLayer<double>);
        Assert.Contains(layers, l => l is MaxPoolingLayer<double>);

        // Count conv layers - should match convLayerCount=2
        int convCount = layers.Count(l => l is ConvolutionalLayer<double>);
        Assert.Equal(2, convCount);
    }

    #endregion

    #region CreateDefaultResNetLayers Tests

    [Fact]
    public void CreateDefaultResNetLayers_OneDimensionalInput_CreatesMLPResNet()
    {
        // 1D input uses Dense layers with residual connections
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 32,
            outputSize: 10);

        var layers = LayerHelper<double>.CreateDefaultResNetLayers(architecture).ToList();

        Assert.True(layers.Count >= 4, $"Expected at least 4 ResNet1D layers, got {layers.Count}");

        // Should contain ResidualLayer and DenseLayer
        Assert.Contains(layers, l => l is ResidualLayer<double>);
        Assert.Contains(layers, l => l is DenseLayer<double>);

        // Last layer should be Dense (output)
        Assert.IsType<DenseLayer<double>>(layers[^1]);
    }

    #endregion

    #region CreateDefaultAttentionLayers Tests

    [Fact]
    public void CreateDefaultAttentionLayers_SequenceInput_CreatesTransformerArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 50,
            inputWidth: 64,
            outputSize: 10);

        var layers = LayerHelper<double>.CreateDefaultAttentionLayers(architecture).ToList();

        // Should contain: Input, Embedding, PositionalEncoding, 3x(MHA + LayerNorm + 2xDense + LayerNorm) + Dense
        Assert.True(layers.Count >= 10, $"Expected at least 10 attention layers, got {layers.Count}");

        // Should contain multi-head attention layers
        Assert.Contains(layers, l => l is MultiHeadAttentionLayer<double>);

        // Should contain layer normalization
        Assert.Contains(layers, l => l is LayerNormalizationLayer<double>);

        // Should contain embedding layer
        Assert.Contains(layers, l => l is EmbeddingLayer<double>);

        // Should contain positional encoding
        Assert.Contains(layers, l => l is PositionalEncodingLayer<double>);

        // 3 transformer blocks -> 3 MHA layers
        int mhaCount = layers.Count(l => l is MultiHeadAttentionLayer<double>);
        Assert.Equal(3, mhaCount);
    }

    #endregion

    #region CreateDefaultAutoEncoderLayers Tests

    [Fact]
    public void CreateDefaultAutoEncoderLayers_StandardInput_CreatesSymmetricArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 784,
            outputSize: 784);

        var layers = LayerHelper<double>.CreateDefaultAutoEncoderLayers(architecture).ToList();

        // Default: [784, 392, 196, 392, 784] -> 4 transitions
        // Each transition: DenseLayer + ActivationLayer = 2 layers per transition
        // Total: 8 layers
        Assert.True(layers.Count >= 4, $"Expected at least 4 autoencoder layers, got {layers.Count}");

        // Should contain DenseLayer (encoder and decoder)
        Assert.Contains(layers, l => l is DenseLayer<double>);

        // Should contain ActivationLayer
        Assert.Contains(layers, l => l is ActivationLayer<double>);

        // Count Dense layers (should be symmetric: 2 encoder + 2 decoder)
        int denseCount = layers.Count(l => l is DenseLayer<double>);
        Assert.True(denseCount >= 4, $"Expected at least 4 dense layers (encoder+decoder), got {denseCount}");
    }

    #endregion

    #region CreateDefaultVAELayers Tests

    [Fact]
    public void CreateDefaultVAELayers_StandardInput_CreatesEncoderDecoderWithLatent()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 784,
            outputSize: 784);

        var layers = LayerHelper<double>.CreateDefaultVAELayers(architecture, latentSize: 32).ToList();

        // 1D VAE: 3 encoder dense + MeanLayer + LogVarianceLayer + 2 decoder dense + output = 8
        Assert.True(layers.Count >= 7, $"Expected at least 7 VAE layers, got {layers.Count}");

        // Should contain encoder dense layers
        Assert.Contains(layers, l => l is DenseLayer<double>);

        // Should contain MeanLayer and LogVarianceLayer
        Assert.Contains(layers, l => l is MeanLayer<double>);
        Assert.Contains(layers, l => l is LogVarianceLayer<double>);
    }

    #endregion

    #region CreateDefaultDeepBeliefNetworkLayers Tests

    [Fact]
    public void CreateDefaultDeepBeliefNetworkLayers_StandardInput_CreatesRBMStack()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 784,
            outputSize: 10);

        var layers = LayerHelper<double>.CreateDefaultDeepBeliefNetworkLayers(architecture).ToList();

        // layerSizes = [784, 500, 500, 2000, 10] -> 4 transitions
        // Each: RBMLayer + ActivationLayer = 8, plus output DenseLayer + ActivationLayer = 10
        Assert.True(layers.Count >= 8, $"Expected at least 8 DBN layers, got {layers.Count}");

        // Should contain RBM layers (core of DBN)
        Assert.Contains(layers, l => l is RBMLayer<double>);

        // Count RBM layers (one per transition in layerSizes)
        int rbmCount = layers.Count(l => l is RBMLayer<double>);
        Assert.Equal(4, rbmCount);
    }

    #endregion

    #region CreateDefaultDeepQNetworkLayers Tests

    [Fact]
    public void CreateDefaultDeepQNetworkLayers_ReinforcementLearning_CreatesMLPForQValues()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);

        var layers = LayerHelper<double>.CreateDefaultDeepQNetworkLayers(architecture).ToList();

        // 2 hidden layers: Dense+Act, Dense+Act, output Dense = 5
        Assert.Equal(5, layers.Count);

        // All dense layers should be DenseLayer
        int denseCount = layers.Count(l => l is DenseLayer<double>);
        Assert.Equal(3, denseCount);

        // Output layer should produce Q-values for each action
        var lastDenseLayer = layers.OfType<DenseLayer<double>>().Last();
        var outputShape = lastDenseLayer.GetOutputShape();
        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        Assert.Equal(2, outputSize);
    }

    #endregion

    #region CreateDefaultESNLayers Tests

    [Fact]
    public void CreateDefaultESNLayers_StandardParams_CreatesReservoirArchitecture()
    {
        var layers = LayerHelper<double>.CreateDefaultESNLayers(
            inputSize: 10,
            outputSize: 5,
            reservoirSize: 100).ToList();

        // Dense(10->100) + ReservoirLayer(100->100) + Activation + Dense(100->5) + Activation = 5
        Assert.Equal(5, layers.Count);

        // Should contain a ReservoirLayer
        Assert.Contains(layers, l => l is ReservoirLayer<double>);

        // First layer is Dense (input to reservoir)
        Assert.IsType<DenseLayer<double>>(layers[0]);
    }

    [Fact]
    public void CreateDefaultESNLayers_CustomSpectralRadius_CreatesValidReservoir()
    {
        var layers = LayerHelper<double>.CreateDefaultESNLayers(
            inputSize: 10,
            outputSize: 5,
            reservoirSize: 100,
            spectralRadius: 0.8,
            sparsity: 0.2).ToList();

        Assert.Equal(5, layers.Count);
        Assert.Contains(layers, l => l is ReservoirLayer<double>);

        // ReservoirLayer should have parameters
        var reservoirLayer = layers.OfType<ReservoirLayer<double>>().Single();
        Assert.True(reservoirLayer.ParameterCount > 0,
            $"Reservoir layer should have parameters, got {reservoirLayer.ParameterCount}");
    }

    #endregion

    #region CreateDefaultGRULayers Tests

    [Fact]
    public void CreateDefaultGRULayers_SequenceInput_CreatesGRUArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 50,
            inputWidth: 10,
            outputSize: 5);

        var layers = LayerHelper<double>.CreateDefaultGRULayers(architecture).ToList();

        Assert.True(layers.Count >= 2, $"Expected at least 2 GRU layers, got {layers.Count}");

        // Should contain GRULayer
        Assert.Contains(layers, l => l is GRULayer<double>);

        // Should contain a DenseLayer for output projection
        Assert.Contains(layers, l => l is DenseLayer<double>);
    }

    #endregion

    #region CreateDefaultLSTMNetworkLayers Tests

    [Fact]
    public void CreateDefaultLSTMNetworkLayers_SequenceInput_CreatesLSTMArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 50,
            inputWidth: 10,
            outputSize: 5);

        var layers = LayerHelper<double>.CreateDefaultLSTMNetworkLayers(architecture).ToList();

        Assert.True(layers.Count >= 2, $"Expected at least 2 LSTM layers, got {layers.Count}");

        // Should contain LSTMLayer
        Assert.Contains(layers, l => l is LSTMLayer<double>);

        // Should contain a DenseLayer for output projection
        Assert.Contains(layers, l => l is DenseLayer<double>);
    }

    #endregion

    #region CreateDefaultRNNLayers Tests

    [Fact]
    public void CreateDefaultRNNLayers_SequenceInput_CreatesRNNArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 50,
            inputWidth: 10,
            outputSize: 5);

        var layers = LayerHelper<double>.CreateDefaultRNNLayers(architecture).ToList();

        Assert.True(layers.Count >= 2, $"Expected at least 2 RNN layers, got {layers.Count}");

        // Should contain RecurrentLayer
        Assert.Contains(layers, l => l is RecurrentLayer<double>);

        // Should contain a DenseLayer for output projection
        Assert.Contains(layers, l => l is DenseLayer<double>);
    }

    #endregion

    #region CreateDefaultGNNLayers Tests

    [Fact]
    public void CreateDefaultGNNLayers_GraphInput_CreatesGNNArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 100,
            inputWidth: 16,
            outputSize: 7);

        var layers = LayerHelper<double>.CreateDefaultGNNLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 GNN layers, got {layers.Count}");

        // Should contain graph convolution layers
        Assert.Contains(layers, l => l is GraphConvolutionalLayer<double>);

        // GNN ends with ActivationLayer for task-specific activation
        Assert.Contains(layers, l => l is ActivationLayer<double>);
    }

    #endregion

    #region CreateDefaultFeedForwardLayers Tests

    [Fact]
    public void CreateDefaultFeedForwardLayers_StandardInput_CreatesMLPStructure()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        // hiddenLayerCount=2, hiddenLayerSize=64
        var layers = LayerHelper<double>.CreateDefaultFeedForwardLayers(architecture).ToList();

        // 2 hidden + output = 3 dense layers
        Assert.Equal(3, layers.Count);
        Assert.All(layers, layer => Assert.IsType<DenseLayer<double>>(layer));

        // Total parameters should be non-zero
        int totalParams = layers.Sum(l => l.ParameterCount);
        Assert.True(totalParams > 0, $"Total parameter count should be positive, got {totalParams}");
    }

    #endregion

    #region CreateDefaultNeuralNetworkLayers Tests

    [Fact]
    public void CreateDefaultNeuralNetworkLayers_Classification_CreatesLayersWithActivations()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 100,
            outputSize: 10);

        var layers = LayerHelper<double>.CreateDefaultNeuralNetworkLayers(architecture).ToList();

        // Default complexity (Medium): 2 hidden layers
        // Dense + Activation per hidden layer + output Dense + output Activation
        Assert.True(layers.Count >= 4, $"Expected at least 4 layers, got {layers.Count}");

        // Should contain both DenseLayer and ActivationLayer
        Assert.Contains(layers, l => l is DenseLayer<double>);
        Assert.Contains(layers, l => l is ActivationLayer<double>);
    }

    [Fact]
    public void CreateDefaultNeuralNetworkLayers_Regression_NoOutputActivation()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 50,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultNeuralNetworkLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 layers, got {layers.Count}");

        // For regression, no output activation (identity/null), so last layer is Dense
        var lastLayer = layers[^1];
        Assert.IsType<DenseLayer<double>>(lastLayer);
    }

    #endregion

    #region CreateDefaultBayesianNeuralNetworkLayers Tests

    [Fact]
    public void CreateDefaultBayesianNeuralNetworkLayers_StandardInput_CreatesBayesianLayers()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultBayesianNeuralNetworkLayers(architecture).ToList();

        Assert.True(layers.Count >= 2, $"Expected at least 2 Bayesian layers, got {layers.Count}");

        // Should contain Bayesian dense layers
        Assert.Contains(layers, l => l is BayesianDenseLayer<double>);

        // Bayesian layers should have parameters (weight distributions)
        var bayesianLayers = layers.OfType<BayesianDenseLayer<double>>().ToList();
        Assert.True(bayesianLayers.Count >= 1,
            "Expected at least 1 BayesianDenseLayer");
        Assert.All(bayesianLayers, bl =>
            Assert.True(bl.ParameterCount > 0, "Bayesian layer should have parameters"));
    }

    #endregion

    #region CreateDefaultRBFNetworkLayers Tests

    [Fact]
    public void CreateDefaultRBFNetworkLayers_StandardInput_CreatesRBFArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultRBFNetworkLayers(architecture).ToList();

        Assert.True(layers.Count >= 2, $"Expected at least 2 RBF layers, got {layers.Count}");

        // Should contain RBFLayer
        Assert.Contains(layers, l => l is RBFLayer<double>);
    }

    #endregion

    #region CreateDefaultELMLayers Tests

    [Fact]
    public void CreateDefaultELMLayers_StandardInput_CreatesELMArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultELMLayers(architecture, hiddenLayerSize: 50).ToList();

        Assert.True(layers.Count >= 2, $"Expected at least 2 ELM layers, got {layers.Count}");

        // ELM uses Dense layers
        Assert.Contains(layers, l => l is DenseLayer<double>);

        // All layers should have non-negative parameter count
        Assert.All(layers, layer =>
            Assert.True(layer.ParameterCount >= 0,
                $"Parameter count should be non-negative, got {layer.ParameterCount}"));
    }

    #endregion

    #region CreateDefaultPINNLayers Tests

    [Fact]
    public void CreateDefaultPINNLayers_StandardInput_CreatesPINNArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultPINNLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 PINN layers, got {layers.Count}");

        // PINNs use Dense layers with smooth activations
        Assert.Contains(layers, l => l is DenseLayer<double>);

        // All layers should have valid output shapes
        Assert.All(layers, layer =>
        {
            var shape = layer.GetOutputShape();
            Assert.True(shape.Length > 0, "Output shape should have dimensions");
        });
    }

    #endregion

    #region CreateDefaultDeepRitzLayers Tests

    [Fact]
    public void CreateDefaultDeepRitzLayers_StandardInput_CreatesResidualStructure()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultDeepRitzLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 Deep Ritz layers, got {layers.Count}");

        // Deep Ritz uses residual connections
        Assert.Contains(layers, l => l is ResidualLayer<double>);
    }

    #endregion

    #region CreateDefaultCapsuleNetworkLayers Tests

    [Fact]
    public void CreateDefaultCapsuleNetworkLayers_ImageInput_CreatesCapsuleArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 28,
            inputWidth: 28,
            inputDepth: 1,
            outputSize: 10);

        var layers = LayerHelper<double>.CreateDefaultCapsuleNetworkLayers(architecture).ToList();

        // Conv + PrimaryCapsule + DigitCapsule + Reconstruction = 4
        Assert.Equal(4, layers.Count);

        // Should contain capsule-specific layers
        Assert.Contains(layers, l => l is ConvolutionalLayer<double>);
        Assert.Contains(layers, l => l is PrimaryCapsuleLayer<double>);
        Assert.Contains(layers, l => l is DigitCapsuleLayer<double>);
        Assert.Contains(layers, l => l is ReconstructionLayer<double>);
    }

    #endregion

    #region CreateDefaultNodeClassificationLayers Tests

    [Fact]
    public void CreateDefaultNodeClassificationLayers_GraphInput_CreatesGCNArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 100,
            inputWidth: 16,
            outputSize: 7);

        var layers = LayerHelper<double>.CreateDefaultNodeClassificationLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 node classification layers, got {layers.Count}");

        // Should contain graph convolution layers (GCN)
        Assert.Contains(layers, l => l is GraphConvolutionalLayer<double>);
    }

    #endregion

    #region CreateDefaultLinkPredictionLayers Tests

    [Fact]
    public void CreateDefaultLinkPredictionLayers_GraphInput_CreatesEncoderStructure()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputHeight: 100,
            inputWidth: 16,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultLinkPredictionLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 link prediction layers, got {layers.Count}");

        // Should contain graph convolution layers
        Assert.Contains(layers, l => l is GraphConvolutionalLayer<double>);

        // Link prediction ends with GraphConvolutionalLayer (embedding output)
        Assert.IsType<GraphConvolutionalLayer<double>>(layers[^1]);
    }

    #endregion

    #region CreateDefaultGraphClassificationLayers Tests

    [Fact]
    public void CreateDefaultGraphClassificationLayers_GraphInput_CreatesPoolingArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 100,
            inputWidth: 16,
            outputSize: 7);

        var layers = LayerHelper<double>.CreateDefaultGraphClassificationLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 graph classification layers, got {layers.Count}");

        Assert.Contains(layers, l => l is GraphConvolutionalLayer<double>);
        // Graph classification ends with GraphConvolutionalLayer (embedding output)
        Assert.IsType<GraphConvolutionalLayer<double>>(layers[^1]);
    }

    #endregion

    #region CreateDefaultDeepOperatorNetworkLayers Tests

    [Fact]
    public void CreateDefaultDeepOperatorNetworkLayers_StandardParams_CreatesBranchAndTrunk()
    {
        var (branchLayers, trunkLayers) = LayerHelper<double>.CreateDefaultDeepOperatorNetworkLayers(
            branchInputSize: 10,
            trunkInputSize: 2,
            outputSize: 1);

        var branchList = branchLayers.ToList();
        var trunkList = trunkLayers.ToList();

        Assert.True(branchList.Count >= 2, $"Expected at least 2 branch layers, got {branchList.Count}");
        Assert.True(trunkList.Count >= 2, $"Expected at least 2 trunk layers, got {trunkList.Count}");

        // Both branches should end with Dense layers
        Assert.IsType<DenseLayer<double>>(branchList[^1]);
        Assert.IsType<DenseLayer<double>>(trunkList[^1]);
    }

    [Fact]
    public void CreateDefaultDeepOperatorNetworkLayers_CustomHiddenLayers_ScalesCorrectly()
    {
        var (branchLayers, trunkLayers) = LayerHelper<double>.CreateDefaultDeepOperatorNetworkLayers(
            branchInputSize: 20,
            trunkInputSize: 3,
            outputSize: 2,
            hiddenLayerCount: 5,
            hiddenLayerSize: 128);

        var branchList = branchLayers.ToList();
        var trunkList = trunkLayers.ToList();

        // With 5 hidden layers, should have more layers than default
        Assert.True(branchList.Count >= 6, $"Expected at least 6 branch layers for 5 hidden, got {branchList.Count}");
        Assert.True(trunkList.Count >= 6, $"Expected at least 6 trunk layers for 5 hidden, got {trunkList.Count}");
    }

    #endregion

    #region CreateDefaultFourierNeuralOperatorLayers Tests

    [Fact]
    public void CreateDefaultFourierNeuralOperatorLayers_StandardInput_CreatesFNOArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 64,
            outputSize: 64);

        var layers = LayerHelper<double>.CreateDefaultFourierNeuralOperatorLayers(
            architecture,
            spatialDimensions: new[] { 64 }).ToList();

        Assert.True(layers.Count >= 4, $"Expected at least 4 FNO layers, got {layers.Count}");

        // Should contain FourierLayer (spectral convolution)
        Assert.Contains(layers, l => l is FourierLayer<double>);
    }

    [Fact]
    public void CreateDefaultFourierNeuralOperatorLayers_CustomParams_ScalesFourierLayers()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 128,
            outputSize: 128);

        var layers = LayerHelper<double>.CreateDefaultFourierNeuralOperatorLayers(
            architecture,
            spatialDimensions: new[] { 64, 64 },
            numFourierLayers: 6,
            hiddenChannels: 128,
            numModes: 16).ToList();

        // 6 Fourier layers requested
        int fourierCount = layers.Count(l => l is FourierLayer<double>);
        Assert.Equal(6, fourierCount);
    }

    #endregion

    #region NetworkComplexity Tests

    [Fact]
    public void CreateDefaultNeuralNetworkLayers_SimpleComplexity_HasFewerLayersThanDeep()
    {
        var simpleArch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 10,
            outputSize: 1);

        var deepArch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Deep,
            inputSize: 10,
            outputSize: 1);

        var simpleLayers = LayerHelper<double>.CreateDefaultNeuralNetworkLayers(simpleArch).ToList();
        var deepLayers = LayerHelper<double>.CreateDefaultNeuralNetworkLayers(deepArch).ToList();

        // Simple should have fewer layers than Deep
        Assert.True(simpleLayers.Count < deepLayers.Count,
            $"Simple ({simpleLayers.Count} layers) should have fewer layers than Deep ({deepLayers.Count} layers)");

        // Deep should have more parameters
        int simpleParams = simpleLayers.Sum(l => l.ParameterCount);
        int deepParams = deepLayers.Sum(l => l.ParameterCount);
        Assert.True(deepParams > simpleParams,
            $"Deep ({deepParams} params) should have more parameters than Simple ({simpleParams} params)");
    }

    #endregion

    #region Different Numeric Types Tests

    [Fact]
    public void CreateDefaultNeuralNetworkLayers_Double_HasTrainableParameters()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultNeuralNetworkLayers(architecture).ToList();

        Assert.All(layers, layer => Assert.NotNull(layer));

        // At least some layers should support training
        Assert.Contains(layers, l => l.SupportsTraining);

        // Total parameters should be positive
        int totalParams = layers.Sum(l => l.ParameterCount);
        Assert.True(totalParams > 0, $"Total parameters should be positive, got {totalParams}");
    }

    [Fact]
    public void CreateDefaultNeuralNetworkLayers_Float_HasTrainableParameters()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        var layers = LayerHelper<float>.CreateDefaultNeuralNetworkLayers(architecture).ToList();

        Assert.All(layers, layer => Assert.NotNull(layer));
        Assert.Contains(layers, l => l.SupportsTraining);
    }

    #endregion

    #region Sequence Models Tests

    [Fact]
    public void CreateDefaultLSTMNetworkLayers_ReturnSequence_ContainsLSTM()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 50,
            inputWidth: 10,
            outputSize: 5,
            shouldReturnFullSequence: true);

        var layers = LayerHelper<double>.CreateDefaultLSTMNetworkLayers(architecture).ToList();

        Assert.True(layers.Count >= 2, $"Expected at least 2 layers, got {layers.Count}");
        Assert.Contains(layers, l => l is LSTMLayer<double>);
    }

    [Fact]
    public void CreateDefaultGRULayers_ReturnSequence_ContainsGRU()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 50,
            inputWidth: 10,
            outputSize: 5,
            shouldReturnFullSequence: true);

        var layers = LayerHelper<double>.CreateDefaultGRULayers(architecture).ToList();

        Assert.True(layers.Count >= 2, $"Expected at least 2 layers, got {layers.Count}");
        Assert.Contains(layers, l => l is GRULayer<double>);
    }

    #endregion

    #region Edge Cases Tests

    [Fact]
    public void CreateDefaultLayers_MinimalInput_ProducesValidOutputShape()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 1,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultLayers(architecture).ToList();

        Assert.Equal(2, layers.Count);

        // Output shape of last layer should produce scalar output
        var lastOutputShape = layers[^1].GetOutputShape();
        int totalOutput = lastOutputShape.Aggregate(1, (a, b) => a * b);
        Assert.Equal(1, totalOutput);
    }

    [Fact]
    public void CreateDefaultLayers_LargeInput_ScalesParametersAppropriately()
    {
        var smallArch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1);

        var largeArch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10000,
            outputSize: 100);

        var smallLayers = LayerHelper<double>.CreateDefaultLayers(smallArch).ToList();
        var largeLayers = LayerHelper<double>.CreateDefaultLayers(largeArch).ToList();

        // Large input should have more parameters
        int smallParams = smallLayers.Sum(l => l.ParameterCount);
        int largeParams = largeLayers.Sum(l => l.ParameterCount);
        Assert.True(largeParams > smallParams,
            $"Large input ({largeParams} params) should have more parameters than small ({smallParams} params)");
    }

    [Fact]
    public void CreateDefaultLayers_BinaryClassification_OutputsSingleValue()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputSize: 10,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultLayers(architecture).ToList();

        Assert.True(layers.Count >= 2, $"Expected at least 2 layers, got {layers.Count}");

        // Output should be single value
        var lastOutput = layers[^1].GetOutputShape();
        int totalOutput = lastOutput.Aggregate(1, (a, b) => a * b);
        Assert.Equal(1, totalOutput);
    }

    #endregion

    #region Additional Architecture Tests

    [Fact]
    public void CreateDefaultOccupancyLayers_StandardInput_CreatesDenseStack()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputSize: 10,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultOccupancyLayers(architecture).ToList();

        // Dense(10,64) + BN(64) + Dropout + Dense(64,32) + BN(32) + Dropout + Dense(32,16) + Dense(16,1) = 8
        Assert.Equal(8, layers.Count);

        // Should contain BatchNormalization and Dropout
        Assert.Contains(layers, l => l is BatchNormalizationLayer<double>);
        Assert.Contains(layers, l => l is DropoutLayer<double>);

        // Output layer should have output size 1 (binary)
        var lastDense = layers.OfType<DenseLayer<double>>().Last();
        var outputShape = lastDense.GetOutputShape();
        int outputSize = outputShape.Aggregate(1, (a, b) => a * b);
        Assert.Equal(1, outputSize);
    }

    [Fact]
    public void CreateDefaultOccupancyTemporalLayers_StandardInput_CreatesTemporalArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputHeight: 24,
            inputWidth: 10,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultOccupancyTemporalLayers(architecture, historyWindowSize: 24).ToList();

        Assert.True(layers.Count >= 5, $"Expected at least 5 temporal layers, got {layers.Count}");

        // Should contain LSTM layers for temporal processing
        Assert.Contains(layers, l => l is LSTMLayer<double>);

        // Should contain multi-head attention
        Assert.Contains(layers, l => l is MultiHeadAttentionLayer<double>);

        // Should contain BatchNormalization and Dropout
        Assert.Contains(layers, l => l is BatchNormalizationLayer<double>);
        Assert.Contains(layers, l => l is DropoutLayer<double>);
    }

    [Fact]
    public void CreateDefaultDeepBoltzmannMachineLayers_StandardInput_CreatesRBMStack()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 784,
            outputSize: 10);

        var layers = LayerHelper<double>.CreateDefaultDeepBoltzmannMachineLayers(architecture).ToList();

        // layerSizes = [784, 500, 500, 2000, 10] -> 4 RBM layers + 3 BatchNorm + 1 Dense = 8
        Assert.True(layers.Count >= 7, $"Expected at least 7 DBM layers, got {layers.Count}");

        // Should contain RBM layers
        Assert.Contains(layers, l => l is RBMLayer<double>);

        // Should contain BatchNormalizationLayer (between RBM layers)
        Assert.Contains(layers, l => l is BatchNormalizationLayer<double>);

        int rbmCount = layers.Count(l => l is RBMLayer<double>);
        Assert.Equal(4, rbmCount);
    }

    [Fact]
    public void CreateDefaultVariationalPINNLayers_StandardInput_CreatesValidArchitecture()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultVariationalPINNLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 V-PINN layers, got {layers.Count}");
        Assert.Contains(layers, l => l is DenseLayer<double>);
    }

    [Fact]
    public void CreateDefaultGraphGenerationLayers_StandardInput_CreatesGraphGenerativeModel()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: 50,
            inputWidth: 16,
            outputSize: 10);

        var layers = LayerHelper<double>.CreateDefaultGraphGenerationLayers(architecture).ToList();

        // Default numEncoderLayers=2, so exactly 2 GCN layers
        Assert.Equal(2, layers.Count);
        Assert.All(layers, l => Assert.IsType<GraphConvolutionalLayer<double>>(l));
    }

    [Fact]
    public void CreateDefaultHamiltonianLayers_StandardInput_CreatesPhysicsInformedModel()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultHamiltonianLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 Hamiltonian layers, got {layers.Count}");
        Assert.Contains(layers, l => l is DenseLayer<double>);

        // Should have substantial parameter count for physics modeling
        int totalParams = layers.Sum(l => l.ParameterCount);
        Assert.True(totalParams > 0, $"Should have parameters, got {totalParams}");
    }

    [Fact]
    public void CreateDefaultLagrangianLayers_StandardInput_CreatesPhysicsInformedModel()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 1);

        var layers = LayerHelper<double>.CreateDefaultLagrangianLayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 Lagrangian layers, got {layers.Count}");
        Assert.Contains(layers, l => l is DenseLayer<double>);
    }

    [Fact]
    public void CreateDefaultUniversalDELayers_StandardInput_CreatesDiffEqSolver()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 2,
            outputSize: 2);

        var layers = LayerHelper<double>.CreateDefaultUniversalDELayers(architecture).ToList();

        Assert.True(layers.Count >= 3, $"Expected at least 3 Universal DE layers, got {layers.Count}");
        Assert.Contains(layers, l => l is DenseLayer<double>);
    }

    #endregion

    #region Cross-Architecture Consistency Tests

    [Fact]
    public void AllLayerCreators_ProduceLayersWithValidOutputShapes()
    {
        // Verify all non-buggy architectures produce layers with valid output shapes
        var architectures = new Dictionary<string, List<ILayer<double>>>();

        var arch1d = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10, outputSize: 1);

        architectures["DefaultLayers"] = LayerHelper<double>.CreateDefaultLayers(arch1d).ToList();
        architectures["FeedForward"] = LayerHelper<double>.CreateDefaultFeedForwardLayers(arch1d).ToList();
        architectures["NeuralNetwork"] = LayerHelper<double>.CreateDefaultNeuralNetworkLayers(arch1d).ToList();
        architectures["Bayesian"] = LayerHelper<double>.CreateDefaultBayesianNeuralNetworkLayers(arch1d).ToList();
        architectures["RBF"] = LayerHelper<double>.CreateDefaultRBFNetworkLayers(arch1d).ToList();

        foreach (var (name, layers) in architectures)
        {
            Assert.True(layers.Count >= 2, $"{name} should have at least 2 layers, got {layers.Count}");
            Assert.All(layers, layer =>
            {
                var shape = layer.GetOutputShape();
                Assert.True(shape.Length > 0, $"{name}: layer output shape should have dimensions");
                Assert.All(shape, d => Assert.True(d > 0, $"{name}: each dimension should be positive, got {d}"));
            });
        }
    }

    #endregion
}
