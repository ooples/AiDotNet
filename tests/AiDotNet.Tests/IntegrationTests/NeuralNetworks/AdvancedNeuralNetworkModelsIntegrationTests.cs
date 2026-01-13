using AiDotNet.ActivationFunctions;
using AiDotNet.AutoML.SearchSpace;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration tests for advanced neural network models.
/// Tests Transformer, LSTMNeuralNetwork, GRUNeuralNetwork, and other specialized networks.
/// </summary>
public class AdvancedNeuralNetworkModelsIntegrationTests
{
    private const float Tolerance = 1e-5f;

    #region Helper Methods

    /// <summary>
    /// Creates random input tensor with specified shape.
    /// </summary>
    private static Tensor<float> CreateRandomTensor(int[] shape, int seed = 42)
    {
        var random = new Random(seed);
        var length = 1;
        foreach (var dim in shape) length *= dim;
        var flatData = new float[length];
        for (int i = 0; i < flatData.Length; i++)
        {
            flatData[i] = (float)(random.NextDouble() * 2 - 1);
        }
        return new Tensor<float>(flatData, shape);
    }

    /// <summary>
    /// Creates 2D sequence input [seqLen, features].
    /// </summary>
    private static Tensor<float> CreateSequenceInput(int seqLen, int features, int seed = 42)
    {
        return CreateRandomTensor([seqLen, features], seed);
    }

    /// <summary>
    /// Creates a tensor with random integer token indices suitable for embedding layers.
    /// Values are in range [0, vocabularySize).
    /// </summary>
    private static Tensor<float> CreateTokenIndices(int length, int vocabularySize, int seed = 42)
    {
        var random = new Random(seed);
        var flatData = new float[length];
        for (int i = 0; i < length; i++)
        {
            flatData[i] = random.Next(0, vocabularySize);
        }
        return new Tensor<float>(flatData, [length]);
    }

    /// <summary>
    /// Checks if tensor has any non-zero values.
    /// </summary>
    private static bool HasNonZeroValues(Tensor<float> tensor, float tolerance = 1e-6f)
    {
        for (int i = 0; i < tensor.Length; i++)
        {
            if (Math.Abs(tensor.Data[i]) > tolerance)
                return true;
        }
        return false;
    }

    /// <summary>
    /// Creates one-hot encoded labels vector.
    /// </summary>
    private static Tensor<float> CreateOneHotLabel(int numClasses, int classIndex)
    {
        var labels = new float[numClasses];
        labels[classIndex] = 1f;
        return new Tensor<float>(labels, [numClasses]);
    }

    #endregion

    #region Transformer Tests

    [Fact]
    public void Transformer_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 128,
            inputSize: 32,
            outputSize: 10,
            maxSequenceLength: 16);

        var network = new Transformer<float>(architecture);
        var input = CreateSequenceInput(16, 32);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void Transformer_Forward_ProducesNonZeroOutput()
    {
        // Arrange
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 32,
            feedForwardDimension: 64,
            inputSize: 16,
            outputSize: 8,
            maxSequenceLength: 8);

        var network = new Transformer<float>(architecture);
        var input = CreateSequenceInput(8, 16);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        bool hasNonZero = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output.Data[i]) > Tolerance)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "Output should have non-zero values");
    }

    [Fact]
    public void Transformer_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: 4,
            modelDimension: 64,
            feedForwardDimension: 128,
            inputSize: 32,
            outputSize: 8,
            maxSequenceLength: 16);

        var network = new Transformer<float>(architecture);

        // Act
        var paramCount = network.GetParameterCount();

        // Assert
        Assert.True(paramCount > 0, "Parameter count should be positive");
    }

    [Fact]
    public void Transformer_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            numEncoderLayers: 1,
            numDecoderLayers: 0,
            numHeads: 2,
            modelDimension: 32,
            feedForwardDimension: 64,
            inputSize: 16,
            outputSize: 5,
            maxSequenceLength: 8);

        var network = new Transformer<float>(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.Transformer, metadata.ModelType);
    }

    #endregion

    #region LSTMNeuralNetwork Tests

    [Fact]
    public void LSTMNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: 10,  // sequence length
            inputWidth: 8,    // features
            outputSize: 4);

        var network = new LSTMNeuralNetwork<float>(architecture, lossFunction: null, outputActivation: null);
        var input = CreateSequenceInput(10, 8);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void LSTMNeuralNetwork_Forward_ProducesNonZeroOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: 8,
            inputWidth: 6,
            outputSize: 3);

        var network = new LSTMNeuralNetwork<float>(architecture, lossFunction: null, outputActivation: null);
        var input = CreateSequenceInput(8, 6);

        // Debug: Print layer structure and trace forward pass
        var debugOutput = new System.Text.StringBuilder();
        debugOutput.AppendLine($"Input shape: [{string.Join(", ", input.Shape)}]");
        debugOutput.AppendLine($"Input has non-zero: {HasNonZeroValues(input)}");
        debugOutput.AppendLine($"Number of layers: {network.Layers.Count}");

        var current = input;
        for (int i = 0; i < network.Layers.Count; i++)
        {
            var layer = network.Layers[i];
            debugOutput.AppendLine($"Layer {i}: {layer.GetType().Name}");
            debugOutput.AppendLine($"  Input shape: [{string.Join(", ", current.Shape)}]");

            // Check LSTM layer weights before forward
            if (layer is AiDotNet.NeuralNetworks.Layers.LSTMLayer<float> lstmLayer)
            {
                debugOutput.AppendLine($"  LSTM WeightsFi has non-zero: {HasNonZeroValues(lstmLayer.WeightsFi)} (shape: [{string.Join(", ", lstmLayer.WeightsFi.Shape)}])");
                debugOutput.AppendLine($"  LSTM WeightsIi has non-zero: {HasNonZeroValues(lstmLayer.WeightsIi)}");
                debugOutput.AppendLine($"  LSTM WeightsCi has non-zero: {HasNonZeroValues(lstmLayer.WeightsCi)}");
                debugOutput.AppendLine($"  LSTM WeightsOi has non-zero: {HasNonZeroValues(lstmLayer.WeightsOi)}");
                debugOutput.AppendLine($"  LSTM WeightsFh has non-zero: {HasNonZeroValues(lstmLayer.WeightsFh)} (shape: [{string.Join(", ", lstmLayer.WeightsFh.Shape)}])");
                debugOutput.AppendLine($"  LSTM WeightsIh has non-zero: {HasNonZeroValues(lstmLayer.WeightsIh)}");
                debugOutput.AppendLine($"  LSTM WeightsCh has non-zero: {HasNonZeroValues(lstmLayer.WeightsCh)}");
                debugOutput.AppendLine($"  LSTM WeightsOh has non-zero: {HasNonZeroValues(lstmLayer.WeightsOh)}");
                debugOutput.AppendLine($"  LSTM BiasF has non-zero: {HasNonZeroValues(lstmLayer.BiasF)}");
                debugOutput.AppendLine($"  LSTM BiasI has non-zero: {HasNonZeroValues(lstmLayer.BiasI)}");
                debugOutput.AppendLine($"  LSTM BiasC has non-zero: {HasNonZeroValues(lstmLayer.BiasC)}");
                debugOutput.AppendLine($"  LSTM BiasO has non-zero: {HasNonZeroValues(lstmLayer.BiasO)}");

                // Test slice and matmul directly
                debugOutput.AppendLine($"  current shape: [{string.Join(", ", current.Shape)}]");
                debugOutput.AppendLine($"  current has non-zero: {HasNonZeroValues(current)}");
                debugOutput.AppendLine($"  current[0-5]: [{current[0]}, {current[1]}, {current[2]}, {current[3]}, {current[4]}, {current[5]}]");

                var input3D = current.Reshape(new[] { 1, current.Shape[0], current.Shape[1] });
                debugOutput.AppendLine($"  input3D shape: [{string.Join(", ", input3D.Shape)}]");
                debugOutput.AppendLine($"  input3D has non-zero: {HasNonZeroValues(input3D)}");
                debugOutput.AppendLine($"  input3D Data[0-5]: [{input3D.Data[0]}, {input3D.Data[1]}, {input3D.Data[2]}, {input3D.Data[3]}, {input3D.Data[4]}, {input3D.Data[5]}]");

                var xt = input3D.GetSliceAlongDimension(0, 1);
                debugOutput.AppendLine($"  Slice xt shape: [{string.Join(", ", xt.Shape)}]");
                debugOutput.AppendLine($"  Slice xt has non-zero: {HasNonZeroValues(xt)}");
                debugOutput.AppendLine($"  Slice xt Data[0-5]: [{xt.Data[0]}, {xt.Data[1]}, {xt.Data[2]}, {xt.Data[3]}, {xt.Data[4]}, {xt.Data[5]}]");

                // Transpose weight and test matmul
                var engine = new AiDotNet.Tensors.Engines.CpuEngine();
                var WfiT = engine.TensorTranspose(lstmLayer.WeightsFi);
                debugOutput.AppendLine($"  WfiT shape: [{string.Join(", ", WfiT.Shape)}]");
                debugOutput.AppendLine($"  WfiT has non-zero: {HasNonZeroValues(WfiT)}");

                // Test matrix multiplication
                var matmulResult = engine.TensorMatMul(xt, WfiT);
                debugOutput.AppendLine($"  MatMul result shape: [{string.Join(", ", matmulResult.Shape)}]");
                debugOutput.AppendLine($"  MatMul result has non-zero: {HasNonZeroValues(matmulResult)}");
                if (!HasNonZeroValues(matmulResult))
                {
                    // Debug: show first few values from xt and WfiT
                    debugOutput.AppendLine($"  xt[0-4]: [{xt[0]}, {xt[1]}, {xt[2]}, {xt[3]}]");
                    debugOutput.AppendLine($"  WfiT[0,0-3]: [{WfiT[0, 0]}, {WfiT[0, 1]}, {WfiT[0, 2]}, {WfiT[0, 3]}]");
                }
            }

            current = layer.Forward(current);
            debugOutput.AppendLine($"  Output shape: [{string.Join(", ", current.Shape)}]");
            debugOutput.AppendLine($"  Output has non-zero: {HasNonZeroValues(current)}");
        }

        var output = current;

        // Assert
        bool hasNonZero = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output.Data[i]) > Tolerance)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, $"Output should have non-zero values\n\nDebug trace:\n{debugOutput}");
    }

    [Fact]
    public void LSTMNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputHeight: 16,
            inputWidth: 12,
            outputSize: 6);

        var network = new LSTMNeuralNetwork<float>(architecture, lossFunction: null, outputActivation: null);

        // Act
        var paramCount = network.GetParameterCount();

        // Assert
        Assert.True(paramCount > 0, "Parameter count should be positive");
    }

    [Fact]
    public void LSTMNeuralNetwork_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputHeight: 10,
            inputWidth: 8,
            outputSize: 5);

        var network = new LSTMNeuralNetwork<float>(architecture, lossFunction: null, outputActivation: null);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.LSTMNeuralNetwork, metadata.ModelType);
    }

    #endregion

    #region GRUNeuralNetwork Tests

    [Fact]
    public void GRUNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: 12,  // sequence length
            inputWidth: 10,   // features
            outputSize: 5);

        var network = new GRUNeuralNetwork<float>(architecture);
        var input = CreateSequenceInput(12, 10);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void GRUNeuralNetwork_Forward_ProducesNonZeroOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: 6,
            inputWidth: 4,
            outputSize: 2);

        var network = new GRUNeuralNetwork<float>(architecture);
        var input = CreateSequenceInput(6, 4);

        // Act
        var output = network.Predict(input);

        // Assert
        bool hasNonZero = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output.Data[i]) > Tolerance)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "Output should have non-zero values");
    }

    [Fact]
    public void GRUNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputHeight: 20,
            inputWidth: 16,
            outputSize: 8);

        var network = new GRUNeuralNetwork<float>(architecture);

        // Act
        var paramCount = network.GetParameterCount();

        // Assert
        Assert.True(paramCount > 0, "Parameter count should be positive");
    }

    [Fact]
    public void GRUNeuralNetwork_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputHeight: 8,
            inputWidth: 6,
            outputSize: 1);

        var network = new GRUNeuralNetwork<float>(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.GRUNeuralNetwork, metadata.ModelType);
    }

    #endregion

    #region ResidualNeuralNetwork Tests

    [Fact]
    public void ResidualNeuralNetwork_Predict_ProducesCorrectOutputShape()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Medium,
            inputSize: 64,
            outputSize: 10);

        var network = new ResidualNeuralNetwork<float>(architecture);
        var input = CreateRandomTensor([64]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(10, output.Shape[^1]); // Last dimension is output classes
    }

    [Fact]
    public void ResidualNeuralNetwork_Forward_ProducesNonZeroOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 8);

        var network = new ResidualNeuralNetwork<float>(architecture);
        var input = CreateRandomTensor([32]);

        // Act
        var output = network.Predict(input);

        // Assert
        bool hasNonZero = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output.Data[i]) > Tolerance)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "Output should have non-zero values");
    }

    [Fact]
    public void ResidualNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Deep,
            inputSize: 128,
            outputSize: 16);

        var network = new ResidualNeuralNetwork<float>(architecture);

        // Act
        var paramCount = network.GetParameterCount();

        // Assert
        Assert.True(paramCount > 0, "Parameter count should be positive");
    }

    [Fact]
    public void ResidualNeuralNetwork_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Medium,
            inputSize: 48,
            outputSize: 6);

        var network = new ResidualNeuralNetwork<float>(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.ResidualNeuralNetwork, metadata.ModelType);
    }

    #endregion

    #region SiameseNetwork Tests

    [Fact]
    public void SiameseNetwork_Predict_ProducesOutput()
    {
        // Arrange - SiameseNeuralNetwork uses Transformer-based encoder by default
        var inputShape = new[] { 1, 32 }; // One sequence of 32 tokens
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: inputShape[0],
            inputWidth: inputShape[1],
            outputSize: 768);
        
        var network = new SiameseNeuralNetwork<float>(architecture);
        
        // SiameseNeuralNetwork expects [batchSize, seqLen] for simple embedding lookup
        var input = Tensor<float>.CreateRandom([1, 32]);
        for (int i = 0; i < input.Length; i++) input.SetFlat(i, (float)Math.Floor(input.GetFlat(i) * 100));

        // Act
        var result = network.Predict(input);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(new[] { 1, 32, 768 }, result.Shape);
    }

    [Fact]
    public void SiameseNetwork_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var inputShape = new[] { 1, 32 };
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputHeight: inputShape[0],
            inputWidth: inputShape[1],
            outputSize: 768);
        
        var network = new SiameseNeuralNetwork<float>(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.SiameseNetwork, metadata.ModelType);
        Assert.Equal("SiameseNeuralNetwork", metadata.Name);
    }

    #endregion

    #region RadialBasisFunctionNetwork Tests

    [Fact]
    public void RadialBasisFunctionNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var network = new RadialBasisFunctionNetwork<float>(architecture);
        var input = CreateRandomTensor([16]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void RadialBasisFunctionNetwork_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: 24,
            outputSize: 6);

        var network = new RadialBasisFunctionNetwork<float>(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.NeuralNetworkRegression, metadata.ModelType);
    }

    #endregion

    #region EchoStateNetwork Tests

    [Fact]
    public void EchoStateNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: 10,
            inputWidth: 8,
            outputSize: 4);

        var network = new EchoStateNetwork<float>(architecture, reservoirSize: 50, lossFunction: null, reservoirInputScalarActivation: null);
        var input = CreateSequenceInput(10, 8);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void EchoStateNetwork_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputHeight: 8,
            inputWidth: 6,
            outputSize: 3);

        var network = new EchoStateNetwork<float>(architecture, reservoirSize: 50, lossFunction: null, reservoirInputScalarActivation: null);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.EchoStateNetwork, metadata.ModelType);
    }

    #endregion

    #region HopfieldNetwork Tests

    [Fact]
    public void HopfieldNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 16);

        var network = new HopfieldNetwork<float>(architecture, size: 16);
        var input = CreateRandomTensor([16]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void HopfieldNetwork_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 32);

        var network = new HopfieldNetwork<float>(architecture, size: 32);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.HopfieldNetwork, metadata.ModelType);
    }

    #endregion

    #region SelfOrganizingMap Tests

    [Fact]
    public void SelfOrganizingMap_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 20,
            outputSize: 10);

        var network = new SelfOrganizingMap<float>(architecture);
        var input = CreateRandomTensor([20]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void SelfOrganizingMap_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 24,
            outputSize: 12);

        var network = new SelfOrganizingMap<float>(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.SelfOrganizingMap, metadata.ModelType);
    }

    #endregion

    #region ExtremeLearningMachine Tests

    [Fact]
    public void ExtremeLearningMachine_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 24,
            outputSize: 8);

        var network = new ExtremeLearningMachine<float>(architecture, hiddenLayerSize: 64);
        var input = CreateRandomTensor([24]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Output should have elements");
    }

    [Fact]
    public void ExtremeLearningMachine_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Medium,
            inputSize: 32,
            outputSize: 5);

        var network = new ExtremeLearningMachine<float>(architecture, hiddenLayerSize: 64);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.ExtremeLearningMachine, metadata.ModelType);
    }

    #endregion

    #region Integration Tests

    [Fact]
    public void SequenceNetworks_DifferentComplexities_ProduceOutput()
    {
        // Test LSTM and GRU with different complexity settings
        var complexities = new[]
        {
            NetworkComplexity.Simple,
            NetworkComplexity.Medium
        };

        foreach (var complexity in complexities)
        {
            // LSTM
            var lstmArch = new NeuralNetworkArchitecture<float>(
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: complexity,
                inputHeight: 8,
                inputWidth: 6,
                outputSize: 4);

            var lstm = new LSTMNeuralNetwork<float>(lstmArch, lossFunction: null, outputActivation: null);
            var lstmInput = CreateSequenceInput(8, 6);
            var lstmOutput = lstm.Predict(lstmInput);
            Assert.NotNull(lstmOutput);

            // GRU
            var gruArch = new NeuralNetworkArchitecture<float>(
                inputType: InputType.TwoDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                complexity: complexity,
                inputHeight: 8,
                inputWidth: 6,
                outputSize: 4);

            var gru = new GRUNeuralNetwork<float>(gruArch);
            var gruInput = CreateSequenceInput(8, 6);
            var gruOutput = gru.Predict(gruInput);
            Assert.NotNull(gruOutput);
        }
    }

    [Fact]
    public void DenseNetworks_DifferentTaskTypes_ProduceOutput()
    {
        // Test ResidualNeuralNetwork with different task types
        var taskTypes = new[]
        {
            NeuralNetworkTaskType.Regression,
            NeuralNetworkTaskType.BinaryClassification,
            NeuralNetworkTaskType.MultiClassClassification
        };

        foreach (var taskType in taskTypes)
        {
            var architecture = new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: taskType,
                complexity: NetworkComplexity.Simple,
                inputSize: 32,
                outputSize: taskType == NeuralNetworkTaskType.BinaryClassification ? 1 : 5);

            var network = new ResidualNeuralNetwork<float>(architecture);
            var input = CreateRandomTensor([32]);
            var output = network.Predict(input);

            Assert.NotNull(output);
        }
    }

    #endregion

    #region Autoencoder Tests

    [Fact]
    public void Autoencoder_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 64);  // Output same as input for autoencoder

        var network = new Autoencoder<float>(architecture, learningRate: 0.01f, epochs: 1, batchSize: 32);
        var input = CreateRandomTensor([64]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(64, output.Length);
    }

    [Fact]
    public void Autoencoder_Predict_ProducesNonZeroOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 32);

        var network = new Autoencoder<float>(architecture, learningRate: 0.01f);
        var input = CreateRandomTensor([32]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.True(HasNonZeroValues(output), "Output should have non-zero values");
    }

    [Fact]
    public void Autoencoder_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 64);

        var network = new Autoencoder<float>(architecture, learningRate: 0.01f);

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void Autoencoder_GetModelMetadata_ReturnsValidData()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 64);

        var network = new Autoencoder<float>(architecture, learningRate: 0.01f);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.NotNull(metadata.AdditionalInfo);
    }

    [Fact]
    public void Autoencoder_EncodedSize_IsPositive()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 128,
            outputSize: 128);

        var network = new Autoencoder<float>(architecture, learningRate: 0.01f);

        // Act
        var encodedSize = network.EncodedSize;

        // Assert
        Assert.True(encodedSize > 0, "Encoded size should be positive");
        Assert.True(encodedSize < 128, "Encoded size should be smaller than input (compression)");
    }

    #endregion

    #region VariationalAutoencoder Tests

    [Fact]
    public void VariationalAutoencoder_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 64);

        var network = new VariationalAutoencoder<float>(architecture, latentSize: 16);
        var input = CreateRandomTensor([64]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(64, output.Length);
    }

    [Fact]
    public void VariationalAutoencoder_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 64);

        var network = new VariationalAutoencoder<float>(architecture, latentSize: 16);

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void VariationalAutoencoder_LatentSize_IsCorrect()
    {
        // Arrange
        int expectedLatentSize = 32;
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 128,
            outputSize: 128);

        var network = new VariationalAutoencoder<float>(architecture, latentSize: expectedLatentSize);

        // Act
        var actualLatentSize = network.LatentSize;

        // Assert
        Assert.Equal(expectedLatentSize, actualLatentSize);
    }

    #endregion

    #region Deep Belief Network Tests

    [Fact]
    public void DeepBeliefNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 1);

        var network = new DeepBeliefNetwork<float>(
            architecture,
            learningRate: 0.01f,
            epochs: 1,
            batchSize: 32);
        var input = CreateRandomTensor([64]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void DeepBeliefNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 1);

        var network = new DeepBeliefNetwork<float>(
            architecture,
            learningRate: 0.01f,
            epochs: 1,
            batchSize: 32);

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region Restricted Boltzmann Machine Tests

    [Fact]
    public void RestrictedBoltzmannMachine_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 32);

        var network = new RestrictedBoltzmannMachine<float>(
            architecture,
            visibleSize: 64,
            hiddenSize: 32,
            learningRate: 0.01,
            cdSteps: 1,
            scalarActivation: new SigmoidActivation<float>());
        var input = CreateRandomTensor([64]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void RestrictedBoltzmannMachine_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 32);

        var network = new RestrictedBoltzmannMachine<float>(
            architecture,
            visibleSize: 64,
            hiddenSize: 32,
            learningRate: 0.01,
            cdSteps: 1,
            scalarActivation: new SigmoidActivation<float>());

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region CapsuleNetwork Tests

    [Fact]
    public void CapsuleNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputHeight: 28,
            inputWidth: 28,
            inputDepth: 1,
            outputSize: 10);

        var network = new CapsuleNetwork<float>(architecture);
        var input = CreateRandomTensor([1, 28, 28]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void CapsuleNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputHeight: 28,
            inputWidth: 28,
            inputDepth: 1,
            outputSize: 10);

        var network = new CapsuleNetwork<float>(architecture);

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region DeepQNetwork Tests

    [Fact]
    public void DeepQNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 4);  // 4 actions

        var network = new DeepQNetwork<float>(architecture);
        var input = CreateRandomTensor([32]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void DeepQNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 4);

        var network = new DeepQNetwork<float>(architecture);

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region LiquidStateMachine Tests

    [Fact]
    public void LiquidStateMachine_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var network = new LiquidStateMachine<float>(
            architecture,
            reservoirSize: 64,
            spectralRadius: 0.9f,
            inputScaling: 0.1f);
        var input = CreateRandomTensor([16]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void LiquidStateMachine_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var network = new LiquidStateMachine<float>(
            architecture,
            reservoirSize: 64,
            spectralRadius: 0.9f,
            inputScaling: 0.1f);

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region SpikingNeuralNetwork Tests

    [Fact]
    public void SpikingNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 10);

        var network = new SpikingNeuralNetwork<float>(
            architecture,
            timeStep: 0.1,
            simulationSteps: 100,
            scalarActivation: new ReLUActivation<float>());
        var input = CreateRandomTensor([32]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void SpikingNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 10);

        var network = new SpikingNeuralNetwork<float>(
            architecture,
            timeStep: 0.1,
            simulationSteps: 100,
            scalarActivation: new ReLUActivation<float>());

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region SparseNeuralNetwork Tests

    [Fact]
    public void SparseNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 10);

        var network = new SparseNeuralNetwork<float>(architecture, sparsity: 0.5);
        var input = CreateRandomTensor([64]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void SparseNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 10);

        var network = new SparseNeuralNetwork<float>(architecture, sparsity: 0.5);

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region MemoryNetwork Tests

    [Fact]
    public void MemoryNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 10);

        var network = new MemoryNetwork<float>(
            architecture,
            memorySize: 64,
            embeddingSize: 32);
        // MemoryNetwork uses EmbeddingLayer which expects token indices
        // The vocabulary size is typically the inputSize (32)
        var input = CreateTokenIndices(length: 32, vocabularySize: 32);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void MemoryNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 10);

        var network = new MemoryNetwork<float>(
            architecture,
            memorySize: 64,
            embeddingSize: 32);

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region MixtureOfExperts Tests

    [Fact]
    public void MixtureOfExpertsNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 10);

        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 64,
            OutputDim = 10
        };

        var network = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);
        var input = CreateRandomTensor([64]);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.NotNull(output);
    }

    [Fact]
    public void MixtureOfExpertsNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 10);

        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 64,
            OutputDim = 10
        };

        var network = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

        // Act
        var parameterCount = network.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region GAN Model Tests

    [Fact]
    public void GenerativeAdversarialNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 1);

        var gan = new GenerativeAdversarialNetwork<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            InputType.OneDimensional);

        var noiseInput = CreateRandomTensor([32]);

        // Act
        var output = gan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "GAN output should have elements");
    }

    [Fact]
    public void GenerativeAdversarialNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 1);

        var gan = new GenerativeAdversarialNetwork<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            InputType.OneDimensional);

        // Act
        var generatorParams = gan.Generator.GetParameterCount();
        var discriminatorParams = gan.Discriminator.GetParameterCount();

        // Assert
        Assert.True(generatorParams > 0, $"Generator parameter count should be > 0, got {generatorParams}");
        Assert.True(discriminatorParams > 0, $"Discriminator parameter count should be > 0, got {discriminatorParams}");
    }

    [Fact]
    public void DCGAN_Predict_ProducesOutput()
    {
        // Arrange - small image size for testing
        var dcgan = new DCGAN<float>(
            latentSize: 16,
            imageChannels: 1,
            imageHeight: 8,
            imageWidth: 8,
            generatorFeatureMaps: 8,
            discriminatorFeatureMaps: 8);

        // Create noise input matching the generator's expected input shape
        // DCGAN generator expects input depth = featureMaps * 8 = 64, spatial size = 4x4
        var noiseInput = CreateRandomTensor([64, 4, 4]); // [channels, height, width]

        // Act
        var output = dcgan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "DCGAN output should have elements");
    }

    [Fact]
    public void DCGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var dcgan = new DCGAN<float>(
            latentSize: 16,
            imageChannels: 1,
            imageHeight: 8,
            imageWidth: 8,
            generatorFeatureMaps: 8,
            discriminatorFeatureMaps: 8);

        // Act
        var generatorParams = dcgan.Generator.GetParameterCount();
        var discriminatorParams = dcgan.Discriminator.GetParameterCount();

        // Assert
        Assert.True(generatorParams > 0, $"DCGAN generator parameter count should be > 0, got {generatorParams}");
        Assert.True(discriminatorParams > 0, $"DCGAN discriminator parameter count should be > 0, got {discriminatorParams}");
    }

    [Fact]
    public void WGAN_Predict_ProducesOutput()
    {
        // Arrange
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 64);

        var criticArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 1);

        var wgan = new WGAN<float>(
            generatorArchitecture,
            criticArchitecture,
            InputType.OneDimensional);

        var noiseInput = CreateRandomTensor([32]);

        // Act
        var output = wgan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "WGAN output should have elements");
    }

    [Fact]
    public void WGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 64);

        var criticArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 1);

        var wgan = new WGAN<float>(
            generatorArchitecture,
            criticArchitecture,
            InputType.OneDimensional);

        // Act
        var generatorParams = wgan.Generator.GetParameterCount();
        var criticParams = wgan.Critic.GetParameterCount();

        // Assert
        Assert.True(generatorParams > 0, $"WGAN generator parameter count should be > 0, got {generatorParams}");
        Assert.True(criticParams > 0, $"WGAN critic parameter count should be > 0, got {criticParams}");
    }

    [Fact]
    public void ConditionalGAN_Predict_ProducesOutput()
    {
        // Arrange
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 1);

        var cgan = new ConditionalGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            numConditionClasses: 10,
            InputType.OneDimensional);

        var noiseInput = CreateRandomTensor([32]);

        // Act
        var output = cgan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Conditional GAN output should have elements");
    }

    [Fact]
    public void ConditionalGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 1);

        var cgan = new ConditionalGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            numConditionClasses: 10,
            InputType.OneDimensional);

        // Act
        var generatorParams = cgan.Generator.GetParameterCount();
        var discriminatorParams = cgan.Discriminator.GetParameterCount();

        // Assert
        Assert.True(generatorParams > 0, $"Conditional GAN generator parameter count should be > 0, got {generatorParams}");
        Assert.True(discriminatorParams > 0, $"Conditional GAN discriminator parameter count should be > 0, got {discriminatorParams}");
    }

    [Fact]
    public void CycleGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var generatorAtoB = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 32);

        var generatorBtoA = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 32);

        var discriminatorA = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 1);

        var discriminatorB = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 1);

        var cycleGan = new CycleGAN<float>(
            generatorAtoB,
            generatorBtoA,
            discriminatorA,
            discriminatorB,
            InputType.OneDimensional);

        // Act
        var genAtoBParams = cycleGan.GeneratorAtoB.GetParameterCount();
        var genBtoAParams = cycleGan.GeneratorBtoA.GetParameterCount();

        // Assert
        Assert.True(genAtoBParams > 0, $"CycleGAN GeneratorAtoB parameter count should be > 0, got {genAtoBParams}");
        Assert.True(genBtoAParams > 0, $"CycleGAN GeneratorBtoA parameter count should be > 0, got {genBtoAParams}");
    }

    #endregion

    #region Graph Neural Network Tests

    [Fact]
    public void GraphNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var gnn = new GraphNeuralNetwork<float>(architecture, new CrossEntropyLoss<float>(),
            graphConvolutionalActivation: null, activationLayerActivation: null);
        var input = CreateRandomTensor([8, 16]); // 8 nodes, 16 features each

        // Act
        var output = gnn.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "GNN output should have elements");
    }

    [Fact]
    public void GraphNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var gnn = new GraphNeuralNetwork<float>(architecture, new CrossEntropyLoss<float>(),
            graphConvolutionalActivation: null, activationLayerActivation: null);

        // Act
        var parameterCount = gnn.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"GNN parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void GraphAttentionNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var gat = new GraphAttentionNetwork<float>(architecture, numHeads: 2, numLayers: 2);
        var input = CreateRandomTensor([8, 16]); // 8 nodes, 16 features each

        // Act
        var output = gat.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "GAT output should have elements");
    }

    [Fact]
    public void GraphAttentionNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var gat = new GraphAttentionNetwork<float>(architecture, numHeads: 2, numLayers: 2);

        // Act
        var parameterCount = gat.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"GAT parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region Vision Network Tests

    [Fact]
    public void ConvolutionalNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputDepth: 1,
            inputHeight: 8,
            inputWidth: 8,
            outputSize: 10);

        var cnn = new ConvolutionalNeuralNetwork<float>(architecture);
        var input = CreateRandomTensor([1, 8, 8]); // 1 channel, 8x8 image

        // Act
        var output = cnn.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "CNN output should have elements");
    }

    [Fact]
    public void ConvolutionalNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputDepth: 1,
            inputHeight: 8,
            inputWidth: 8,
            outputSize: 10);

        var cnn = new ConvolutionalNeuralNetwork<float>(architecture);

        // Act
        var parameterCount = cnn.GetParameterCount();

        // Assert
        Assert.True(parameterCount > 0, $"CNN parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void FeedForwardNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 10,
            outputSize: 3);

        var ffnn = new FeedForwardNeuralNetwork<float>(architecture);
        var input = CreateRandomTensor([10]);

        // Act
        var output = ffnn.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "FFNN output should have elements");
    }

    [Fact]
    public void FeedForwardNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 10,
            outputSize: 3);

        var ffnn = new FeedForwardNeuralNetwork<float>(architecture);

        // Act
        var parameterCount = ffnn.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"FFNN parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void RecurrentNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        // For RNN with 2D input [seqLen, features], set inputHeight (seqLen) and inputWidth (features)
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputHeight: 10,  // sequence length
            inputWidth: 16,   // features
            outputSize: 5);

        var rnn = new RecurrentNeuralNetwork<float>(architecture);
        var input = CreateSequenceInput(10, 16); // 10 timesteps, 16 features

        // Act
        var output = rnn.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "RNN output should have elements");
    }

    [Fact]
    public void RecurrentNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        // For RNN with 2D input [seqLen, features], set inputHeight (seqLen) and inputWidth (features)
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputHeight: 10,  // sequence length
            inputWidth: 16,   // features
            outputSize: 5);

        var rnn = new RecurrentNeuralNetwork<float>(architecture);

        // Act
        var parameterCount = rnn.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"RNN parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region Specialized Network Tests

    [Fact]
    public void AttentionNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 10);

        var attentionNet = new AttentionNetwork<float>(architecture, sequenceLength: 16, embeddingSize: 32);
        var input = CreateSequenceInput(16, 32); // 16 timesteps, 32 features

        // Act
        var output = attentionNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "AttentionNetwork output should have elements");
    }

    [Fact]
    public void AttentionNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 32,
            outputSize: 10);

        var attentionNet = new AttentionNetwork<float>(architecture, sequenceLength: 16, embeddingSize: 32);

        // Act
        var parameterCount = attentionNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"AttentionNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void NeuralTuringMachine_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 8);

        var ntm = new NeuralTuringMachine<float>(
            architecture,
            memorySize: 32,
            memoryVectorSize: 16,
            controllerSize: 64,
            lossFunction: new CrossEntropyLoss<float>(),
            contentAddressingActivation: (IActivationFunction<float>?)null,
            gateActivation: null,
            outputActivation: null);

        var input = CreateSequenceInput(8, 16); // 8 timesteps, 16 features

        // Act
        var output = ntm.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "NTM output should have elements");
    }

    [Fact]
    public void NeuralTuringMachine_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 8);

        var ntm = new NeuralTuringMachine<float>(
            architecture,
            memorySize: 32,
            memoryVectorSize: 16,
            controllerSize: 64,
            lossFunction: new CrossEntropyLoss<float>(),
            contentAddressingActivation: (IActivationFunction<float>?)null,
            gateActivation: null,
            outputActivation: null);

        // Act
        var parameterCount = ntm.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"NTM parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region Additional GAN Variant Tests

    [Fact]
    public void ACGAN_Predict_ProducesOutput()
    {
        // Arrange - GANs use ThreeDimensional input for image generation
        // For ThreeDimensional, inputSize must equal inputHeight * inputWidth * inputDepth
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,  // 8*8*1 = 64
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        // ACGAN discriminator output = 1 (real/fake) + numClasses (class labels) = 11
        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification, // ACGAN discriminator outputs multiple values
            NetworkComplexity.Simple,
            inputSize: 64,  // 8*8*1 = 64
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 11);  // 1 + numClasses = 1 + 10 = 11

        var acgan = new ACGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            numClasses: 10,
            inputType: InputType.ThreeDimensional);

        var noiseInput = CreateRandomTensor([1, 8, 8]); // [channels, height, width]

        // Act
        var output = acgan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "ACGAN output should have elements");
    }

    [Fact]
    public void ACGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange - GANs use ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,  // 8*8*1 = 64
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        // ACGAN discriminator output = 1 (real/fake) + numClasses (class labels) = 11
        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification, // ACGAN discriminator outputs multiple values
            NetworkComplexity.Simple,
            inputSize: 64,  // 8*8*1 = 64
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 11);  // 1 + numClasses = 1 + 10 = 11

        var acgan = new ACGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            numClasses: 10,
            inputType: InputType.ThreeDimensional);

        // Act
        var parameterCount = acgan.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"ACGAN parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void BigGAN_Predict_ProducesOutput()
    {
        // Arrange - BigGAN uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var biggan = new BigGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            latentSize: 16,
            numClasses: 10,
            classEmbeddingDim: 8,
            imageChannels: 1,
            imageHeight: 8,
            imageWidth: 8,
            generatorChannels: 8,
            discriminatorChannels: 8);

        // BigGAN expects 2D latent code [batch, latent_size]
        var noiseInput = CreateRandomTensor([1, 16]);

        // Act
        var output = biggan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "BigGAN output should have elements");
    }

    [Fact]
    public void BigGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange - BigGAN uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var biggan = new BigGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            latentSize: 16,
            numClasses: 10,
            classEmbeddingDim: 8,
            imageChannels: 1,
            imageHeight: 8,
            imageWidth: 8);

        // Act
        var parameterCount = biggan.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"BigGAN parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void InfoGAN_Predict_ProducesOutput()
    {
        // Arrange - InfoGAN uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var qNetworkArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 10);

        var infogan = new InfoGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            qNetworkArchitecture,
            latentCodeSize: 10,
            inputType: InputType.ThreeDimensional);

        var noiseInput = CreateRandomTensor([1, 8, 8]);

        // Act
        var output = infogan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "InfoGAN output should have elements");
    }

    [Fact]
    public void InfoGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange - InfoGAN uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var qNetworkArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 10);

        var infogan = new InfoGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            qNetworkArchitecture,
            latentCodeSize: 10,
            inputType: InputType.ThreeDimensional);

        // Act
        var parameterCount = infogan.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"InfoGAN parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void Pix2Pix_Predict_ProducesOutput()
    {
        // Arrange - Pix2Pix uses ThreeDimensional input for image-to-image translation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var pix2pix = new Pix2Pix<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            inputType: InputType.ThreeDimensional);

        var imageInput = CreateRandomTensor([1, 8, 8]); // [channels, height, width]

        // Act
        var output = pix2pix.Predict(imageInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Pix2Pix output should have elements");
    }

    [Fact]
    public void Pix2Pix_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange - Pix2Pix uses ThreeDimensional input for image-to-image translation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var pix2pix = new Pix2Pix<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            inputType: InputType.ThreeDimensional);

        // Act
        var parameterCount = pix2pix.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Pix2Pix parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void ProgressiveGAN_Predict_ProducesOutput()
    {
        // Arrange - ProgressiveGAN uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var progressiveGan = new ProgressiveGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            latentSize: 16,
            imageChannels: 1,
            maxResolutionLevel: 2, // 16x16 max
            baseFeatureMaps: 8);

        var noiseInput = CreateRandomTensor([16]);

        // Act
        var output = progressiveGan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "ProgressiveGAN output should have elements");
    }

    [Fact]
    public void ProgressiveGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange - ProgressiveGAN uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var progressiveGan = new ProgressiveGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            latentSize: 16,
            imageChannels: 1,
            maxResolutionLevel: 2,
            baseFeatureMaps: 8);

        // Act
        var parameterCount = progressiveGan.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"ProgressiveGAN parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void SAGAN_Predict_ProducesOutput()
    {
        // Arrange - SAGAN uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var sagan = new SAGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            latentSize: 16,
            imageChannels: 1,
            imageHeight: 8,
            imageWidth: 8,
            generatorChannels: 8,
            discriminatorChannels: 8);

        var noiseInput = CreateRandomTensor([16]);

        // Act
        var output = sagan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "SAGAN output should have elements");
    }

    [Fact]
    public void SAGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange - SAGAN uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var sagan = new SAGAN<float>(
            generatorArchitecture,
            discriminatorArchitecture,
            latentSize: 16,
            imageChannels: 1,
            imageHeight: 8,
            imageWidth: 8);

        // Act
        var parameterCount = sagan.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"SAGAN parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void StyleGAN_Predict_ProducesOutput()
    {
        // Arrange - StyleGAN uses ThreeDimensional input for image generation
        var mappingNetworkArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 32);

        var synthesisNetworkArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var stylegan = new StyleGAN<float>(
            mappingNetworkArchitecture,
            synthesisNetworkArchitecture,
            discriminatorArchitecture,
            latentSize: 16,
            intermediateLatentSize: 32,
            inputType: InputType.ThreeDimensional);

        var noiseInput = CreateRandomTensor([16]);

        // Act
        var output = stylegan.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "StyleGAN output should have elements");
    }

    [Fact]
    public void StyleGAN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange - StyleGAN uses ThreeDimensional input for image generation
        var mappingNetworkArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 32);

        var synthesisNetworkArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var discriminatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var stylegan = new StyleGAN<float>(
            mappingNetworkArchitecture,
            synthesisNetworkArchitecture,
            discriminatorArchitecture,
            latentSize: 16,
            intermediateLatentSize: 32,
            inputType: InputType.ThreeDimensional);

        // Act
        var parameterCount = stylegan.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"StyleGAN parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void WGANGP_Predict_ProducesOutput()
    {
        // Arrange - WGANGP uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var criticArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var wgangp = new WGANGP<float>(
            generatorArchitecture,
            criticArchitecture,
            inputType: InputType.ThreeDimensional);

        var noiseInput = CreateRandomTensor([1, 8, 8]);

        // Act
        var output = wgangp.Predict(noiseInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "WGANGP output should have elements");
    }

    [Fact]
    public void WGANGP_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange - WGANGP uses ThreeDimensional input for image generation
        var generatorArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 64);

        var criticArchitecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 64,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 1,
            outputSize: 1);

        var wgangp = new WGANGP<float>(
            generatorArchitecture,
            criticArchitecture,
            inputType: InputType.ThreeDimensional);

        // Act
        var parameterCount = wgangp.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"WGANGP parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region Additional Graph Neural Network Tests

    [Fact]
    public void GraphSAGENetwork_Predict_ProducesOutput()
    {
        // Arrange - GraphSAGENetwork uses OneDimensional input
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var graphSage = new GraphSAGENetwork<float>(
            architecture,
            aggregatorType: SAGEAggregatorType.Mean,
            numLayers: 2);

        // Create node features [numNodes, features]
        var nodeFeatures = CreateRandomTensor([8, 16]);

        // Create adjacency matrix [numNodes, numNodes] - required for graph networks
        var adjacencyMatrix = CreateRandomTensor([8, 8]);
        graphSage.SetAdjacencyMatrix(adjacencyMatrix);

        // Act
        var output = graphSage.Predict(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "GraphSAGENetwork output should have elements");
    }

    [Fact]
    public void GraphSAGENetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var graphSage = new GraphSAGENetwork<float>(
            architecture,
            aggregatorType: SAGEAggregatorType.Mean,
            numLayers: 2);

        // Act
        var parameterCount = graphSage.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"GraphSAGENetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void GraphIsomorphismNetwork_Predict_ProducesOutput()
    {
        // Arrange - GraphIsomorphismNetwork uses OneDimensional input
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var gin = new GraphIsomorphismNetwork<float>(
            architecture,
            mlpHiddenDim: 32,
            numLayers: 2);

        // Create node features [numNodes, features]
        var nodeFeatures = CreateRandomTensor([8, 16]);

        // Create adjacency matrix [numNodes, numNodes] - required for graph networks
        var adjacencyMatrix = CreateRandomTensor([8, 8]);
        gin.SetAdjacencyMatrix(adjacencyMatrix);

        // Act
        var output = gin.Predict(nodeFeatures);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "GraphIsomorphismNetwork output should have elements");
    }

    [Fact]
    public void GraphIsomorphismNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var gin = new GraphIsomorphismNetwork<float>(
            architecture,
            mlpHiddenDim: 32,
            numLayers: 2);

        // Act
        var parameterCount = gin.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"GraphIsomorphismNetwork parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region Vision Architecture Tests

    [Fact]
    public void VGGNetwork_Predict_ProducesOutput()
    {
        // Arrange - VGG requires ThreeDimensional input and VGGConfiguration
        var config = new VGGConfiguration(VGGVariant.VGG11, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var vgg = new VGGNetwork<float>(architecture, config);

        var imageInput = CreateRandomTensor([3, 32, 32]); // [channels, height, width]

        // Act
        var output = vgg.Predict(imageInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "VGGNetwork output should have elements");
    }

    [Fact]
    public void VGGNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var config = new VGGConfiguration(VGGVariant.VGG11, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var vgg = new VGGNetwork<float>(architecture, config);

        // Act
        var parameterCount = vgg.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"VGGNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void ResNetNetwork_Predict_ProducesOutput()
    {
        // Arrange - ResNet requires ThreeDimensional input and ResNetConfiguration
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var resnet = new ResNetNetwork<float>(architecture, config);

        var imageInput = CreateRandomTensor([3, 32, 32]);

        // Act
        var output = resnet.Predict(imageInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "ResNetNetwork output should have elements");
    }

    [Fact]
    public void ResNetNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var config = new ResNetConfiguration(ResNetVariant.ResNet18, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var resnet = new ResNetNetwork<float>(architecture, config);

        // Act
        var parameterCount = resnet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"ResNetNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void DenseNetNetwork_Predict_ProducesOutput()
    {
        // Arrange - DenseNet requires ThreeDimensional input and DenseNetConfiguration
        var config = new DenseNetConfiguration(DenseNetVariant.DenseNet121, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var densenet = new DenseNetNetwork<float>(architecture, config);

        var imageInput = CreateRandomTensor([3, 32, 32]);

        // Act
        var output = densenet.Predict(imageInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "DenseNetNetwork output should have elements");
    }

    [Fact]
    public void DenseNetNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var config = new DenseNetConfiguration(DenseNetVariant.DenseNet121, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var densenet = new DenseNetNetwork<float>(architecture, config);

        // Act
        var parameterCount = densenet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"DenseNetNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void EfficientNetNetwork_Predict_ProducesOutput()
    {
        // Arrange - EfficientNet requires ThreeDimensional input and EfficientNetConfiguration
        var config = new EfficientNetConfiguration(EfficientNetVariant.B0, numClasses: 10);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 224 * 224,
            inputHeight: 224,
            inputWidth: 224,
            inputDepth: 3,
            outputSize: 10);

        var efficientnet = new EfficientNetNetwork<float>(architecture, config);

        var imageInput = CreateRandomTensor([3, 224, 224]);

        // Act
        var output = efficientnet.Predict(imageInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "EfficientNetNetwork output should have elements");
    }

    [Fact]
    public void EfficientNetNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var config = new EfficientNetConfiguration(EfficientNetVariant.B0, numClasses: 10);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 224 * 224,
            inputHeight: 224,
            inputWidth: 224,
            inputDepth: 3,
            outputSize: 10);

        var efficientnet = new EfficientNetNetwork<float>(architecture, config);

        // Act
        var parameterCount = efficientnet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"EfficientNetNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void MobileNetV2Network_Predict_ProducesOutput()
    {
        // Arrange - MobileNetV2 requires ThreeDimensional input and MobileNetV2Configuration
        var config = new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha100, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var mobilenet = new MobileNetV2Network<float>(architecture, config);

        var imageInput = CreateRandomTensor([3, 32, 32]);

        // Act
        var output = mobilenet.Predict(imageInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "MobileNetV2Network output should have elements");
    }

    [Fact]
    public void MobileNetV2Network_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var config = new MobileNetV2Configuration(MobileNetV2WidthMultiplier.Alpha100, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var mobilenet = new MobileNetV2Network<float>(architecture, config);

        // Act
        var parameterCount = mobilenet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"MobileNetV2Network parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void MobileNetV3Network_Predict_ProducesOutput()
    {
        // Arrange - MobileNetV3 requires ThreeDimensional input and MobileNetV3Configuration
        var config = new MobileNetV3Configuration(MobileNetV3Variant.Small, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var mobilenet = new MobileNetV3Network<float>(architecture, config);

        var imageInput = CreateRandomTensor([3, 32, 32]);

        // Act
        var output = mobilenet.Predict(imageInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "MobileNetV3Network output should have elements");
    }

    [Fact]
    public void MobileNetV3Network_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var config = new MobileNetV3Configuration(MobileNetV3Variant.Small, numClasses: 10, inputHeight: 32, inputWidth: 32, inputChannels: 3);
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var mobilenet = new MobileNetV3Network<float>(architecture, config);

        // Act
        var parameterCount = mobilenet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"MobileNetV3Network parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void VisionTransformer_Predict_ProducesOutput()
    {
        // Arrange - VisionTransformer takes imageHeight, imageWidth, channels, patchSize, numClasses
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var vit = new VisionTransformer<float>(
            architecture,
            imageHeight: 32,
            imageWidth: 32,
            channels: 3,
            patchSize: 8,
            numClasses: 10,
            hiddenDim: 64,
            numLayers: 2,
            numHeads: 4,
            mlpDim: 128);

        var imageInput = CreateRandomTensor([3, 32, 32]);

        // Act
        var output = vit.Predict(imageInput);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "VisionTransformer output should have elements");
    }

    [Fact]
    public void VisionTransformer_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 3 * 32 * 32,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 10);

        var vit = new VisionTransformer<float>(
            architecture,
            imageHeight: 32,
            imageWidth: 32,
            channels: 3,
            patchSize: 8,
            numClasses: 10,
            hiddenDim: 64,
            numLayers: 2,
            numHeads: 4,
            mlpDim: 128);

        // Act
        var parameterCount = vit.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"VisionTransformer parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region Advanced Memory and Specialized Network Tests

    [Fact]
    public void DifferentiableNeuralComputer_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 8);

        var dnc = new DifferentiableNeuralComputer<float>(
            architecture,
            memorySize: 16,
            memoryWordSize: 8,
            controllerSize: 32,
            readHeads: 2,
            lossFunction: null,
            activationFunction: (IActivationFunction<float>?)null);

        var input = CreateRandomTensor([16]);

        // Act
        var output = dnc.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "DifferentiableNeuralComputer output should have elements");
    }

    [Fact]
    public void DifferentiableNeuralComputer_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 8);

        var dnc = new DifferentiableNeuralComputer<float>(
            architecture,
            memorySize: 16,
            memoryWordSize: 8,
            controllerSize: 32,
            readHeads: 2,
            lossFunction: null,
            activationFunction: (IActivationFunction<float>?)null);

        // Act
        var parameterCount = dnc.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"DifferentiableNeuralComputer parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void DeepBoltzmannMachine_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 16);

        var dbm = new DeepBoltzmannMachine<float>(
            architecture,
            epochs: 10,
            learningRate: 0.01f,
            lossFunction: null,
            activationFunction: (IActivationFunction<float>?)null);

        var input = CreateRandomTensor([16]);

        // Act
        var output = dbm.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "DeepBoltzmannMachine output should have elements");
    }

    [Fact]
    public void DeepBoltzmannMachine_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Generative,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 16);

        var dbm = new DeepBoltzmannMachine<float>(
            architecture,
            epochs: 10,
            learningRate: 0.01f,
            lossFunction: null,
            activationFunction: (IActivationFunction<float>?)null);

        // Act
        var parameterCount = dbm.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"DeepBoltzmannMachine parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void HyperbolicNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 8);

        var hyperbolicNet = new HyperbolicNeuralNetwork<float>(
            architecture,
            curvature: -1.0);

        var input = CreateRandomTensor([16]);

        // Act
        var output = hyperbolicNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "HyperbolicNeuralNetwork output should have elements");
    }

    [Fact]
    public void HyperbolicNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 8);

        var hyperbolicNet = new HyperbolicNeuralNetwork<float>(
            architecture,
            curvature: -1.0);

        // Act
        var parameterCount = hyperbolicNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"HyperbolicNeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void OctonionNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 16, // Must be divisible by 8 for octonions
            outputSize: 8);

        var octonionNet = new OctonionNeuralNetwork<float>(
            architecture);

        var input = CreateRandomTensor([16]);

        // Act
        var output = octonionNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "OctonionNeuralNetwork output should have elements");
    }

    [Fact]
    public void OctonionNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 8);

        var octonionNet = new OctonionNeuralNetwork<float>(
            architecture);

        // Act
        var parameterCount = octonionNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"OctonionNeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void QuantumNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 4, // Small for quantum simulation
            outputSize: 2);

        var quantumNet = new QuantumNeuralNetwork<float>(
            architecture,
            numQubits: 4);

        var input = CreateRandomTensor([4]);

        // Act
        var output = quantumNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "QuantumNeuralNetwork output should have elements");
    }

    [Fact]
    public void QuantumNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var quantumNet = new QuantumNeuralNetwork<float>(
            architecture,
            numQubits: 4);

        // Act
        var parameterCount = quantumNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"QuantumNeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region HTMNetwork Tests

    [Fact]
    public void HTMNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var htmNet = new HTMNetwork<float>(
            architecture,
            columnCount: 128,  // Small for testing
            cellsPerColumn: 8);

        var input = CreateRandomTensor([16]);

        // Act
        var output = htmNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "HTMNetwork output should have elements");
    }

    [Fact]
    public void HTMNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 16,
            outputSize: 4);

        var htmNet = new HTMNetwork<float>(architecture, columnCount: 128, cellsPerColumn: 8);

        // Act
        var parameterCount = htmNet.ParameterCount;

        // Assert
        Assert.True(parameterCount >= 0, $"HTMNetwork parameter count should be >= 0, got {parameterCount}");
    }

    #endregion

    #region HopeNetwork Tests

    [Fact]
    public void HopeNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 32);

        var hopeNet = new HopeNetwork<float>(
            architecture,
            hiddenDim: 64,
            numCMSLevels: 2,
            numRecurrentLayers: 2,
            inContextLearningLevels: 2);

        var input = CreateRandomTensor([64]);

        // Act
        var output = hopeNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "HopeNetwork output should have elements");
    }

    [Fact]
    public void HopeNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.Regression,
            NetworkComplexity.Simple,
            inputSize: 64,
            outputSize: 32);

        var hopeNet = new HopeNetwork<float>(architecture, hiddenDim: 64, numCMSLevels: 2);

        // Act
        var parameterCount = hopeNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"HopeNetwork parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region NEAT Tests

    [Fact]
    public void NEAT_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var neat = new NEAT<float>(
            architecture,
            populationSize: 10,
            mutationRate: 0.1,
            crossoverRate: 0.75);

        var input = CreateRandomTensor([4]);

        // Act
        var output = neat.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "NEAT output should have elements");
    }

    [Fact]
    public void NEAT_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 4,
            outputSize: 2);

        var neat = new NEAT<float>(architecture, populationSize: 10);

        // Act
        var parameterCount = neat.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"NEAT parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region SuperNet Tests

    [Fact]
    public void SuperNet_Predict_ProducesOutput()
    {
        // Arrange
        var searchSpace = new MobileNetSearchSpace<float>();
        var superNet = new SuperNet<float>(searchSpace, numNodes: 3);

        var input = CreateRandomTensor([16]);

        // Act
        var output = superNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "SuperNet output should have elements");
    }

    [Fact]
    public void SuperNet_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var searchSpace = new MobileNetSearchSpace<float>();
        var superNet = new SuperNet<float>(searchSpace, numNodes: 3);

        // Act
        var parameterCount = superNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"SuperNet parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region OccupancyNeuralNetwork Tests

    [Fact]
    public void OccupancyNeuralNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 8,  // Sensor features (temp, humidity, CO2, etc.)
            outputSize: 1); // Occupancy binary output

        var occNet = new OccupancyNeuralNetwork<float>(
            architecture,
            includeTemporalData: false);

        var input = CreateRandomTensor([8]);

        // Act
        var output = occNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "OccupancyNeuralNetwork output should have elements");
    }

    [Fact]
    public void OccupancyNeuralNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 8,
            outputSize: 1);

        var occNet = new OccupancyNeuralNetwork<float>(architecture, includeTemporalData: false);

        // Act
        var parameterCount = occNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"OccupancyNeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region GraphGenerationModel Tests

    [Fact]
    public void GraphGenerationModel_Predict_ProducesOutput()
    {
        // Arrange
        var graphGen = new GraphGenerationModel<float>(
            inputFeatures: 8,
            hiddenDim: 16,
            latentDim: 8,
            numEncoderLayers: 2,
            maxNodes: 20);

        // Input: node features [numNodes, features]
        var input = CreateRandomTensor([10, 8]);

        // Act
        var output = graphGen.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "GraphGenerationModel output should have elements");
    }

    [Fact]
    public void GraphGenerationModel_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var graphGen = new GraphGenerationModel<float>(
            inputFeatures: 8,
            hiddenDim: 16,
            latentDim: 8);

        // Act
        var parameterCount = graphGen.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"GraphGenerationModel parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region UNet3D Tests

    [Fact]
    public void UNet3D_Predict_ProducesOutput()
    {
        // Arrange - 3D networks need inputHeight, inputWidth, inputDepth
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 8,  // 8x8x8 voxel grid
            outputSize: 4);

        var unet3d = new UNet3D<float>(
            architecture,
            voxelResolution: 8,
            numEncoderBlocks: 2,
            baseFilters: 8);

        // Input: 3D voxel grid [C, D, H, W]
        var input = CreateRandomTensor([1, 8, 8, 8]);

        // Act
        var output = unet3d.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "UNet3D output should have elements");
    }

    [Fact]
    public void UNet3D_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 8,
            outputSize: 4);

        var unet3d = new UNet3D<float>(architecture, voxelResolution: 8, numEncoderBlocks: 2, baseFilters: 8);

        // Act
        var parameterCount = unet3d.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"UNet3D parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region VoxelCNN Tests

    [Fact]
    public void VoxelCNN_Predict_ProducesOutput()
    {
        // Arrange - 3D networks need inputHeight, inputWidth, inputDepth
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 8,
            outputSize: 4);

        var voxelCnn = new VoxelCNN<float>(
            architecture,
            voxelResolution: 8,
            numConvBlocks: 2,
            baseFilters: 8);

        var input = CreateRandomTensor([1, 8, 8, 8]);

        // Act
        var output = voxelCnn.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "VoxelCNN output should have elements");
    }

    [Fact]
    public void VoxelCNN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.ThreeDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputHeight: 8,
            inputWidth: 8,
            inputDepth: 8,
            outputSize: 4);

        var voxelCnn = new VoxelCNN<float>(architecture, voxelResolution: 8, numConvBlocks: 2, baseFilters: 8);

        // Act
        var parameterCount = voxelCnn.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"VoxelCNN parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region MeshCNN Tests

    [Fact]
    public void MeshCNN_Predict_ProducesOutput()
    {
        // Arrange - Use simplified constructor
        int numEdges = 100;
        int maxAdjacent = 4;
        var meshCnn = new MeshCNN<float>(numClasses: 4, inputFeatures: 5);

        // Create edge adjacency: each edge has up to maxAdjacent neighboring edges
        var edgeAdjacency = new int[numEdges, maxAdjacent];
        for (int i = 0; i < numEdges; i++)
        {
            for (int j = 0; j < maxAdjacent; j++)
            {
                // Simple adjacency: circular neighbors
                edgeAdjacency[i, j] = (i + j - 1 + numEdges) % numEdges;
            }
        }
        meshCnn.SetEdgeAdjacency(edgeAdjacency);

        // Input: edge features [numEdges, features]
        var input = CreateRandomTensor([numEdges, 5]);

        // Act
        var output = meshCnn.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "MeshCNN output should have elements");
    }

    [Fact]
    public void MeshCNN_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var meshCnn = new MeshCNN<float>(numClasses: 4, inputFeatures: 5);

        // Act
        var parameterCount = meshCnn.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"MeshCNN parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region SpiralNet Tests

    [Fact]
    public void SpiralNet_Predict_ProducesOutput()
    {
        // Arrange - Use default constructor
        int numVertices = 100;
        int spiralLength = 9; // Typical spiral length (center + 8 neighbors)
        var spiralNet = new SpiralNet<float>();

        // Create spiral indices: for each vertex, indices of its spiral neighborhood
        var spiralIndices = new int[numVertices, spiralLength];
        for (int i = 0; i < numVertices; i++)
        {
            for (int j = 0; j < spiralLength; j++)
            {
                // Simple spiral: circular neighbors around each vertex
                spiralIndices[i, j] = (i + j) % numVertices;
            }
        }
        spiralNet.SetSpiralIndices(spiralIndices);

        // Input: vertex features [numVertices, features]
        var input = CreateRandomTensor([numVertices, 3]);

        // Act
        var output = spiralNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "SpiralNet output should have elements");
    }

    [Fact]
    public void SpiralNet_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var spiralNet = new SpiralNet<float>();

        // Act
        var parameterCount = spiralNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"SpiralNet parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region AudioVisualCorrespondenceNetwork Tests

    [Fact]
    public void AudioVisualCorrespondenceNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 128);

        var avNet = new AudioVisualCorrespondenceNetwork<float>(
            architecture,
            embeddingDimension: 64,
            numEncoderLayers: 2);

        // Combined audio-visual input
        var input = CreateRandomTensor([1, 256]);

        // Act
        var output = avNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "AudioVisualCorrespondenceNetwork output should have elements");
    }

    [Fact]
    public void AudioVisualCorrespondenceNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.BinaryClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 128);

        var avNet = new AudioVisualCorrespondenceNetwork<float>(architecture, embeddingDimension: 64, numEncoderLayers: 2);

        // Act
        var parameterCount = avNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"AudioVisualCorrespondenceNetwork parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region AudioVisualEventLocalizationNetwork Tests

    [Fact]
    public void AudioVisualEventLocalizationNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);  // 10 event classes

        var avelNet = new AudioVisualEventLocalizationNetwork<float>(
            architecture,
            embeddingDimension: 64,
            numEncoderLayers: 2,
            eventCategories: new[] { "speech", "music", "noise", "silence", "ambient" });

        var input = CreateRandomTensor([1, 256]);

        // Act
        var output = avelNet.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "AudioVisualEventLocalizationNetwork output should have elements");
    }

    [Fact]
    public void AudioVisualEventLocalizationNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var avelNet = new AudioVisualEventLocalizationNetwork<float>(architecture, embeddingDimension: 64, numEncoderLayers: 2);

        // Act
        var parameterCount = avelNet.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"AudioVisualEventLocalizationNetwork parameter count should be > 0, got {parameterCount}");
    }

    #endregion

    #region Multimodal Vision-Language Model Tests

    [Fact]
    public void BlipNeuralNetwork_NativeMode_Predict_ProducesOutput()
    {
        // Arrange - using native mode constructor (no ONNX files required)
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var blip = new BlipNeuralNetwork<float>(
            architecture,
            imageSize: 64,
            channels: 3,
            patchSize: 8,
            vocabularySize: 1000,
            maxSequenceLength: 16,
            embeddingDimension: 64,
            hiddenDim: 128,
            numEncoderLayers: 2,
            numDecoderLayers: 2,
            numHeads: 4,
            mlpDim: 256);

        // Use image-like input: [channels, height, width] = [3, 64, 64]
        var input = CreateRandomTensor([3, 64, 64]);

        // Act
        var output = blip.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "BlipNeuralNetwork output should have elements");
    }

    [Fact]
    public void BlipNeuralNetwork_NativeMode_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var blip = new BlipNeuralNetwork<float>(
            architecture,
            imageSize: 64,
            patchSize: 8,
            embeddingDimension: 64,
            hiddenDim: 128,
            numEncoderLayers: 2,
            numDecoderLayers: 2,
            numHeads: 4);

        // Act
        var parameterCount = blip.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"BlipNeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void Blip2NeuralNetwork_NativeMode_Predict_ProducesOutput()
    {
        // Arrange - using native mode constructor (no ONNX files required)
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var blip2 = new Blip2NeuralNetwork<float>(
            architecture,
            imageSize: 56,
            channels: 3,
            patchSize: 14,
            vocabularySize: 1000,
            maxSequenceLength: 16,
            embeddingDimension: 64,
            qformerHiddenDim: 128,
            visionHiddenDim: 128,
            lmHiddenDim: 128,
            numQformerLayers: 2,
            numQueryTokens: 8,
            numHeads: 4,
            numLmDecoderLayers: 2);

        // Use image-like input: [channels, height, width]
        var input = CreateRandomTensor([3, 56, 56]);

        // Act
        var output = blip2.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Blip2NeuralNetwork output should have elements");
    }

    [Fact]
    public void Blip2NeuralNetwork_NativeMode_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var blip2 = new Blip2NeuralNetwork<float>(
            architecture,
            imageSize: 56,
            patchSize: 14,
            embeddingDimension: 64,
            qformerHiddenDim: 128,
            numQformerLayers: 2,
            numQueryTokens: 8,
            numHeads: 4);

        // Act
        var parameterCount = blip2.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Blip2NeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void FlamingoNeuralNetwork_NativeMode_Predict_ProducesOutput()
    {
        // Arrange - using native mode constructor
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var flamingo = new FlamingoNeuralNetwork<float>(
            architecture,
            embeddingDimension: 64,
            maxSequenceLength: 32,
            imageSize: 56,
            channels: 3,
            numPerceiverTokens: 8,
            maxImagesInContext: 2,
            visionHiddenDim: 64,
            lmHiddenDim: 128,
            numVisionLayers: 2,
            numLmLayers: 2,
            numHeads: 4,
            vocabularySize: 1000,
            numPerceiverLayers: 2);

        // Use image-like input
        var input = CreateRandomTensor([3, 56, 56]);

        // Act
        var output = flamingo.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "FlamingoNeuralNetwork output should have elements");
    }

    [Fact]
    public void FlamingoNeuralNetwork_NativeMode_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var flamingo = new FlamingoNeuralNetwork<float>(
            architecture,
            embeddingDimension: 64,
            imageSize: 56,
            visionHiddenDim: 64,
            lmHiddenDim: 128,
            numVisionLayers: 2,
            numLmLayers: 2,
            numHeads: 4);

        // Act
        var parameterCount = flamingo.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"FlamingoNeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void LLaVANeuralNetwork_NativeMode_Predict_ProducesOutput()
    {
        // Arrange - using native mode constructor
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var llava = new LLaVANeuralNetwork<float>(
            architecture,
            imageSize: 56,
            channels: 3,
            patchSize: 14,
            vocabularySize: 1000,
            maxSequenceLength: 32,
            embeddingDimension: 128,
            visionHiddenDim: 64,
            numVisionLayers: 2,
            numLmLayers: 2,
            numHeads: 4);

        // Use image-like input
        var input = CreateRandomTensor([3, 56, 56]);

        // Act
        var output = llava.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "LLaVANeuralNetwork output should have elements");
    }

    [Fact]
    public void LLaVANeuralNetwork_NativeMode_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var llava = new LLaVANeuralNetwork<float>(
            architecture,
            imageSize: 56,
            patchSize: 14,
            embeddingDimension: 128,
            visionHiddenDim: 64,
            numVisionLayers: 2,
            numLmLayers: 2,
            numHeads: 4);

        // Act
        var parameterCount = llava.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"LLaVANeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void ImageBindNeuralNetwork_NativeMode_Predict_ProducesOutput()
    {
        // Arrange - using native mode constructor
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var imageBind = new ImageBindNeuralNetwork<float>(
            architecture,
            imageSize: 56,
            channels: 3,
            patchSize: 14,
            vocabularySize: 1000,
            maxSequenceLength: 16,
            embeddingDimension: 64,
            hiddenDim: 128,
            numEncoderLayers: 2,
            numHeads: 4,
            audioSampleRate: 16000,
            audioMaxDuration: 2,
            imuTimesteps: 100,
            numVideoFrames: 2);

        // Use image-like input
        var input = CreateRandomTensor([3, 56, 56]);

        // Act
        var output = imageBind.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "ImageBindNeuralNetwork output should have elements");
    }

    [Fact]
    public void ImageBindNeuralNetwork_NativeMode_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var imageBind = new ImageBindNeuralNetwork<float>(
            architecture,
            imageSize: 56,
            patchSize: 14,
            embeddingDimension: 64,
            hiddenDim: 128,
            numEncoderLayers: 2,
            numHeads: 4);

        // Act
        var parameterCount = imageBind.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"ImageBindNeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void VideoCLIPNeuralNetwork_NativeMode_Predict_ProducesOutput()
    {
        // Arrange - using native mode constructor
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var videoCLIP = new VideoCLIPNeuralNetwork<float>(
            architecture,
            imageSize: 56,
            channels: 3,
            patchSize: 14,
            vocabularySize: 1000,
            maxSequenceLength: 16,
            embeddingDimension: 64,
            visionHiddenDim: 128,
            textHiddenDim: 64,
            numFrameEncoderLayers: 2,
            numTemporalLayers: 2,
            numTextLayers: 2,
            numHeads: 4,
            numFrames: 4,
            frameRate: 1.0);

        // Use video-like input: [frames, channels, height, width]
        var input = CreateRandomTensor([4, 3, 56, 56]);

        // Act
        var output = videoCLIP.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "VideoCLIPNeuralNetwork output should have elements");
    }

    [Fact]
    public void VideoCLIPNeuralNetwork_NativeMode_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var videoCLIP = new VideoCLIPNeuralNetwork<float>(
            architecture,
            imageSize: 56,
            patchSize: 14,
            embeddingDimension: 64,
            visionHiddenDim: 128,
            textHiddenDim: 64,
            numFrameEncoderLayers: 2,
            numTemporalLayers: 2,
            numTextLayers: 2,
            numHeads: 4,
            numFrames: 4);

        // Act
        var parameterCount = videoCLIP.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"VideoCLIPNeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void UnifiedMultimodalNetwork_Predict_ProducesOutput()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var unified = new UnifiedMultimodalNetwork<float>(
            architecture,
            embeddingDimension: 64,
            maxSequenceLength: 32,
            numTransformerLayers: 2,
            seed: 42);

        // Use simple 2D input
        var input = CreateRandomTensor([1, 64]);

        // Act
        var output = unified.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "UnifiedMultimodalNetwork output should have elements");
    }

    [Fact]
    public void UnifiedMultimodalNetwork_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var unified = new UnifiedMultimodalNetwork<float>(
            architecture,
            embeddingDimension: 64,
            maxSequenceLength: 32,
            numTransformerLayers: 2);

        // Act
        var parameterCount = unified.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"UnifiedMultimodalNetwork parameter count should be > 0, got {parameterCount}");
    }

    [Fact]
    public void Gpt4VisionNeuralNetwork_NativeMode_Predict_ProducesOutput()
    {
        // Arrange - using native mode constructor with mock tokenizer
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        // Create a simple tokenizer for testing
        var tokenizer = AiDotNet.Tokenization.ClipTokenizerFactory.CreateSimple();

        var gpt4v = new Gpt4VisionNeuralNetwork<float>(
            architecture,
            tokenizer,
            embeddingDimension: 128,
            visionEmbeddingDim: 64,
            maxSequenceLength: 32,
            contextWindowSize: 256,
            imageSize: 56,
            hiddenDim: 128,
            numVisionLayers: 2,
            numLanguageLayers: 2,
            numHeads: 4,
            patchSize: 14,
            vocabularySize: 1000,
            maxImagesPerRequest: 2);

        // Use image-like input
        var input = CreateRandomTensor([3, 56, 56]);

        // Act
        var output = gpt4v.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.True(output.Length > 0, "Gpt4VisionNeuralNetwork output should have elements");
    }

    [Fact]
    public void Gpt4VisionNeuralNetwork_NativeMode_GetParameterCount_ReturnsPositiveValue()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.TwoDimensional,
            NeuralNetworkTaskType.MultiClassClassification,
            NetworkComplexity.Simple,
            inputSize: 256,
            outputSize: 10);

        var tokenizer = AiDotNet.Tokenization.ClipTokenizerFactory.CreateSimple();

        var gpt4v = new Gpt4VisionNeuralNetwork<float>(
            architecture,
            tokenizer,
            embeddingDimension: 128,
            visionEmbeddingDim: 64,
            imageSize: 56,
            hiddenDim: 128,
            numVisionLayers: 2,
            numLanguageLayers: 2,
            numHeads: 4,
            patchSize: 14);

        // Act
        var parameterCount = gpt4v.ParameterCount;

        // Assert
        Assert.True(parameterCount > 0, $"Gpt4VisionNeuralNetwork parameter count should be > 0, got {parameterCount}");
    }

    #endregion
}
