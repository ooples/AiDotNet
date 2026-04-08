using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Unit tests for advanced algebra neural network models.
/// </summary>
public class AdvancedAlgebraNetworkTests
{
    #region OctonionNeuralNetwork Tests

    [Fact]
    public void OctonionNeuralNetwork_Construction_CreatesValidNetwork()
    {
        // Arrange - Create architecture with custom octonion layers
        var layers = new List<ILayer<double>>
        {
            new OctonionLinearLayer<double>(4, 2) // 4*8=32 input, 2*8=16 output
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 32,
            outputSize: 16,
            layers: layers
        );

        // Act
        var network = new OctonionNeuralNetwork<double>(architecture);

        // Assert
        Assert.NotNull(network);
        Assert.True(network.LayerCount > 0);
        Assert.True(network.SupportsTraining);
    }

    [Fact]
    public void OctonionNeuralNetwork_Predict_ReturnsCorrectShape()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new OctonionLinearLayer<double>(2, 1) // 16 input, 8 output
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 16,
            outputSize: 8,
            layers: layers
        );
        var network = new OctonionNeuralNetwork<double>(architecture);
        var input = new Tensor<double>([16]);
        for (int i = 0; i < 16; i++) input[i] = 0.1 * (i + 1);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.Equal(1, output.Rank);
        Assert.Equal(8, output.Shape[0]);
    }

    [Fact]
    public void OctonionNeuralNetwork_Train_CompletesWithoutError()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new OctonionLinearLayer<double>(2, 1)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 16,
            outputSize: 8,
            layers: layers
        );
        var network = new OctonionNeuralNetwork<double>(architecture);

        var input = new Tensor<double>([16]);
        var target = new Tensor<double>([8]);
        for (int i = 0; i < 16; i++) input[i] = 0.1;
        for (int i = 0; i < 8; i++) target[i] = 0.5;

        // Act & Assert - should not throw
        network.Train(input, target);
    }

    [Fact]
    public void OctonionNeuralNetwork_GetModelMetadata_ReturnsValidMetadata()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new OctonionLinearLayer<double>(2, 1)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 16,
            outputSize: 8,
            layers: layers
        );
        var network = new OctonionNeuralNetwork<double>(architecture);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal("OctonionNeuralNetwork", metadata.AdditionalInfo["NetworkType"]);
        Assert.True((int)metadata.AdditionalInfo["ParameterCount"] > 0);
    }

    [Fact]
    public void OctonionNeuralNetwork_MultiLayer_CreatesCorrectLayers()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new OctonionLinearLayer<double>(4, 8), // 32 -> 64
            new OctonionLinearLayer<double>(8, 4), // 64 -> 32
            new OctonionLinearLayer<double>(4, 1)  // 32 -> 8
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 32,
            outputSize: 8,
            layers: layers
        );

        // Act
        var network = new OctonionNeuralNetwork<double>(architecture);

        // Assert
        Assert.Equal(3, network.LayerCount);
    }

    #endregion

    #region HyperbolicNeuralNetwork Tests

    [Fact]
    public void HyperbolicNeuralNetwork_Construction_CreatesValidNetwork()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new HyperbolicLinearLayer<double>(10, 5)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 5,
            layers: layers
        );

        // Act
        var network = new HyperbolicNeuralNetwork<double>(architecture);

        // Assert
        Assert.NotNull(network);
        Assert.True(network.LayerCount > 0);
        Assert.True(network.SupportsTraining);
    }

    [Fact]
    public void HyperbolicNeuralNetwork_Construction_RejectsPositiveCurvature()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new HyperbolicLinearLayer<double>(10, 5)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 5,
            layers: layers
        );

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new HyperbolicNeuralNetwork<double>(architecture, curvature: 1.0));
    }

    [Fact]
    public void HyperbolicNeuralNetwork_Predict_ReturnsCorrectShape()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new HyperbolicLinearLayer<double>(10, 5)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 5,
            layers: layers
        );
        var network = new HyperbolicNeuralNetwork<double>(architecture);

        // Input should be inside Poincare ball (small values)
        var input = new Tensor<double>([10]);
        for (int i = 0; i < 10; i++)
            input[i] = 0.01 * (i + 1);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.Equal(1, output.Rank);
        Assert.Equal(5, output.Shape[0]);
    }

    [Fact]
    public void HyperbolicNeuralNetwork_Train_CompletesWithoutError()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new HyperbolicLinearLayer<double>(10, 5)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 5,
            layers: layers
        );
        var network = new HyperbolicNeuralNetwork<double>(architecture, curvature: -1.0);

        var input = new Tensor<double>([10]);
        var target = new Tensor<double>([5]);
        for (int i = 0; i < 10; i++) input[i] = 0.01;
        for (int i = 0; i < 5; i++) target[i] = 0.1;

        // Act & Assert - should not throw
        network.Train(input, target);
    }

    [Fact]
    public void HyperbolicNeuralNetwork_GetModelMetadata_IncludesCurvature()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new HyperbolicLinearLayer<double>(10, 5, curvature: -2.0)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 5,
            layers: layers
        );
        var network = new HyperbolicNeuralNetwork<double>(architecture, curvature: -2.0);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.Equal("HyperbolicNeuralNetwork", metadata.AdditionalInfo["NetworkType"]);
        Assert.Equal(-2.0, (double)metadata.AdditionalInfo["Curvature"]);
    }

    [Fact]
    public void HyperbolicNeuralNetwork_MultiLayer_CreatesCorrectLayers()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new HyperbolicLinearLayer<double>(10, 20),
            new HyperbolicLinearLayer<double>(20, 15),
            new HyperbolicLinearLayer<double>(15, 5)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 5,
            layers: layers
        );

        // Act
        var network = new HyperbolicNeuralNetwork<double>(architecture);

        // Assert
        Assert.Equal(3, network.LayerCount);
    }

    #endregion

    #region SparseNeuralNetwork Tests

    [Fact]
    public void SparseNeuralNetwork_Construction_CreatesValidNetwork()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new SparseLinearLayer<double>(100, 50, sparsity: 0.9)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 100,
            outputSize: 50,
            layers: layers
        );

        // Act
        var network = new SparseNeuralNetwork<double>(architecture, sparsity: 0.9);

        // Assert
        Assert.NotNull(network);
        Assert.True(network.LayerCount > 0);
        Assert.True(network.SupportsTraining);
    }

    [Fact]
    public void SparseNeuralNetwork_Construction_RejectsInvalidSparsity()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new SparseLinearLayer<double>(100, 50)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 100,
            outputSize: 50,
            layers: layers
        );

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new SparseNeuralNetwork<double>(architecture, sparsity: -0.1));
        Assert.Throws<ArgumentException>(() =>
            new SparseNeuralNetwork<double>(architecture, sparsity: 1.0));
        Assert.Throws<ArgumentException>(() =>
            new SparseNeuralNetwork<double>(architecture, sparsity: 1.5));
    }

    [Fact]
    public void SparseNeuralNetwork_Predict_ReturnsCorrectShape()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new SparseLinearLayer<double>(20, 10, sparsity: 0.8)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 20,
            outputSize: 10,
            layers: layers
        );
        var network = new SparseNeuralNetwork<double>(architecture, sparsity: 0.8);

        var input = new Tensor<double>([20]);
        for (int i = 0; i < 20; i++)
            input[i] = 0.1 * (i + 1);

        // Act
        var output = network.Predict(input);

        // Assert
        Assert.Equal(1, output.Rank);
        Assert.Equal(10, output.Shape[0]);
    }

    [Fact]
    public void SparseNeuralNetwork_ParameterCount_ReflectsSparsity()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new SparseLinearLayer<double>(100, 50, sparsity: 0.9)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 100,
            outputSize: 50,
            layers: layers
        );
        // Dense would have 100*50 + 50 = 5050 parameters
        var sparseNetwork = new SparseNeuralNetwork<double>(architecture, sparsity: 0.9);

        // Act
        int paramCount = sparseNetwork.GetParameterCount();

        // Assert - should be much less than dense
        Assert.True(paramCount < 1000,
            $"Sparse network should have fewer parameters than dense. Got: {paramCount}");
        Assert.True(paramCount > 50,
            $"Sparse network should have at least bias parameters. Got: {paramCount}");
    }

    [Fact]
    public void SparseNeuralNetwork_Train_CompletesWithoutError()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new SparseLinearLayer<double>(20, 10, sparsity: 0.5)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 20,
            outputSize: 10,
            layers: layers
        );
        var network = new SparseNeuralNetwork<double>(architecture, sparsity: 0.5);

        var input = new Tensor<double>([20]);
        var target = new Tensor<double>([10]);
        for (int i = 0; i < 20; i++) input[i] = 0.1;
        for (int i = 0; i < 10; i++) target[i] = 0.5;

        // Act & Assert - should not throw
        network.Train(input, target);
    }

    [Fact]
    public void SparseNeuralNetwork_GetModelMetadata_IncludesSparsity()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new SparseLinearLayer<double>(100, 50, sparsity: 0.95)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 100,
            outputSize: 50,
            layers: layers
        );
        var network = new SparseNeuralNetwork<double>(architecture, sparsity: 0.95);

        // Act
        var metadata = network.GetModelMetadata();

        // Assert
        Assert.Equal("SparseNeuralNetwork", metadata.AdditionalInfo["NetworkType"]);
        Assert.Equal(0.95, (double)metadata.AdditionalInfo["Sparsity"]);
    }

    [Fact]
    public void SparseNeuralNetwork_MultiLayer_CreatesCorrectLayers()
    {
        // Arrange
        var layers = new List<ILayer<double>>
        {
            new SparseLinearLayer<double>(100, 50, sparsity: 0.8),
            new SparseLinearLayer<double>(50, 25, sparsity: 0.8),
            new SparseLinearLayer<double>(25, 10, sparsity: 0.8)
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 100,
            outputSize: 10,
            layers: layers
        );

        // Act
        var network = new SparseNeuralNetwork<double>(architecture, sparsity: 0.8);

        // Assert
        Assert.Equal(3, network.LayerCount);
    }

    #endregion
}
