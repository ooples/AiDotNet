using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Unit tests for the Expert layer, which serves as a container for sequential layers in MoE architectures.
/// </summary>
public class ExpertTests
{
    [Fact]
    public void Constructor_WithValidLayers_InitializesCorrectly()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(10, 20, new ReLUActivation<float>()),
            new DenseLayer<float>(20, 10, new ReLUActivation<float>())
        };

        // Act
        var expert = new Expert<float>(layers, new[] { 10 }, new[] { 10 });

        // Assert
        Assert.NotNull(expert);
        Assert.True(expert.SupportsTraining);
        Assert.True(expert.ParameterCount > 0);
    }

    [Fact]
    public void Constructor_WithEmptyLayerList_ThrowsArgumentException()
    {
        // Arrange
        var layers = new List<ILayer<float>>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new Expert<float>(layers, new[] { 10 }, new[] { 10 }));
    }

    [Fact]
    public void Constructor_WithNullLayerList_ThrowsArgumentException()
    {
        // Act & Assert
#pragma warning disable CS8625 // Cannot convert null literal to non-nullable reference type.
        Assert.Throws<ArgumentException>(() =>
            new Expert<float>(null, new[] { 10 }, new[] { 10 }));
#pragma warning restore CS8625
    }

    [Fact]
    public void Forward_WithValidInput_ReturnsCorrectShape()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(10, 20, new ReLUActivation<float>()),
            new DenseLayer<float>(20, 10, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 10 }, new[] { 10 });
        var input = new Tensor<float>(new[] { 1, 10 });
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 0.5f;
        }

        // Act
        var output = expert.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(10, output.Shape[1]);
    }

    [Fact]
    public void Forward_ProcessesDataThroughAllLayers()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(5, 3, new ReLUActivation<float>()),
            new DenseLayer<float>(3, 2, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 5 }, new[] { 2 });
        var input = new Tensor<float>(new[] { 1, 5 });
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 1.0f;
        }

        // Act
        var output = expert.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(1, output.Shape[0]);
        Assert.Equal(2, output.Shape[1]);
        // Verify output is not all zeros (layers are processing)
        bool hasNonZero = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (output[i] != 0.0f)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "Expert should produce non-zero output");
    }

    [Fact]
    public void Backward_WithValidGradient_ReturnsCorrectShape()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(10, 20, new ReLUActivation<float>()),
            new DenseLayer<float>(20, 10, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 10 }, new[] { 10 });
        var input = new Tensor<float>(new[] { 1, 10 });
        var outputGradient = new Tensor<float>(new[] { 1, 10 });
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 0.5f;
            outputGradient[i] = 0.1f;
        }

        // Act
        expert.Forward(input);
        var inputGradient = expert.Backward(outputGradient);

        // Assert
        Assert.Equal(input.Rank, inputGradient.Rank);
        Assert.Equal(input.Shape[0], inputGradient.Shape[0]);
        Assert.Equal(input.Shape[1], inputGradient.Shape[1]);
    }

    [Fact]
    public void UpdateParameters_ModifiesLayerParameters()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(5, 5, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 5 }, new[] { 5 });
        var input = new Tensor<float>(new[] { 1, 5 });
        var outputGradient = new Tensor<float>(new[] { 1, 5 });
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 1.0f;
            outputGradient[i] = 0.1f;
        }

        // Get initial parameters
        var initialParams = expert.GetParameters();

        // Act
        expert.Forward(input);
        expert.Backward(outputGradient);
        expert.UpdateParameters(0.01f);
        var updatedParams = expert.GetParameters();

        // Assert
        Assert.Equal(initialParams.Length, updatedParams.Length);
        // At least one parameter should have changed
        bool hasChanged = false;
        for (int i = 0; i < initialParams.Length; i++)
        {
            if (initialParams[i] != updatedParams[i])
            {
                hasChanged = true;
                break;
            }
        }
        Assert.True(hasChanged, "Parameters should change after update");
    }

    [Fact]
    public void GetParameters_ReturnsAllLayerParameters()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(5, 3, new ReLUActivation<float>()),
            new DenseLayer<float>(3, 2, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 5 }, new[] { 2 });

        // Act
        var parameters = expert.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.Equal(expert.ParameterCount, parameters.Length);
        Assert.True(parameters.Length > 0);
    }

    [Fact]
    public void SetParameters_UpdatesAllLayerParameters()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(5, 3, new ReLUActivation<float>()),
            new DenseLayer<float>(3, 2, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 5 }, new[] { 2 });
        var originalParams = expert.GetParameters();

        // Create new parameters (all zeros)
        var newParams = new Vector<float>(new float[expert.ParameterCount]);

        // Act
        expert.SetParameters(newParams);
        var retrievedParams = expert.GetParameters();

        // Assert
        for (int i = 0; i < retrievedParams.Length; i++)
        {
            Assert.Equal(0.0f, retrievedParams[i]);
        }
    }

    [Fact]
    public void SetParameters_WithIncorrectLength_ThrowsArgumentException()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(5, 3, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 5 }, new[] { 3 });
        var wrongSizeParams = new Vector<float>(new float[10]); // Wrong size

        // Act & Assert
        Assert.Throws<ArgumentException>(() => expert.SetParameters(wrongSizeParams));
    }

    [Fact]
    public void ResetState_ClearsLayerStates()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(5, 5, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 5 }, new[] { 5 });
        var input = new Tensor<float>(new[] { 1, 5 });
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 1.0f;
        }

        // Act
        expert.Forward(input);
        expert.ResetState();

        // Assert - Should not throw even after reset
        var output = expert.Forward(input);
        Assert.NotNull(output);
    }

    [Fact]
    public void Clone_CreatesIndependentCopy()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(5, 3, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 5 }, new[] { 3 });
        var input = new Tensor<float>(new[] { 1, 5 });
        var gradient = new Tensor<float>(new[] { 1, 3 });
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = 1.0f;
        }
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = 0.1f;
        }

        // Act
        var clone = expert.Clone();

        // Update original
        expert.Forward(input);
        expert.Backward(gradient);
        expert.UpdateParameters(0.1f);

        var originalParams = expert.GetParameters();
        var clonedParams = ((Expert<float>)clone).GetParameters();

        // Assert
        Assert.NotNull(clone);
        Assert.IsType<Expert<float>>(clone);
        // Parameters should be different after updating original
        bool hasDifference = false;
        for (int i = 0; i < originalParams.Length; i++)
        {
            if (originalParams[i] != clonedParams[i])
            {
                hasDifference = true;
                break;
            }
        }
        Assert.True(hasDifference, "Clone should be independent of original");
    }

    [Fact]
    public void ParameterCount_ReflectsSumOfAllLayers()
    {
        // Arrange
        var layer1 = new DenseLayer<float>(10, 5); // 10*5 + 5 = 55 parameters
        var layer2 = new DenseLayer<float>(5, 3);  // 5*3 + 3 = 18 parameters
        var layers = new List<ILayer<float>> { layer1, layer2 };

        var expert = new Expert<float>(layers, new[] { 10 }, new[] { 3 });

        // Act
        int paramCount = expert.ParameterCount;
        int expectedCount = layer1.ParameterCount + layer2.ParameterCount;

        // Assert
        Assert.Equal(expectedCount, paramCount);
    }

    [Fact]
    public void SupportsTraining_ReturnsTrueWhenAnyLayerIsTrainable()
    {
        // Arrange
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(5, 3, new ReLUActivation<float>())
        };
        var expert = new Expert<float>(layers, new[] { 5 }, new[] { 3 });

        // Act & Assert
        Assert.True(expert.SupportsTraining);
    }
}
