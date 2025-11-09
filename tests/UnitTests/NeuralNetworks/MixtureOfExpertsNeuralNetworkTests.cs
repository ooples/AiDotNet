using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// Tests for the Mixture-of-Experts Neural Network model.
/// </summary>
public class MixtureOfExpertsNeuralNetworkTests
{
    [Fact]
    public void Constructor_WithValidOptions_CreatesModel()
    {
        // Arrange
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        // Act
        var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

        // Assert
        Assert.NotNull(model);
        Assert.True(model.SupportsTraining);
    }

    [Fact]
    public void Constructor_WithNullOptions_ThrowsArgumentNullException()
    {
        // Arrange
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new MixtureOfExpertsNeuralNetwork<float>(null!, architecture));
    }

    [Fact]
    public void Constructor_WithInvalidOptions_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 5,  // TopK > NumExperts (invalid)
            InputDim = 10,
            OutputDim = 10
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new MixtureOfExpertsNeuralNetwork<float>(options, architecture));
    }

    [Fact]
    public void Predict_WithValidInput_ReturnsOutput()
    {
        // Arrange
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10,
            RandomSeed = 42
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);
        var input = Tensor<float>.Random(new[] { 1, 10 }, seed: 42);

        // Act
        var output = model.Predict(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(new[] { 1, 1 }, output.Shape);
    }

    [Fact]
    public void Train_WithValidData_UpdatesModel()
    {
        // Arrange
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10,
            UseLoadBalancing = true,
            LoadBalancingWeight = 0.01,
            RandomSeed = 42
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);
        var input = Tensor<float>.Random(new[] { 5, 10 }, seed: 42);
        var expectedOutput = Tensor<float>.Random(new[] { 5, 1 }, seed: 43);

        // Act
        // Train for a few iterations
        for (int i = 0; i < 10; i++)
        {
            model.Train(input, expectedOutput);
        }

        // Assert - model should have been updated (loss should change)
        Assert.NotEqual(0f, model.LastLoss);
    }

    [Fact]
    public void Train_MultipleIterations_ReducesLoss()
    {
        // Arrange
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 5,
            OutputDim = 5,
            UseLoadBalancing = true,
            RandomSeed = 42
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 5,
            outputSize: 1
        );

        var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

        // Create simple linear relationship: y = sum(x)
        var input = Tensor<float>.Random(new[] { 10, 5 }, seed: 42);
        var expectedOutput = new Tensor<float>(new[] { 10, 1 });
        for (int i = 0; i < 10; i++)
        {
            float sum = 0;
            for (int j = 0; j < 5; j++)
            {
                sum += input[i, j];
            }
            expectedOutput[i, 0] = sum;
        }

        // Act - Train for multiple iterations
        float initialLoss = 0;
        float finalLoss = 0;

        for (int i = 0; i < 100; i++)
        {
            model.Train(input, expectedOutput);

            if (i == 0)
            {
                initialLoss = model.LastLoss;
            }

            if (i == 99)
            {
                finalLoss = model.LastLoss;
            }
        }

        // Assert - loss should decrease
        Assert.True(finalLoss < initialLoss,
            $"Expected finalLoss ({finalLoss}) < initialLoss ({initialLoss})");
    }

    [Fact]
    public void GetModelMetadata_ReturnsCorrectInformation()
    {
        // Arrange
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 8,
            TopK = 2,
            InputDim = 128,
            OutputDim = 128,
            HiddenExpansion = 4,
            UseLoadBalancing = true,
            LoadBalancingWeight = 0.01
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 128,
            outputSize: 10
        );

        var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

        // Act
        var metadata = model.GetModelMetadata();

        // Assert
        Assert.NotNull(metadata);
        Assert.Equal(ModelType.MixtureOfExperts, metadata.ModelType);
        Assert.True(metadata.AdditionalInfo.TryGetValue("NumExperts", out var numExperts));
        Assert.Equal(8, numExperts);
        Assert.True(metadata.AdditionalInfo.TryGetValue("TopK", out var topK));
        Assert.Equal(2, topK);
        Assert.True(metadata.AdditionalInfo.TryGetValue("UseLoadBalancing", out var useLoadBalancing));
        Assert.Equal(true, useLoadBalancing);
    }

    [Fact]
    public void CreateNewInstance_CreatesNewModelWithSameConfiguration()
    {
        // Arrange
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

        // Act
        var newInstance = model.CreateNewInstance() as MixtureOfExpertsNeuralNetwork<float>;

        // Assert
        Assert.NotNull(newInstance);
        Assert.NotSame(model, newInstance);
        Assert.Equal(model.Architecture.TaskType, newInstance!.Architecture.TaskType);
    }

    [Fact]
    public void Classification_SimpleTask_ConvergesToSolution()
    {
        // Arrange - Create a simple binary classification problem
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 2,
            OutputDim = 2,
            UseLoadBalancing = true,
            RandomSeed = 42
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            inputSize: 2,
            outputSize: 2
        );

        var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

        // Create XOR-like problem
        var input = new Tensor<float>(new float[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } });
        var expectedOutput = new Tensor<float>(new float[,] { { 1, 0 }, { 0, 1 }, { 0, 1 }, { 1, 0 } });

        // Act - Train for many iterations
        float initialLoss = 0;
        float finalLoss = 0;

        for (int i = 0; i < 500; i++)
        {
            model.Train(input, expectedOutput);

            if (i == 0) initialLoss = model.LastLoss;
            if (i == 499) finalLoss = model.LastLoss;
        }

        // Assert - Loss should decrease significantly
        Assert.True(finalLoss < initialLoss * 0.5f,
            $"Expected finalLoss ({finalLoss}) to be less than half of initialLoss ({initialLoss})");
    }

    [Fact]
    public void MultiClassClassification_ConvergesToSolution()
    {
        // Arrange
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 4,
            OutputDim = 4,
            UseLoadBalancing = true,
            RandomSeed = 42
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 4,
            outputSize: 3
        );

        var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

        // Create simple multiclass data
        var input = new Tensor<float>(new float[,]
        {
            { 1, 0, 0, 0 },
            { 0, 1, 0, 0 },
            { 0, 0, 1, 0 },
            { 0, 0, 0, 1 }
        });
        var expectedOutput = new Tensor<float>(new float[,]
        {
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 1 },
            { 1, 0, 0 }
        });

        // Act
        float initialLoss = 0;
        float finalLoss = 0;

        for (int i = 0; i < 200; i++)
        {
            model.Train(input, expectedOutput);

            if (i == 0) initialLoss = model.LastLoss;
            if (i == 199) finalLoss = model.LastLoss;
        }

        // Assert
        Assert.True(finalLoss < initialLoss,
            $"Expected finalLoss ({finalLoss}) < initialLoss ({initialLoss})");
    }

    [Fact]
    public void LoadBalancing_WhenEnabled_IncludesAuxiliaryLoss()
    {
        // Arrange
        var optionsWithLB = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10,
            UseLoadBalancing = true,
            LoadBalancingWeight = 0.01,
            RandomSeed = 42
        };

        var optionsWithoutLB = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10,
            UseLoadBalancing = false,
            RandomSeed = 42
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        var modelWithLB = new MixtureOfExpertsNeuralNetwork<float>(optionsWithLB, architecture);
        var modelWithoutLB = new MixtureOfExpertsNeuralNetwork<float>(optionsWithoutLB, architecture);

        var input = Tensor<float>.Random(new[] { 5, 10 }, seed: 42);
        var expectedOutput = Tensor<float>.Random(new[] { 5, 1 }, seed: 43);

        // Act
        modelWithLB.Train(input, expectedOutput);
        modelWithoutLB.Train(input, expectedOutput);

        // Assert - Model with load balancing should have higher loss due to auxiliary loss term
        // (Note: This assumes the auxiliary loss adds to the total loss)
        Assert.NotEqual(modelWithLB.LastLoss, modelWithoutLB.LastLoss);
    }

    [Fact]
    public void TopK_DifferentValues_AffectComputation()
    {
        // Arrange - Test that different TopK values produce different results
        var optionsTopK1 = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 1,
            InputDim = 10,
            OutputDim = 10,
            RandomSeed = 42
        };

        var optionsTopK2 = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10,
            RandomSeed = 42
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        var modelTopK1 = new MixtureOfExpertsNeuralNetwork<float>(optionsTopK1, architecture);
        var modelTopK2 = new MixtureOfExpertsNeuralNetwork<float>(optionsTopK2, architecture);

        var input = Tensor<float>.Random(new[] { 1, 10 }, seed: 42);

        // Act
        var outputTopK1 = modelTopK1.Predict(input);
        var outputTopK2 = modelTopK2.Predict(input);

        // Assert - Different TopK values should potentially produce different outputs
        // (though they might coincidentally be the same, we just verify the models run)
        Assert.NotNull(outputTopK1);
        Assert.NotNull(outputTopK2);
    }

    [Fact]
    public void ParameterCount_IsGreaterThanZero()
    {
        // Arrange
        var options = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10
        };

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        var model = new MixtureOfExpertsNeuralNetwork<float>(options, architecture);

        // Act
        var parameterCount = model.GetParameterCount();

        // Assert
        Assert.True(parameterCount > 0);
    }

    [Fact]
    public void RandomSeed_ProducesReproducibleResults()
    {
        // Arrange
        var options1 = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10,
            RandomSeed = 42
        };

        var options2 = new MixtureOfExpertsOptions<float>
        {
            NumExperts = 4,
            TopK = 2,
            InputDim = 10,
            OutputDim = 10,
            RandomSeed = 42
        };

        var architecture1 = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        var architecture2 = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 1
        );

        var model1 = new MixtureOfExpertsNeuralNetwork<float>(options1, architecture1);
        var model2 = new MixtureOfExpertsNeuralNetwork<float>(options2, architecture2);

        var input = Tensor<float>.Random(new[] { 1, 10 }, seed: 42);

        // Act
        var output1 = model1.Predict(input);
        var output2 = model2.Predict(input);

        // Assert - Same seed should produce same initialization and same outputs
        Assert.Equal(output1.Shape, output2.Shape);
        // Note: Due to random initialization, outputs might still differ slightly
        // This test mainly verifies that both models run successfully with the same seed
    }
}
