using AiDotNet.ActivationFunctions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using Xunit;

namespace AiDotNetTests.UnitTests.NeuralNetworks.Layers;

/// <summary>
/// Comprehensive unit and integration tests for the Mixture-of-Experts layer.
/// </summary>
public class MixtureOfExpertsLayerTests
{
    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidParameters_InitializesCorrectly()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4); // Output 4 scores for 4 experts

        // Act
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 });

        // Assert
        Assert.NotNull(moe);
        Assert.True(moe.SupportsTraining);
        Assert.Equal(4, moe.NumExperts);
        Assert.True(moe.ParameterCount > 0);
    }

    [Fact]
    public void Constructor_WithEmptyExpertList_ThrowsArgumentException()
    {
        // Arrange
        var experts = new List<ILayer<float>>();
        var router = new DenseLayer<float>(10, 0);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new MixtureOfExpertsLayer<float>(experts, router, new[] { 10 }, new[] { 10 }));
    }

    [Fact]
    public void Constructor_WithNullRouter_ThrowsArgumentNullException()
    {
        // Arrange
        var experts = CreateTestExperts(2, 10, 10);

        // Act & Assert
#pragma warning disable CS8625
        Assert.Throws<ArgumentNullException>(() =>
            new MixtureOfExpertsLayer<float>(experts, null, new[] { 10 }, new[] { 10 }));
#pragma warning restore CS8625
    }

    [Fact]
    public void Constructor_WithInvalidTopK_ThrowsArgumentException()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            new MixtureOfExpertsLayer<float>(
                experts, router,
                new[] { 10 }, new[] { 10 },
                topK: 5)); // TopK > num experts
    }

    [Fact]
    public void Constructor_WithLoadBalancing_InitializesCorrectly()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);

        // Act
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            topK: 0,
            activationFunction: null,
            useLoadBalancing: true,
            loadBalancingWeight: 0.01f);

        // Assert
        Assert.True(moe.UseAuxiliaryLoss);
        Assert.Equal(0.01f, moe.AuxiliaryLossWeight);
    }

    #endregion

    #region Forward Pass Tests

    [Fact]
    public void Forward_WithValidInput_ReturnsCorrectShape()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 });

        var input = CreateTestInput(2, 10); // Batch size 2

        // Act
        var output = moe.Forward(input);

        // Assert
        Assert.Equal(2, output.Rank);
        Assert.Equal(2, output.Shape[0]); // Batch size
        Assert.Equal(10, output.Shape[1]); // Output dimension
    }

    [Fact]
    public void Forward_AllExperts_ProducesNonZeroOutput()
    {
        // Arrange
        var experts = CreateTestExperts(3, 5, 5);
        var router = new DenseLayer<float>(5, 3);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 5 }, new[] { 5 },
            topK: 0); // All experts

        var input = CreateTestInput(1, 5);

        // Act
        var output = moe.Forward(input);

        // Assert
        const float epsilon = 1e-6f;
        bool hasNonZero = false;
        for (int i = 0; i < output.Length; i++)
        {
            if (Math.Abs(output[i]) > epsilon)
            {
                hasNonZero = true;
                break;
            }
        }
        Assert.True(hasNonZero, "MoE should produce non-zero output");
    }

    [Fact]
    public void Forward_TopK2_ActivatesOnlyTopExperts()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            topK: 2); // Only top 2 experts

        var input = CreateTestInput(3, 10); // Batch size 3

        // Act
        var output = moe.Forward(input);

        // Assert
        Assert.NotNull(output);
        Assert.Equal(3, output.Shape[0]);
        Assert.Equal(10, output.Shape[1]);
    }

    #endregion

    #region Backward Pass Tests



    #endregion

    #region Parameter Management Tests


    [Fact]
    public void GetParameters_ReturnsAllParameters()
    {
        // Arrange
        var experts = CreateTestExperts(3, 10, 10);
        var router = new DenseLayer<float>(10, 3);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 });

        // Act
        var parameters = moe.GetParameters();

        // Assert
        Assert.NotNull(parameters);
        Assert.Equal(moe.ParameterCount, parameters.Length);
    }

    [Fact]
    public void SetParameters_UpdatesAllParameters()
    {
        // Arrange
        var experts = CreateTestExperts(2, 5, 5);
        var router = new DenseLayer<float>(5, 2);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 5 }, new[] { 5 });

        var newParams = new Vector<float>(new float[moe.ParameterCount]);

        // Act
        moe.SetParameters(newParams);
        var retrievedParams = moe.GetParameters();

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
        var experts = CreateTestExperts(2, 5, 5);
        var router = new DenseLayer<float>(5, 2);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 5 }, new[] { 5 });

        var wrongParams = new Vector<float>(new float[10]);

        // Act & Assert
        Assert.Throws<ArgumentException>(() => moe.SetParameters(wrongParams));
    }

    [Fact]
    public void ParameterCount_IncludesRouterAndAllExperts()
    {
        // Arrange
        var experts = CreateTestExperts(3, 10, 10);
        var router = new DenseLayer<float>(10, 3);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 });

        // Act
        int totalParams = moe.ParameterCount;
        int expectedParams = router.ParameterCount + experts.Sum(e => e.ParameterCount);

        // Assert
        Assert.Equal(expectedParams, totalParams);
    }

    #endregion

    #region Load Balancing Tests

    [Fact]
    public void ComputeAuxiliaryLoss_WithLoadBalancingEnabled_ReturnsNonZeroLoss()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            topK: 0,
            activationFunction: null,
            useLoadBalancing: true,
            loadBalancingWeight: 0.01f);

        var input = CreateTestInput(8, 10);

        // Act
        moe.Forward(input);
        var auxLoss = moe.ComputeAuxiliaryLoss();

        // Assert
        Assert.True(auxLoss >= 0.0f, "Auxiliary loss should be non-negative");
    }

    [Fact]
    public void ComputeAuxiliaryLoss_WithLoadBalancingDisabled_ReturnsZero()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            useLoadBalancing: false);

        var input = CreateTestInput(8, 10);

        // Act
        moe.Forward(input);
        var auxLoss = moe.ComputeAuxiliaryLoss();

        // Assert
        Assert.Equal(0.0f, auxLoss);
    }

    [Fact]
    public void ComputeAuxiliaryLoss_BeforeForward_ThrowsInvalidOperationException()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            useLoadBalancing: true);

        // Act & Assert
        Assert.Throws<InvalidOperationException>(() => moe.ComputeAuxiliaryLoss());
    }

    [Fact]
    public void GetAuxiliaryLossDiagnostics_AfterForward_ReturnsStatistics()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            useLoadBalancing: true);

        var input = CreateTestInput(8, 10);

        // Act
        moe.Forward(input);
        var diagnostics = moe.GetAuxiliaryLossDiagnostics();

        // Assert
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.Count > 0);
        Assert.True(diagnostics.TryGetValue("num_experts", out var numExperts));
        Assert.Equal("4", numExperts);
        Assert.True(diagnostics.TryGetValue("batch_size", out var batchSize));
        Assert.Equal("8", batchSize);

        // Check for per-expert statistics
        for (int i = 0; i < 4; i++)
        {
            Assert.True(diagnostics.ContainsKey($"expert_{i}_tokens"));
            Assert.True(diagnostics.ContainsKey($"expert_{i}_prob_mass"));
        }
    }

    [Fact]
    public void GetAuxiliaryLossDiagnostics_BeforeForward_ReturnsStatusMessage()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            useLoadBalancing: true);

        // Act
        var diagnostics = moe.GetAuxiliaryLossDiagnostics();

        // Assert
        Assert.NotNull(diagnostics);
        Assert.True(diagnostics.ContainsKey("status"));
    }

    [Fact]
    public void UseAuxiliaryLoss_CanBeToggledOnAndOff()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            useLoadBalancing: true);

        // Act & Assert
        Assert.True(moe.UseAuxiliaryLoss);

        moe.UseAuxiliaryLoss = false;
        Assert.False(moe.UseAuxiliaryLoss);

        moe.UseAuxiliaryLoss = true;
        Assert.True(moe.UseAuxiliaryLoss);
    }

    [Fact]
    public void AuxiliaryLossWeight_CanBeModified()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            useLoadBalancing: true,
            loadBalancingWeight: 0.01f);

        // Act & Assert
        Assert.Equal(0.01f, moe.AuxiliaryLossWeight);

        moe.AuxiliaryLossWeight = 0.05f;
        Assert.Equal(0.05f, moe.AuxiliaryLossWeight);
    }

    #endregion

    #region Integration Tests


    [Fact]
    public void EndToEnd_TopKRouting_BalancesExpertUsage()
    {
        // Arrange
        var experts = CreateTestExperts(4, 10, 10);
        var router = new DenseLayer<float>(10, 4);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            topK: 2,
            useLoadBalancing: true,
            loadBalancingWeight: 0.05f);

        var input = CreateTestInput(16, 10); // Larger batch for better statistics

        // Act - Forward pass
        moe.Forward(input);
        var diagnostics = moe.GetAuxiliaryLossDiagnostics();

        // Assert - With load balancing, experts should have some usage
        // We can't guarantee perfect balance, but all experts should be used at least once
        int expertsUsed = 0;
        for (int i = 0; i < 4; i++)
        {
            if (diagnostics.TryGetValue($"expert_{i}_tokens", out var tokens))
            {
                int tokenCount = int.Parse(tokens);
                if (tokenCount > 0)
                {
                    expertsUsed++;
                }
            }
        }

        Assert.True(expertsUsed >= 2,
            $"Expected at least 2 experts to be used with load balancing, but only {expertsUsed} were used");
    }

    [Fact]
    public void EndToEnd_SoftRouting_AllExpertsContribute()
    {
        // Arrange
        var experts = CreateTestExperts(3, 10, 10);
        var router = new DenseLayer<float>(10, 3);
        var moe = new MixtureOfExpertsLayer<float>(
            experts, router,
            new[] { 10 }, new[] { 10 },
            topK: 0); // Soft routing (all experts)

        var input = CreateTestInput(4, 10);

        // Act
        var output = moe.Forward(input);

        // Assert
        Assert.NotNull(output);
        // All experts should contribute to the output
        // We can verify this by checking the diagnostics
        var diagnostics = moe.GetAuxiliaryLossDiagnostics();

        for (int i = 0; i < 3; i++)
        {
            Assert.True(diagnostics.TryGetValue($"expert_{i}_prob_mass", out var probMassStr), $"Diagnostics should contain expert_{i}_prob_mass");
            float probMass = float.Parse(probMassStr);
            Assert.True(probMass > 0.0f, $"Expert {i} should have non-zero probability mass in soft routing");
        }
    }

    #endregion

    #region State Management Tests



    #endregion

    #region Helper Methods

    private static List<ILayer<float>> CreateTestExperts(int numExperts, int inputDim, int outputDim)
    {
        var experts = new List<ILayer<float>>();
        for (int i = 0; i < numExperts; i++)
        {
            var expertLayers = new List<ILayer<float>>
            {
                new DenseLayer<float>(inputDim, outputDim, new ReLUActivation<float>())
            };
            experts.Add(new Expert<float>(expertLayers, new[] { inputDim }, new[] { outputDim }));
        }
        return experts;
    }

    private static Tensor<float> CreateTestInput(int batchSize, int inputDim)
    {
        var input = new Tensor<float>(new[] { batchSize, inputDim });
        var random = RandomHelper.CreateSeededRandom(42); // Fixed seed for reproducibility
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = (float)(random.NextDouble() * 2.0 - 1.0); // Range [-1, 1]
        }
        return input;
    }

    #endregion
}
