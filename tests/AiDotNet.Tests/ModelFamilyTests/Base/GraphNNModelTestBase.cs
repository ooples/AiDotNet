using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for Graph Neural Network models.
/// Inherits all neural network invariant tests and adds graph-specific invariants:
/// self-loop stability, zero-input robustness, and structural sensitivity.
/// </summary>
public abstract class GraphNNModelTestBase : NeuralNetworkModelTestBase
{
    // =====================================================
    // GRAPH INVARIANT: Self-Loops Should Not Cause Numerical Issues
    // A diagonal-heavy adjacency matrix (self-loops) is common in graph
    // networks. The model should handle it without producing NaN/Inf.
    // =====================================================

    [Fact]
    public void SelfLoops_ShouldNotCauseNumericalIssues()
    {
        var network = CreateNetwork();

        // Create a diagonal-heavy input (simulating self-loops in adjacency)
        var input = new Tensor<double>(InputShape);
        int totalSize = 1;
        foreach (var d in InputShape) totalSize *= d;

        // Fill with small values, then set diagonal-like positions to 1.0
        for (int i = 0; i < input.Length; i++)
            input[i] = 0.01;

        // Set diagonal elements (stride = last_dim + 1 for square-ish tensors)
        int lastDim = InputShape[InputShape.Length - 1];
        for (int i = 0; i < input.Length; i += lastDim + 1)
        {
            if (i < input.Length)
                input[i] = 1.0;
        }

        var output = network.Predict(input);

        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN with self-loop input — numerical instability.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity with self-loop input — overflow.");
        }
    }

    // =====================================================
    // GRAPH INVARIANT: Zero Input Should Not Crash
    // An all-zero input (isolated nodes, no features) is a valid edge case.
    // The model should produce finite, non-empty output.
    // =====================================================

    [Fact]
    public void ZeroInput_ShouldNotCrash()
    {
        var network = CreateNetwork();

        var input = new Tensor<double>(InputShape);
        // All zeros by default

        var output = network.Predict(input);

        Assert.True(output.Length > 0, "Output should not be empty for zero input.");
        for (int i = 0; i < output.Length; i++)
        {
            Assert.False(double.IsNaN(output[i]),
                $"Output[{i}] is NaN for zero input — model should handle empty graphs.");
            Assert.False(double.IsInfinity(output[i]),
                $"Output[{i}] is Infinity for zero input.");
        }
    }

    // =====================================================
    // GRAPH INVARIANT: Structural Sensitivity
    // Structurally different inputs should produce different outputs.
    // A graph network that ignores structure is fundamentally broken.
    // =====================================================

    [Fact]
    public void DifferentStructures_ProduceDifferentOutputs()
    {
        var network = CreateNetwork();

        // Input 1: identity-like structure (strong self-connections)
        var input1 = new Tensor<double>(InputShape);
        for (int i = 0; i < input1.Length; i++)
            input1[i] = 0.1;
        int lastDim1 = InputShape[InputShape.Length - 1];
        for (int i = 0; i < input1.Length; i += lastDim1 + 1)
        {
            if (i < input1.Length)
                input1[i] = 1.0;
        }

        // Input 2: uniform structure (all connections equal)
        var input2 = new Tensor<double>(InputShape);
        for (int i = 0; i < input2.Length; i++)
            input2[i] = 0.5;

        var output1 = network.Predict(input1);
        var output2 = network.Predict(input2);

        bool anyDifferent = false;
        int minLen = Math.Min(output1.Length, output2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(output1[i] - output2[i]) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "Graph network produces identical output for structurally different inputs — " +
            "model may be ignoring graph structure.");
    }
}
