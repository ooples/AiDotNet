using AiDotNet.ActivationFunctions;
using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Autodiff;

/// <summary>
/// Integration tests for NeuralNetworkDerivatives to verify gradient and Hessian computation
/// for neural networks with various architectures and activation functions.
/// </summary>
/// <remarks>
/// <para>
/// These tests verify that the analytic derivatives computed via automatic differentiation
/// match numerical derivatives computed via finite differences.
/// </para>
/// <para><b>For Beginners:</b> These tests ensure our neural network derivatives are correct.
///
/// We test:
/// 1. First derivatives (gradients/Jacobian) - how outputs change with inputs
/// 2. Second derivatives (Hessian) - curvature of the output surface
/// 3. Different activation functions (ReLU, Sigmoid, Tanh, etc.)
/// 4. Different network architectures (single layer, multi-layer)
///
/// The key verification is comparing analytic gradients (fast, exact) with
/// numerical gradients (slow, approximate but reliable).
/// </para>
/// </remarks>
public class NeuralNetworkDerivativesIntegrationTests
{
    private const double Tolerance = 1e-3;
    private const double HessianTolerance = 5e-2; // Second derivatives are less precise

    #region Basic Gradient Tests

    [Fact]
    public void ComputeGradient_SingleDenseLayer_IdentityActivation_MatchesNumerical()
    {
        // Arrange: Create a simple 2-input, 1-output network with identity activation
        var network = CreateNetwork(new[] { 2, 1 }, useReLU: false);
        var inputs = new double[] { 0.5, 0.8 };

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 1);

        // Assert: For identity activation, gradient should equal the weights
        Assert.NotNull(gradients);
        Assert.Equal(1, gradients.GetLength(0)); // 1 output
        Assert.Equal(2, gradients.GetLength(1)); // 2 inputs

        // Verify with numerical gradient
        var numericalGradients = ComputeNumericalGradient(network, inputs, 1);
        for (int i = 0; i < 2; i++)
        {
            Assert.Equal(numericalGradients[0, i], gradients[0, i], Tolerance);
        }
    }

    [Fact]
    public void ComputeGradient_SingleDenseLayer_ReLUActivation_MatchesNumerical()
    {
        // Arrange: Network with ReLU activation
        var network = CreateNetwork(new[] { 3, 2 }, useReLU: true);
        var inputs = new double[] { 0.3, -0.5, 0.7 }; // Mix of positive and negative inputs

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 2);

        // Assert
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradient(network, inputs, 2);

        for (int outIdx = 0; outIdx < 2; outIdx++)
        {
            for (int inIdx = 0; inIdx < 3; inIdx++)
            {
                Assert.Equal(numericalGradients[outIdx, inIdx], gradients[outIdx, inIdx], Tolerance);
            }
        }
    }

    [Fact]
    public void ComputeGradient_SingleDenseLayer_SigmoidActivation_MatchesNumerical()
    {
        // Arrange: Network with Sigmoid activation
        var network = CreateNetworkWithActivation(new[] { 2, 2 }, new SigmoidActivation<double>());
        var inputs = new double[] { 0.5, -0.3 };

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 2);

        // Assert
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradient(network, inputs, 2);

        for (int outIdx = 0; outIdx < 2; outIdx++)
        {
            for (int inIdx = 0; inIdx < 2; inIdx++)
            {
                Assert.Equal(numericalGradients[outIdx, inIdx], gradients[outIdx, inIdx], Tolerance);
            }
        }
    }

    [Fact]
    public void ComputeGradient_SingleDenseLayer_TanhActivation_MatchesNumerical()
    {
        // Arrange: Network with Tanh activation
        var network = CreateNetworkWithActivation(new[] { 2, 2 }, new TanhActivation<double>());
        var inputs = new double[] { 0.4, 0.6 };

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 2);

        // Assert
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradient(network, inputs, 2);

        for (int outIdx = 0; outIdx < 2; outIdx++)
        {
            for (int inIdx = 0; inIdx < 2; inIdx++)
            {
                Assert.Equal(numericalGradients[outIdx, inIdx], gradients[outIdx, inIdx], Tolerance);
            }
        }
    }

    [Fact]
    public void ComputeGradient_SingleDenseLayer_LeakyReLUActivation_MatchesNumerical()
    {
        // Arrange: Network with LeakyReLU activation
        var network = CreateNetworkWithActivation(new[] { 2, 2 }, new LeakyReLUActivation<double>());
        var inputs = new double[] { 0.5, -0.5 }; // One positive, one negative

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 2);

        // Assert
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradient(network, inputs, 2);

        for (int outIdx = 0; outIdx < 2; outIdx++)
        {
            for (int inIdx = 0; inIdx < 2; inIdx++)
            {
                Assert.Equal(numericalGradients[outIdx, inIdx], gradients[outIdx, inIdx], Tolerance);
            }
        }
    }

    [Fact]
    public void ComputeGradient_SingleDenseLayer_GELUActivation_MatchesNumerical()
    {
        // Arrange: Network with GELU activation
        var network = CreateNetworkWithActivation(new[] { 2, 2 }, new GELUActivation<double>());
        var inputs = new double[] { 0.3, 0.7 };

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 2);

        // Assert
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradient(network, inputs, 2);

        for (int outIdx = 0; outIdx < 2; outIdx++)
        {
            for (int inIdx = 0; inIdx < 2; inIdx++)
            {
                Assert.Equal(numericalGradients[outIdx, inIdx], gradients[outIdx, inIdx], Tolerance);
            }
        }
    }

    #endregion

    #region Multi-Layer Network Tests

    [Fact]
    public void ComputeGradient_TwoLayerNetwork_IdentityActivation_MatchesNumerical()
    {
        // Arrange: Two-layer network: 2 -> 3 -> 1
        var network = CreateNetwork(new[] { 2, 3, 1 }, useReLU: false);
        var inputs = new double[] { 0.5, 0.8 };

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 1);

        // Assert
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradient(network, inputs, 1);

        for (int inIdx = 0; inIdx < 2; inIdx++)
        {
            Assert.Equal(numericalGradients[0, inIdx], gradients[0, inIdx], Tolerance);
        }
    }

    [Fact]
    public void ComputeGradient_ThreeLayerNetwork_MixedActivations_MatchesNumerical()
    {
        // Arrange: Three-layer network with different activations
        var network = CreateMultiActivationNetwork(new[] { 3, 4, 3, 2 },
            new IActivationFunction<double>[]
            {
                new ReLUActivation<double>(),
                new TanhActivation<double>(),
                new SigmoidActivation<double>()
            });
        var inputs = new double[] { 0.2, 0.5, 0.9 };

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 2);

        // Assert
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradient(network, inputs, 2);

        for (int outIdx = 0; outIdx < 2; outIdx++)
        {
            for (int inIdx = 0; inIdx < 3; inIdx++)
            {
                Assert.Equal(numericalGradients[outIdx, inIdx], gradients[outIdx, inIdx], Tolerance);
            }
        }
    }

    [Fact]
    public void ComputeGradient_DeepNetwork_SigmoidActivation_MatchesNumerical()
    {
        // Arrange: Deep network: 2 -> 4 -> 4 -> 4 -> 2
        var network = CreateNetworkWithActivation(new[] { 2, 4, 4, 4, 2 }, new SigmoidActivation<double>());
        var inputs = new double[] { 0.4, 0.6 };

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 2);

        // Assert
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradient(network, inputs, 2);

        for (int outIdx = 0; outIdx < 2; outIdx++)
        {
            for (int inIdx = 0; inIdx < 2; inIdx++)
            {
                Assert.Equal(numericalGradients[outIdx, inIdx], gradients[outIdx, inIdx], Tolerance);
            }
        }
    }

    #endregion

    #region Hessian Tests

    [Fact]
    public void ComputeHessian_SingleDenseLayer_IdentityActivation_IsZero()
    {
        // Arrange: For identity activation, Hessian should be zero (linear function)
        var network = CreateNetwork(new[] { 2, 1 }, useReLU: false);
        var inputs = new double[] { 0.5, 0.8 };

        // Act
        var hessian = NeuralNetworkDerivatives<double>.ComputeHessian(network, inputs, 0);

        // Assert: Identity activation should have zero Hessian
        Assert.NotNull(hessian);
        Assert.Equal(2, hessian.GetLength(0));
        Assert.Equal(2, hessian.GetLength(1));

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(0.0, hessian[i, j], HessianTolerance);
            }
        }
    }

    [Fact]
    public void ComputeHessian_SingleDenseLayer_SigmoidActivation_IsSymmetric()
    {
        // Arrange: Hessians should be symmetric for any smooth function
        var network = CreateNetworkWithActivation(new[] { 3, 1 }, new SigmoidActivation<double>());
        var inputs = new double[] { 0.3, 0.5, 0.7 };

        // Act
        var hessian = NeuralNetworkDerivatives<double>.ComputeHessian(network, inputs, 0);

        // Assert: Hessian should be symmetric
        Assert.NotNull(hessian);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Assert.Equal(hessian[i, j], hessian[j, i], HessianTolerance);
            }
        }
    }

    [Fact]
    public void ComputeHessian_SingleDenseLayer_SigmoidActivation_MatchesNumerical()
    {
        // Arrange
        var network = CreateNetworkWithActivation(new[] { 2, 1 }, new SigmoidActivation<double>());
        var inputs = new double[] { 0.4, 0.6 };

        // Act
        var hessian = NeuralNetworkDerivatives<double>.ComputeHessian(network, inputs, 0);

        // Assert
        Assert.NotNull(hessian);
        var numericalHessian = ComputeNumericalHessian(network, inputs, 0);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(numericalHessian[i, j], hessian[i, j], HessianTolerance);
            }
        }
    }

    [Fact]
    public void ComputeHessian_SingleDenseLayer_TanhActivation_MatchesNumerical()
    {
        // Arrange
        var network = CreateNetworkWithActivation(new[] { 2, 1 }, new TanhActivation<double>());
        var inputs = new double[] { 0.3, -0.3 };

        // Act
        var hessian = NeuralNetworkDerivatives<double>.ComputeHessian(network, inputs, 0);

        // Assert
        Assert.NotNull(hessian);
        var numericalHessian = ComputeNumericalHessian(network, inputs, 0);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(numericalHessian[i, j], hessian[i, j], HessianTolerance);
            }
        }
    }

    [Fact]
    public void ComputeHessian_TwoLayerNetwork_SigmoidActivation_MatchesNumerical()
    {
        // Arrange: Two-layer network
        var network = CreateNetworkWithActivation(new[] { 2, 3, 1 }, new SigmoidActivation<double>());
        var inputs = new double[] { 0.5, 0.5 };

        // Act
        var hessian = NeuralNetworkDerivatives<double>.ComputeHessian(network, inputs, 0);

        // Assert
        Assert.NotNull(hessian);
        var numericalHessian = ComputeNumericalHessian(network, inputs, 0);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(numericalHessian[i, j], hessian[i, j], HessianTolerance);
            }
        }
    }

    #endregion

    #region ComputeDerivatives Tests

    [Fact]
    public void ComputeDerivatives_ReturnsFirstAndSecondDerivatives()
    {
        // Arrange
        var network = CreateNetworkWithActivation(new[] { 2, 1 }, new SigmoidActivation<double>());
        var inputs = new double[] { 0.5, 0.5 };

        // Act
        var derivatives = NeuralNetworkDerivatives<double>.ComputeDerivatives(network, inputs, 1);

        // Assert
        Assert.NotNull(derivatives);
        Assert.NotNull(derivatives.FirstDerivatives);
        Assert.NotNull(derivatives.SecondDerivatives);
        Assert.Equal(1, derivatives.FirstDerivatives.GetLength(0)); // 1 output
        Assert.Equal(2, derivatives.FirstDerivatives.GetLength(1)); // 2 inputs
        Assert.Equal(1, derivatives.SecondDerivatives.GetLength(0)); // 1 output
        Assert.Equal(2, derivatives.SecondDerivatives.GetLength(1)); // 2 inputs
        Assert.Equal(2, derivatives.SecondDerivatives.GetLength(2)); // 2 inputs
    }

    [Fact]
    public void ComputeDerivatives_FirstDerivatives_MatchComputeGradient()
    {
        // Arrange
        var network = CreateNetworkWithActivation(new[] { 2, 2 }, new TanhActivation<double>());
        var inputs = new double[] { 0.4, 0.6 };

        // Act
        var derivatives = NeuralNetworkDerivatives<double>.ComputeDerivatives(network, inputs, 2);
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 2);

        // Assert: First derivatives should match gradient
        for (int outIdx = 0; outIdx < 2; outIdx++)
        {
            for (int inIdx = 0; inIdx < 2; inIdx++)
            {
                Assert.Equal(gradients[outIdx, inIdx], derivatives.FirstDerivatives![outIdx, inIdx], 1e-10);
            }
        }
    }

    [Fact]
    public void ComputeDerivatives_SecondDerivatives_MatchComputeHessian()
    {
        // Arrange
        var network = CreateNetworkWithActivation(new[] { 2, 1 }, new SigmoidActivation<double>());
        var inputs = new double[] { 0.3, 0.7 };

        // Act
        var derivatives = NeuralNetworkDerivatives<double>.ComputeDerivatives(network, inputs, 1);
        var hessian = NeuralNetworkDerivatives<double>.ComputeHessian(network, inputs, 0);

        // Assert: Second derivatives should match Hessian
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                Assert.Equal(hessian[i, j], derivatives.SecondDerivatives![0, i, j], 1e-10);
            }
        }
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void ComputeGradient_NullNetwork_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            NeuralNetworkDerivatives<double>.ComputeGradient(null!, new double[] { 1.0 }, 1));
    }

    [Fact]
    public void ComputeGradient_NullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var network = CreateNetwork(new[] { 2, 1 }, useReLU: false);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            NeuralNetworkDerivatives<double>.ComputeGradient(network, null!, 1));
    }

    [Fact]
    public void ComputeDerivatives_NullNetwork_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            NeuralNetworkDerivatives<double>.ComputeDerivatives(null!, new double[] { 1.0 }, 1));
    }

    [Fact]
    public void ComputeDerivatives_NullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var network = CreateNetwork(new[] { 2, 1 }, useReLU: false);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            NeuralNetworkDerivatives<double>.ComputeDerivatives(network, null!, 1));
    }

    [Fact]
    public void ComputeDerivatives_ZeroOutputDim_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var network = CreateNetwork(new[] { 2, 1 }, useReLU: false);
        var inputs = new double[] { 0.5, 0.5 };

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            NeuralNetworkDerivatives<double>.ComputeDerivatives(network, inputs, 0));
    }

    [Fact]
    public void ComputeDerivatives_NegativeOutputDim_ThrowsArgumentOutOfRangeException()
    {
        // Arrange
        var network = CreateNetwork(new[] { 2, 1 }, useReLU: false);
        var inputs = new double[] { 0.5, 0.5 };

        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            NeuralNetworkDerivatives<double>.ComputeDerivatives(network, inputs, -1));
    }

    [Fact]
    public void ComputeHessian_NullNetwork_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            NeuralNetworkDerivatives<double>.ComputeHessian(null!, new double[] { 1.0 }, 0));
    }

    [Fact]
    public void ComputeHessian_NullInputs_ThrowsArgumentNullException()
    {
        // Arrange
        var network = CreateNetwork(new[] { 2, 1 }, useReLU: false);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            NeuralNetworkDerivatives<double>.ComputeHessian(network, null!, 0));
    }

    #endregion

    #region Float Type Tests

    [Fact]
    public void ComputeGradient_FloatType_MatchesNumerical()
    {
        // Arrange: Test with float instead of double
        var network = CreateNetworkFloat(new[] { 2, 1 });
        var inputs = new float[] { 0.5f, 0.8f };

        // Act
        var gradients = NeuralNetworkDerivatives<float>.ComputeGradient(network, inputs, 1);

        // Assert
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradientFloat(network, inputs, 1);

        // Use larger tolerance for float due to precision
        for (int inIdx = 0; inIdx < 2; inIdx++)
        {
            Assert.Equal(numericalGradients[0, inIdx], gradients[0, inIdx], 1e-2f);
        }
    }

    #endregion

    #region Chain Rule Verification

    [Fact]
    public void ChainRule_TwoLayerComposition_GradientIsProduct()
    {
        // Arrange: Create network where we can verify chain rule manually
        // For f(g(x)) where f and g are linear, df/dx = df/dg * dg/dx
        var network = CreateNetwork(new[] { 1, 1, 1 }, useReLU: false);
        var inputs = new double[] { 1.0 };

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 1);

        // Assert: Gradient should be product of weights (chain rule for linear functions)
        Assert.NotNull(gradients);

        // Verify with numerical gradient
        var numericalGradients = ComputeNumericalGradient(network, inputs, 1);
        Assert.Equal(numericalGradients[0, 0], gradients[0, 0], Tolerance);
    }

    [Fact]
    public void ChainRule_NonlinearActivation_GradientMatchesNumerical()
    {
        // Arrange: Test chain rule with nonlinear activation
        var network = CreateNetworkWithActivation(new[] { 1, 2, 1 }, new SigmoidActivation<double>());
        var inputs = new double[] { 0.5 };

        // Act
        var gradients = NeuralNetworkDerivatives<double>.ComputeGradient(network, inputs, 1);

        // Assert: Verify chain rule produces correct result
        Assert.NotNull(gradients);
        var numericalGradients = ComputeNumericalGradient(network, inputs, 1);
        Assert.Equal(numericalGradients[0, 0], gradients[0, 0], Tolerance);
    }

    #endregion

    #region Helper Methods

    private static NeuralNetworkBase<double> CreateNetwork(int[] layerSizes, bool useReLU)
    {
        // Create layers with the specified activation function
        var layers = new List<ILayer<double>>();
        IActivationFunction<double> activation = useReLU
            ? new ReLUActivation<double>()
            : new IdentityActivation<double>();

        // Create a dense layer for each layer transition
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            layers.Add(new DenseLayer<double>(layerSizes[i], layerSizes[i + 1], activation));
        }

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: layerSizes[0],
            outputSize: layerSizes[^1],
            layers: layers);

        return new FeedForwardNeuralNetwork<double>(architecture);
    }

    private static NeuralNetworkBase<double> CreateNetworkWithActivation(
        int[] layerSizes,
        IActivationFunction<double> activation)
    {
        // Create layers with the specified activation function
        var layers = new List<ILayer<double>>();

        // Create a dense layer for each layer transition with the specified activation
        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            layers.Add(new DenseLayer<double>(layerSizes[i], layerSizes[i + 1], activation));
        }

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: layerSizes[0],
            outputSize: layerSizes[^1],
            layers: layers);

        return new FeedForwardNeuralNetwork<double>(architecture);
    }

    private static NeuralNetworkBase<double> CreateMultiActivationNetwork(
        int[] layerSizes,
        IActivationFunction<double>[] activations)
    {
        if (activations.Length != layerSizes.Length - 1)
        {
            throw new ArgumentException("Must have one activation per layer transition");
        }

        // Create layers with different activation functions for each layer transition
        var layers = new List<ILayer<double>>();

        for (int i = 0; i < layerSizes.Length - 1; i++)
        {
            layers.Add(new DenseLayer<double>(layerSizes[i], layerSizes[i + 1], activations[i]));
        }

        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: layerSizes[0],
            outputSize: layerSizes[^1],
            layers: layers);

        return new FeedForwardNeuralNetwork<double>(architecture);
    }

    private static NeuralNetworkBase<float> CreateNetworkFloat(int[] layerSizes)
    {
        // Use NeuralNetworkArchitecture to create the network
        var inputSize = layerSizes[0];
        var outputSize = layerSizes[^1];

        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: inputSize,
            outputSize: outputSize);

        return new FeedForwardNeuralNetwork<float>(architecture);
    }

    private static double[,] ComputeNumericalGradient(
        NeuralNetworkBase<double> network,
        double[] inputs,
        int outputDim,
        double epsilon = 1e-5)
    {
        var gradients = new double[outputDim, inputs.Length];
        var perturbed = (double[])inputs.Clone();

        for (int i = 0; i < inputs.Length; i++)
        {
            double original = perturbed[i];

            perturbed[i] = original + epsilon;
            var outputPlus = EvaluateNetwork(network, perturbed, outputDim);

            perturbed[i] = original - epsilon;
            var outputMinus = EvaluateNetwork(network, perturbed, outputDim);

            for (int j = 0; j < outputDim; j++)
            {
                gradients[j, i] = (outputPlus[j] - outputMinus[j]) / (2 * epsilon);
            }

            perturbed[i] = original;
        }

        return gradients;
    }

    private static float[,] ComputeNumericalGradientFloat(
        NeuralNetworkBase<float> network,
        float[] inputs,
        int outputDim,
        float epsilon = 1e-4f)
    {
        var gradients = new float[outputDim, inputs.Length];
        var perturbed = (float[])inputs.Clone();

        for (int i = 0; i < inputs.Length; i++)
        {
            float original = perturbed[i];

            perturbed[i] = original + epsilon;
            var outputPlus = EvaluateNetworkFloat(network, perturbed, outputDim);

            perturbed[i] = original - epsilon;
            var outputMinus = EvaluateNetworkFloat(network, perturbed, outputDim);

            for (int j = 0; j < outputDim; j++)
            {
                gradients[j, i] = (outputPlus[j] - outputMinus[j]) / (2 * epsilon);
            }

            perturbed[i] = original;
        }

        return gradients;
    }

    private static double[,] ComputeNumericalHessian(
        NeuralNetworkBase<double> network,
        double[] inputs,
        int outputIndex,
        double epsilon = 1e-4)
    {
        int n = inputs.Length;
        var hessian = new double[n, n];
        var perturbed = (double[])inputs.Clone();

        // Compute diagonal elements
        for (int i = 0; i < n; i++)
        {
            double original = perturbed[i];

            perturbed[i] = original + epsilon;
            var fPlus = EvaluateNetwork(network, perturbed, outputIndex + 1)[outputIndex];

            perturbed[i] = original - epsilon;
            var fMinus = EvaluateNetwork(network, perturbed, outputIndex + 1)[outputIndex];

            perturbed[i] = original;
            var f0 = EvaluateNetwork(network, perturbed, outputIndex + 1)[outputIndex];

            hessian[i, i] = (fPlus - 2 * f0 + fMinus) / (epsilon * epsilon);
        }

        // Compute off-diagonal elements
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double originalI = perturbed[i];
                double originalJ = perturbed[j];

                perturbed[i] = originalI + epsilon;
                perturbed[j] = originalJ + epsilon;
                var fPP = EvaluateNetwork(network, perturbed, outputIndex + 1)[outputIndex];

                perturbed[j] = originalJ - epsilon;
                var fPM = EvaluateNetwork(network, perturbed, outputIndex + 1)[outputIndex];

                perturbed[i] = originalI - epsilon;
                perturbed[j] = originalJ + epsilon;
                var fMP = EvaluateNetwork(network, perturbed, outputIndex + 1)[outputIndex];

                perturbed[j] = originalJ - epsilon;
                var fMM = EvaluateNetwork(network, perturbed, outputIndex + 1)[outputIndex];

                double cross = (fPP - fPM - fMP + fMM) / (4 * epsilon * epsilon);
                hessian[i, j] = cross;
                hessian[j, i] = cross;

                perturbed[i] = originalI;
                perturbed[j] = originalJ;
            }
        }

        return hessian;
    }

    private static double[] EvaluateNetwork(
        NeuralNetworkBase<double> network,
        double[] inputs,
        int outputDim)
    {
        // Create 1D unbatched tensor to match NeuralNetworkArchitecture expectations
        var inputTensor = new Tensor<double>(new[] { inputs.Length });
        for (int i = 0; i < inputs.Length; i++)
        {
            inputTensor[i] = inputs[i];
        }

        var outputTensor = network.Predict(inputTensor);
        var output = new double[outputDim];

        // Handle 1D output [outputDim] or 2D batched output [1, outputDim]
        if (outputTensor.Rank == 1)
        {
            for (int i = 0; i < outputDim && i < outputTensor.Shape[0]; i++)
            {
                output[i] = outputTensor[i];
            }
        }
        else if (outputTensor.Rank == 2 && outputTensor.Shape[0] == 1)
        {
            for (int i = 0; i < outputDim && i < outputTensor.Shape[1]; i++)
            {
                output[i] = outputTensor[0, i];
            }
        }

        return output;
    }

    private static float[] EvaluateNetworkFloat(
        NeuralNetworkBase<float> network,
        float[] inputs,
        int outputDim)
    {
        // Create 1D unbatched tensor to match NeuralNetworkArchitecture expectations
        var inputTensor = new Tensor<float>(new[] { inputs.Length });
        for (int i = 0; i < inputs.Length; i++)
        {
            inputTensor[i] = inputs[i];
        }

        var outputTensor = network.Predict(inputTensor);
        var output = new float[outputDim];

        // Handle 1D output [outputDim] or 2D batched output [1, outputDim]
        if (outputTensor.Rank == 1)
        {
            for (int i = 0; i < outputDim && i < outputTensor.Shape[0]; i++)
            {
                output[i] = outputTensor[i];
            }
        }
        else if (outputTensor.Rank == 2 && outputTensor.Shape[0] == 1)
        {
            for (int i = 0; i < outputDim && i < outputTensor.Shape[1]; i++)
            {
                output[i] = outputTensor[0, i];
            }
        }

        return output;
    }

    #endregion
}
