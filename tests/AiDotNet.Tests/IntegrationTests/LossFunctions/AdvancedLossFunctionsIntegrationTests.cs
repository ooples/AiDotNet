using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.LossFunctions;

/// <summary>
/// Integration tests for advanced loss function classes.
/// Tests CTCLoss, NoiseContrastiveEstimationLoss, PerceptualLoss, QuantumLoss, and RotationPredictionLoss.
/// </summary>
public class AdvancedLossFunctionsIntegrationTests
{
    private const double Tolerance = 1e-6;

    #region CTCLoss Tests

    [Fact]
    public void CTCLoss_BasicLossComputation_ReturnsPositiveLoss()
    {
        // Arrange
        var ctcLoss = new CTCLoss<double>(blankIndex: 0, inputsAreLogProbs: true);

        // Create log probabilities tensor [batch=1, time=5, classes=4]
        // Classes: 0=blank, 1='a', 2='b', 3='c'
        var logProbs = new Tensor<double>(new[] { 1, 5, 4 });

        // Fill with log probabilities (uniform-ish distribution)
        for (int t = 0; t < 5; t++)
        {
            for (int c = 0; c < 4; c++)
            {
                logProbs[0, t, c] = Math.Log(0.25); // Equal probabilities
            }
        }

        // Target: "ab" (labels 1, 2)
        int[][] targets = { new[] { 1, 2 } };
        int[] inputLengths = { 5 };
        int[] targetLengths = { 2 };

        // Act
        var loss = ctcLoss.CalculateLoss(logProbs, targets, inputLengths, targetLengths);

        // Assert
        Assert.True(loss > 0, "CTC loss should be positive for non-trivial sequences");
    }

    [Fact]
    public void CTCLoss_HighProbabilityPath_ReturnsLowerLoss()
    {
        // Arrange
        var ctcLoss = new CTCLoss<double>(blankIndex: 0, inputsAreLogProbs: true);

        // Create two scenarios: one with high probability for target, one with low
        var logProbsHigh = new Tensor<double>(new[] { 1, 5, 3 });
        var logProbsLow = new Tensor<double>(new[] { 1, 5, 3 });

        // High probability path for target "a" (label 1): blank-a-blank-a-blank
        // t=0: high prob for blank (0)
        logProbsHigh[0, 0, 0] = Math.Log(0.8);
        logProbsHigh[0, 0, 1] = Math.Log(0.1);
        logProbsHigh[0, 0, 2] = Math.Log(0.1);
        // t=1: high prob for 'a' (1)
        logProbsHigh[0, 1, 0] = Math.Log(0.1);
        logProbsHigh[0, 1, 1] = Math.Log(0.8);
        logProbsHigh[0, 1, 2] = Math.Log(0.1);
        // t=2-4: continue pattern
        for (int t = 2; t < 5; t++)
        {
            logProbsHigh[0, t, 0] = Math.Log(0.7);
            logProbsHigh[0, t, 1] = Math.Log(0.2);
            logProbsHigh[0, t, 2] = Math.Log(0.1);
        }

        // Low probability path - wrong class has high probability
        for (int t = 0; t < 5; t++)
        {
            logProbsLow[0, t, 0] = Math.Log(0.1);
            logProbsLow[0, t, 1] = Math.Log(0.1);
            logProbsLow[0, t, 2] = Math.Log(0.8); // Wrong class
        }

        int[][] targets = { new[] { 1 } };
        int[] inputLengths = { 5 };
        int[] targetLengths = { 1 };

        // Act
        var lossHigh = ctcLoss.CalculateLoss(logProbsHigh, targets, inputLengths, targetLengths);
        var lossLow = ctcLoss.CalculateLoss(logProbsLow, targets, inputLengths, targetLengths);

        // Assert
        Assert.True(lossHigh < lossLow, "High probability path should have lower loss");
    }

    [Fact]
    public void CTCLoss_BlankTokenHandling_WorksCorrectly()
    {
        // Arrange
        var ctcLoss = new CTCLoss<double>(blankIndex: 0, inputsAreLogProbs: true);

        // Sequence with blanks between repeated characters
        var logProbs = new Tensor<double>(new[] { 1, 6, 3 });

        // Target: "aa" requires blank between the two 'a's
        // Pattern should be: a-blank-a or blank-a-blank-a-blank, etc.
        for (int t = 0; t < 6; t++)
        {
            logProbs[0, t, 0] = Math.Log(0.33); // blank
            logProbs[0, t, 1] = Math.Log(0.34); // 'a'
            logProbs[0, t, 2] = Math.Log(0.33); // 'b'
        }

        int[][] targets = { new[] { 1, 1 } }; // "aa"
        int[] inputLengths = { 6 };
        int[] targetLengths = { 2 };

        // Act
        var loss = ctcLoss.CalculateLoss(logProbs, targets, inputLengths, targetLengths);

        // Assert
        Assert.True(loss > 0, "CTC loss should be positive");
        Assert.False(double.IsNaN(loss), "CTC loss should not be NaN");
        Assert.False(double.IsInfinity(loss), "CTC loss should not be infinity");
    }

    [Fact]
    public void CTCLoss_Gradient_ReturnsCorrectShape()
    {
        // Arrange
        var ctcLoss = new CTCLoss<double>(blankIndex: 0, inputsAreLogProbs: true);

        var logProbs = new Tensor<double>(new[] { 2, 4, 3 });
        for (int b = 0; b < 2; b++)
        {
            for (int t = 0; t < 4; t++)
            {
                for (int c = 0; c < 3; c++)
                {
                    logProbs[b, t, c] = Math.Log(0.33);
                }
            }
        }

        int[][] targets = { new[] { 1 }, new[] { 2 } };
        int[] inputLengths = { 4, 4 };
        int[] targetLengths = { 1, 1 };

        // Act
        var gradient = ctcLoss.CalculateGradient(logProbs, targets, inputLengths, targetLengths);

        // Assert
        Assert.Equal(logProbs.Shape.Length, gradient.Shape.Length);
        Assert.Equal(logProbs.Shape[0], gradient.Shape[0]); // batch
        Assert.Equal(logProbs.Shape[1], gradient.Shape[1]); // time
        Assert.Equal(logProbs.Shape[2], gradient.Shape[2]); // classes
    }

    [Fact]
    public void CTCLoss_InvalidInputs_ThrowsArgumentException()
    {
        // Arrange
        var ctcLoss = new CTCLoss<double>(blankIndex: 0);
        var logProbs = new Tensor<double>(new[] { 1, 5, 4 });

        // Target length exceeds input length
        int[][] targets = { new[] { 1, 2, 3, 4, 5, 6 } };
        int[] inputLengths = { 5 };
        int[] targetLengths = { 6 };

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            ctcLoss.CalculateLoss(logProbs, targets, inputLengths, targetLengths));
    }

    [Fact]
    public void CTCLoss_NegativeBlankIndex_ThrowsArgumentOutOfRangeException()
    {
        // Act & Assert
        Assert.Throws<ArgumentOutOfRangeException>(() => new CTCLoss<double>(blankIndex: -1));
    }

    [Fact]
    public void CTCLossAdapter_BasicUsage_ReturnsPositiveLoss()
    {
        // Arrange
        int numClasses = 4;
        var adapter = new CTCLossAdapter<double>(numClasses, blankIndex: 0);

        // Create flattened log probabilities [batch=1, time=3, classes=4] = 12 elements
        int batchSize = 1;
        int timeSteps = 3;
        var predicted = new Vector<double>(batchSize * timeSteps * numClasses);
        for (int i = 0; i < predicted.Length; i++)
        {
            predicted[i] = Math.Log(0.25);
        }

        // Actual format: [batchSize, targetLength, ...labels]
        // For single batch with target "a" (label 1)
        var actual = new Vector<double>(new double[] { 1, 1, 1 }); // batchSize=1, targetLen=1, label=1

        // Act
        var loss = adapter.CalculateLoss(predicted, actual);

        // Assert
        Assert.True(loss > 0, "CTCLossAdapter should return positive loss");
    }

    #endregion

    #region NoiseContrastiveEstimationLoss Tests

    [Fact]
    public void NCELoss_BasicComputation_ReturnsPositiveLoss()
    {
        // Arrange
        int numNoiseSamples = 5;
        var nceLoss = new NoiseContrastiveEstimationLoss<double>(numNoiseSamples);

        // Target logits (scores for true samples)
        var targetLogits = new Vector<double>(new[] { 2.0, 1.5, 3.0 });

        // Noise logits (scores for noise samples)
        var noiseLogits = new Matrix<double>(3, numNoiseSamples);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < numNoiseSamples; j++)
            {
                noiseLogits[i, j] = -1.0; // Low scores for noise
            }
        }

        // Act
        var loss = nceLoss.Calculate(targetLogits, noiseLogits);

        // Assert
        Assert.True(loss > 0, "NCE loss should be positive");
        Assert.False(double.IsNaN(loss), "NCE loss should not be NaN");
    }

    [Fact]
    public void NCELoss_HighTargetLowNoise_ReturnsLowLoss()
    {
        // Arrange
        int numNoiseSamples = 5;
        var nceLoss = new NoiseContrastiveEstimationLoss<double>(numNoiseSamples);

        // High scores for targets, low scores for noise
        var targetLogitsHigh = new Vector<double>(new[] { 5.0, 5.0, 5.0 });
        var noiseLogitsLow = new Matrix<double>(3, numNoiseSamples);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < numNoiseSamples; j++)
            {
                noiseLogitsLow[i, j] = -5.0;
            }
        }

        // Low scores for targets, high scores for noise
        var targetLogitsLow = new Vector<double>(new[] { -5.0, -5.0, -5.0 });
        var noiseLogitsHigh = new Matrix<double>(3, numNoiseSamples);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < numNoiseSamples; j++)
            {
                noiseLogitsHigh[i, j] = 5.0;
            }
        }

        // Act
        var lossGoodCase = nceLoss.Calculate(targetLogitsHigh, noiseLogitsLow);
        var lossBadCase = nceLoss.Calculate(targetLogitsLow, noiseLogitsHigh);

        // Assert
        Assert.True(lossGoodCase < lossBadCase,
            "High target / low noise should have lower loss than low target / high noise");
    }

    [Fact]
    public void NCELoss_Gradient_ReturnsCorrectShapes()
    {
        // Arrange
        int numNoiseSamples = 5;
        var nceLoss = new NoiseContrastiveEstimationLoss<double>(numNoiseSamples);

        var targetLogits = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var noiseLogits = new Matrix<double>(3, numNoiseSamples);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < numNoiseSamples; j++)
            {
                noiseLogits[i, j] = 0.5;
            }
        }

        // Act
        var (targetGradient, noiseGradient) = nceLoss.CalculateDerivative(targetLogits, noiseLogits);

        // Assert
        Assert.Equal(targetLogits.Length, targetGradient.Length);
        Assert.Equal(noiseLogits.Rows, noiseGradient.Rows);
        Assert.Equal(noiseLogits.Columns, noiseGradient.Columns);
    }

    [Fact]
    public void NCELoss_VaryingNoiseSamples_WorksCorrectly()
    {
        // Arrange & Act & Assert
        foreach (int numNoiseSamples in new[] { 1, 5, 10, 20 })
        {
            var nceLoss = new NoiseContrastiveEstimationLoss<double>(numNoiseSamples);
            var targetLogits = new Vector<double>(new[] { 1.0, 2.0 });
            var noiseLogits = new Matrix<double>(2, numNoiseSamples);

            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < numNoiseSamples; j++)
                {
                    noiseLogits[i, j] = 0.0;
                }
            }

            var loss = nceLoss.Calculate(targetLogits, noiseLogits);

            Assert.True(loss >= 0, $"Loss should be non-negative for {numNoiseSamples} noise samples");
            Assert.False(double.IsNaN(loss), $"Loss should not be NaN for {numNoiseSamples} noise samples");
        }
    }

    [Fact]
    public void NCELoss_StandardInterface_ThrowsNotSupportedException()
    {
        // Arrange
        var nceLoss = new NoiseContrastiveEstimationLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 2.0 });
        var actual = new Vector<double>(new[] { 1.0, 0.0 });

        // Act & Assert
        Assert.Throws<NotSupportedException>(() => nceLoss.CalculateLoss(predicted, actual));
        Assert.Throws<NotSupportedException>(() => nceLoss.CalculateDerivative(predicted, actual));
    }

    [Fact]
    public void NCELoss_DimensionMismatch_ThrowsArgumentException()
    {
        // Arrange
        var nceLoss = new NoiseContrastiveEstimationLoss<double>(5);
        var targetLogits = new Vector<double>(new[] { 1.0, 2.0, 3.0 });
        var noiseLogits = new Matrix<double>(2, 5); // Wrong number of rows

        // Act & Assert
        Assert.Throws<ArgumentException>(() => nceLoss.Calculate(targetLogits, noiseLogits));
    }

    #endregion

    #region PerceptualLoss Tests

    [Fact]
    public void PerceptualLoss_IdenticalImages_ReturnsZeroLoss()
    {
        // Arrange
        // Simple feature extractor that returns the image as a single feature vector
        Func<Matrix<double>, Vector<Vector<double>>> featureExtractor = (img) =>
        {
            var features = new Vector<Vector<double>>(1);
            var flatFeatures = new Vector<double>(img.Rows * img.Columns);
            int idx = 0;
            for (int i = 0; i < img.Rows; i++)
            {
                for (int j = 0; j < img.Columns; j++)
                {
                    flatFeatures[idx++] = img[i, j];
                }
            }
            features[0] = flatFeatures;
            return features;
        };

        var weights = new Vector<double>(new[] { 1.0 });
        var perceptualLoss = new PerceptualLoss<double>(featureExtractor, weights);

        var image = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                image[i, j] = i * 4 + j;
            }
        }

        // Act
        var loss = perceptualLoss.Calculate(image, image);

        // Assert
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void PerceptualLoss_DifferentImages_ReturnsPositiveLoss()
    {
        // Arrange
        Func<Matrix<double>, Vector<Vector<double>>> featureExtractor = (img) =>
        {
            var features = new Vector<Vector<double>>(1);
            var flatFeatures = new Vector<double>(img.Rows * img.Columns);
            int idx = 0;
            for (int i = 0; i < img.Rows; i++)
            {
                for (int j = 0; j < img.Columns; j++)
                {
                    flatFeatures[idx++] = img[i, j];
                }
            }
            features[0] = flatFeatures;
            return features;
        };

        var weights = new Vector<double>(new[] { 1.0 });
        var perceptualLoss = new PerceptualLoss<double>(featureExtractor, weights);

        var generated = new Matrix<double>(4, 4);
        var target = new Matrix<double>(4, 4);
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                generated[i, j] = 0.0;
                target[i, j] = 1.0;
            }
        }

        // Act
        var loss = perceptualLoss.Calculate(generated, target);

        // Assert
        Assert.True(loss > 0, "Perceptual loss should be positive for different images");
    }

    [Fact]
    public void PerceptualLoss_MultipleFeatureLayers_WeightsAppliedCorrectly()
    {
        // Arrange
        // Feature extractor that returns two layers of features
        Func<Matrix<double>, Vector<Vector<double>>> featureExtractor = (img) =>
        {
            var features = new Vector<Vector<double>>(2);
            // Layer 1: sum of all pixels
            features[0] = new Vector<double>(new[] { img.Sum() });
            // Layer 2: average of all pixels
            features[1] = new Vector<double>(new[] { img.Sum() / (img.Rows * img.Columns) });
            return features;
        };

        var weightsEqual = new Vector<double>(new[] { 1.0, 1.0 });
        var weightsFirstOnly = new Vector<double>(new[] { 1.0, 0.0 });

        var perceptualLossEqual = new PerceptualLoss<double>(featureExtractor, weightsEqual);
        var perceptualLossFirstOnly = new PerceptualLoss<double>(featureExtractor, weightsFirstOnly);

        var generated = new Matrix<double>(2, 2);
        var target = new Matrix<double>(2, 2);
        generated[0, 0] = 0; generated[0, 1] = 0; generated[1, 0] = 0; generated[1, 1] = 0;
        target[0, 0] = 1; target[0, 1] = 1; target[1, 0] = 1; target[1, 1] = 1;

        // Act
        var lossEqual = perceptualLossEqual.Calculate(generated, target);
        var lossFirstOnly = perceptualLossFirstOnly.Calculate(generated, target);

        // Assert
        Assert.True(lossEqual > lossFirstOnly,
            "Equal weights should give higher loss than first-only weights");
    }

    [Fact]
    public void PerceptualLoss_StandardInterface_ThrowsNotSupportedException()
    {
        // Arrange
        Func<Matrix<double>, Vector<Vector<double>>> featureExtractor = (img) =>
            new Vector<Vector<double>>(0);

        var perceptualLoss = new PerceptualLoss<double>(featureExtractor, new Vector<double>(0));
        var predicted = new Vector<double>(new[] { 1.0 });
        var actual = new Vector<double>(new[] { 1.0 });

        // Act & Assert
        Assert.Throws<NotSupportedException>(() => perceptualLoss.CalculateLoss(predicted, actual));
        Assert.Throws<NotSupportedException>(() => perceptualLoss.CalculateDerivative(predicted, actual));
    }

    [Fact]
    public void PerceptualLoss_NullFeatureExtractor_ThrowsArgumentNullException()
    {
        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            new PerceptualLoss<double>(null!, new Vector<double>(1)));
    }

    #endregion

    #region QuantumLoss Tests

    [Fact]
    public void QuantumLoss_IdenticalStates_ReturnsZeroLoss()
    {
        // Arrange
        var quantumLoss = new QuantumLoss<double>();

        // Quantum state |0> = (1, 0) represented as [real, imaginary]
        var state = new Vector<double>(new[] { 1.0, 0.0 });

        // Act
        var loss = quantumLoss.CalculateLoss(state, state);

        // Assert
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void QuantumLoss_OrthogonalStates_ReturnsPositiveLoss()
    {
        // Arrange
        var quantumLoss = new QuantumLoss<double>();

        // Two orthogonal quantum states in a 2D Hilbert space
        // Each complex number is stored as [real, imaginary] pairs
        // |0⟩ = [1+0i, 0+0i] = first basis vector
        // |1⟩ = [0+0i, 1+0i] = second basis vector
        // These are orthogonal: ⟨0|1⟩ = (1)(0) + (0)(1) = 0
        var state0 = new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0 }); // |0⟩
        var state1 = new Vector<double>(new[] { 0.0, 0.0, 1.0, 0.0 }); // |1⟩

        // Act
        var loss = quantumLoss.CalculateLoss(state0, state1);

        // Assert
        // Orthogonal states have zero inner product, so fidelity = 0 and loss = 1
        Assert.True(loss > 0, "Orthogonal states should have positive loss");
        Assert.Equal(1.0, loss, Tolerance); // Loss should be exactly 1 for orthogonal states
        Assert.False(double.IsNaN(loss), "Loss should not be NaN");
    }

    [Fact]
    public void QuantumLoss_SuperpositionStates_ReturnsCorrectLoss()
    {
        // Arrange
        var quantumLoss = new QuantumLoss<double>();

        // |+> = (1/sqrt(2), 1/sqrt(2)) = normalized superposition
        double invSqrt2 = 1.0 / Math.Sqrt(2);
        var plusState = new Vector<double>(new[] { invSqrt2, 0.0, invSqrt2, 0.0 });

        // |0> = (1, 0, 0, 0) for two-qubit comparison
        var zeroState = new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0 });

        // Act
        var loss = quantumLoss.CalculateLoss(plusState, zeroState);

        // Assert
        Assert.True(loss > 0, "Different states should have positive loss");
        Assert.True(loss < 1, "Non-orthogonal states should have loss less than 1");
    }

    [Fact]
    public void QuantumLoss_Gradient_ReturnsCorrectShape()
    {
        // Arrange
        var quantumLoss = new QuantumLoss<double>();

        // Two complex numbers = 4 real values
        var predicted = new Vector<double>(new[] { 0.6, 0.0, 0.8, 0.0 });
        var expected = new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0 });

        // Act
        var gradient = quantumLoss.CalculateDerivative(predicted, expected);

        // Assert
        Assert.Equal(predicted.Length, gradient.Length);
    }

    [Fact]
    public void QuantumLoss_ComplexAmplitudes_HandledCorrectly()
    {
        // Arrange
        var quantumLoss = new QuantumLoss<double>();

        // State with complex amplitudes: (1/sqrt(2), i/sqrt(2))
        double invSqrt2 = 1.0 / Math.Sqrt(2);
        // [real1, imag1, real2, imag2] = [1/sqrt(2), 0, 0, 1/sqrt(2)]
        var complexState = new Vector<double>(new[] { invSqrt2, 0.0, 0.0, invSqrt2 });

        // Act
        var loss = quantumLoss.CalculateLoss(complexState, complexState);

        // Assert
        Assert.Equal(0.0, loss, Tolerance);
    }

    [Fact]
    public void QuantumLoss_DimensionMismatch_ThrowsArgumentException()
    {
        // Arrange
        var quantumLoss = new QuantumLoss<double>();
        var predicted = new Vector<double>(new[] { 1.0, 0.0 });
        var expected = new Vector<double>(new[] { 1.0, 0.0, 0.0, 0.0 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() => quantumLoss.CalculateLoss(predicted, expected));
    }

    #endregion

    #region RotationPredictionLoss Tests

    [Fact]
    public void RotationPredictionLoss_TensorInput_Creates4RotationsPerImage()
    {
        // Arrange
        var rotationLoss = new RotationPredictionLoss<double>();

        // Create a simple 2x2 grayscale image batch [N=2, H=2, W=2, C=1] (4D input)
        var input = new Tensor<double>(new[] { 2, 2, 2, 1 });
        input[0, 0, 0, 0] = 1; input[0, 0, 1, 0] = 2;
        input[0, 1, 0, 0] = 3; input[0, 1, 1, 0] = 4;
        input[1, 0, 0, 0] = 5; input[1, 0, 1, 0] = 6;
        input[1, 1, 0, 0] = 7; input[1, 1, 1, 0] = 8;

        // Act
        var (augmentedX, augmentedY) = rotationLoss.CreateTask<Tensor<double>, Tensor<double>>(input);

        // Assert
        Assert.Equal(8, augmentedX.Shape[0]); // 2 images * 4 rotations
        Assert.Equal(2, augmentedX.Shape[1]); // Height preserved
        Assert.Equal(2, augmentedX.Shape[2]); // Width preserved
        Assert.Equal(8, augmentedY.Shape[0]); // Same batch size
        Assert.Equal(4, augmentedY.Shape[1]); // 4-class one-hot
    }

    [Fact]
    public void RotationPredictionLoss_MatrixInput_Creates4RotationsPerImage()
    {
        // Arrange
        var rotationLoss = new RotationPredictionLoss<double>();

        // Create a batch of 3 flattened 4x4 images (16 pixels each)
        var input = new Matrix<double>(3, 16);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                input[i, j] = i * 16 + j;
            }
        }

        // Act
        var (augmentedX, augmentedY) = rotationLoss.CreateTask<Matrix<double>, Matrix<double>>(input);

        // Assert
        Assert.Equal(12, augmentedX.Rows); // 3 images * 4 rotations
        Assert.Equal(16, augmentedX.Columns); // Flattened size preserved
        Assert.Equal(12, augmentedY.Rows); // Same batch size
        Assert.Equal(4, augmentedY.Columns); // 4-class one-hot
    }

    [Fact]
    public void RotationPredictionLoss_0DegreeRotation_PreservesImage()
    {
        // Arrange
        var rotationLoss = new RotationPredictionLoss<double>();

        // Create a simple 2x2 grayscale image batch [N=1, H=2, W=2] (3D input)
        var input = new Tensor<double>(new[] { 1, 2, 2 });
        input[0, 0, 0] = 1; input[0, 0, 1] = 2;
        input[0, 1, 0] = 3; input[0, 1, 1] = 4;

        // Act
        var (augmentedX, augmentedY) = rotationLoss.CreateTask<Tensor<double>, Tensor<double>>(input);

        // Assert - First rotation (0 degrees) should preserve the image
        // Output has 4 dimensions [N, H, W, C] even for grayscale input
        Assert.Equal(1.0, augmentedX[0, 0, 0, 0], Tolerance);
        Assert.Equal(2.0, augmentedX[0, 0, 1, 0], Tolerance);
        Assert.Equal(3.0, augmentedX[0, 1, 0, 0], Tolerance);
        Assert.Equal(4.0, augmentedX[0, 1, 1, 0], Tolerance);
    }

    [Fact]
    public void RotationPredictionLoss_180DegreeRotation_FlipsImage()
    {
        // Arrange
        var rotationLoss = new RotationPredictionLoss<double>();

        // Create a simple 2x2 grayscale image batch [N=1, H=2, W=2] (3D input)
        var input = new Tensor<double>(new[] { 1, 2, 2 });
        input[0, 0, 0] = 1; input[0, 0, 1] = 2;
        input[0, 1, 0] = 3; input[0, 1, 1] = 4;

        // Act
        var (augmentedX, augmentedY) = rotationLoss.CreateTask<Tensor<double>, Tensor<double>>(input);

        // Assert - Third rotation (180 degrees, index 2) should flip the image
        // Original: [1 2]   Rotated 180: [4 3]
        //           [3 4]                 [2 1]
        Assert.Equal(4.0, augmentedX[2, 0, 0, 0], Tolerance);
        Assert.Equal(3.0, augmentedX[2, 0, 1, 0], Tolerance);
        Assert.Equal(2.0, augmentedX[2, 1, 0, 0], Tolerance);
        Assert.Equal(1.0, augmentedX[2, 1, 1, 0], Tolerance);
    }

    [Fact]
    public void RotationPredictionLoss_OneHotLabels_CorrectlyEncoded()
    {
        // Arrange
        var rotationLoss = new RotationPredictionLoss<double>();

        // Create a simple 2x2 grayscale image batch [N=1, H=2, W=2] (3D input)
        var input = new Tensor<double>(new[] { 1, 2, 2 });

        // Act
        var (_, augmentedY) = rotationLoss.CreateTask<Tensor<double>, Tensor<double>>(input);

        // Assert - Check one-hot encoding for all 4 rotations
        // Rotation 0 (0 degrees): [1, 0, 0, 0]
        Assert.Equal(1.0, augmentedY[0, 0], Tolerance);
        Assert.Equal(0.0, augmentedY[0, 1], Tolerance);
        Assert.Equal(0.0, augmentedY[0, 2], Tolerance);
        Assert.Equal(0.0, augmentedY[0, 3], Tolerance);

        // Rotation 1 (90 degrees): [0, 1, 0, 0]
        Assert.Equal(0.0, augmentedY[1, 0], Tolerance);
        Assert.Equal(1.0, augmentedY[1, 1], Tolerance);

        // Rotation 2 (180 degrees): [0, 0, 1, 0]
        Assert.Equal(1.0, augmentedY[2, 2], Tolerance);

        // Rotation 3 (270 degrees): [0, 0, 0, 1]
        Assert.Equal(1.0, augmentedY[3, 3], Tolerance);
    }

    [Fact]
    public void RotationPredictionLoss_NonSquareMatrix_ThrowsArgumentException()
    {
        // Arrange
        var rotationLoss = new RotationPredictionLoss<double>();

        // Non-square flattened size (15 is not a perfect square)
        var input = new Matrix<double>(1, 15);

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            rotationLoss.CreateTask<Matrix<double>, Matrix<double>>(input));
    }

    [Fact]
    public void RotationPredictionLoss_TooFewDimensions_ThrowsArgumentException()
    {
        // Arrange
        var rotationLoss = new RotationPredictionLoss<double>();

        // 2D tensor is not enough (needs at least 3D: [N, H, W])
        var input = new Tensor<double>(new[] { 4, 4 });

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            rotationLoss.CreateTask<Tensor<double>, Tensor<double>>(input));
    }

    [Fact]
    public void RotationPredictionLoss_MatrixToVectorOutput_ReturnsClassIndices()
    {
        // Arrange
        var rotationLoss = new RotationPredictionLoss<double>();

        var input = new Matrix<double>(2, 4); // 2 images of 2x2

        // Act
        var (_, augmentedY) = rotationLoss.CreateTask<Matrix<double>, Vector<double>>(input);

        // Assert - Should return class indices [0, 1, 2, 3, 0, 1, 2, 3]
        Assert.Equal(8, augmentedY.Length); // 2 images * 4 rotations
        Assert.Equal(0.0, augmentedY[0], Tolerance); // First image, 0 degrees
        Assert.Equal(1.0, augmentedY[1], Tolerance); // First image, 90 degrees
        Assert.Equal(2.0, augmentedY[2], Tolerance); // First image, 180 degrees
        Assert.Equal(3.0, augmentedY[3], Tolerance); // First image, 270 degrees
    }

    #endregion

    #region Edge Case Tests

    [Fact]
    public void CTCLoss_SingleCharacterTarget_WorksCorrectly()
    {
        // Arrange
        var ctcLoss = new CTCLoss<double>(blankIndex: 0, inputsAreLogProbs: true);

        var logProbs = new Tensor<double>(new[] { 1, 3, 2 });
        for (int t = 0; t < 3; t++)
        {
            logProbs[0, t, 0] = Math.Log(0.5);
            logProbs[0, t, 1] = Math.Log(0.5);
        }

        int[][] targets = { new[] { 1 } }; // Single character
        int[] inputLengths = { 3 };
        int[] targetLengths = { 1 };

        // Act
        var loss = ctcLoss.CalculateLoss(logProbs, targets, inputLengths, targetLengths);

        // Assert
        Assert.True(loss > 0, "Loss should be positive");
        Assert.False(double.IsNaN(loss), "Loss should not be NaN");
    }

    [Fact]
    public void NCELoss_SingleSample_WorksCorrectly()
    {
        // Arrange
        var nceLoss = new NoiseContrastiveEstimationLoss<double>(3);

        var targetLogits = new Vector<double>(new[] { 1.0 });
        var noiseLogits = new Matrix<double>(1, 3);
        noiseLogits[0, 0] = -1.0;
        noiseLogits[0, 1] = -1.0;
        noiseLogits[0, 2] = -1.0;

        // Act
        var loss = nceLoss.Calculate(targetLogits, noiseLogits);

        // Assert
        Assert.True(loss >= 0, "Loss should be non-negative");
    }

    [Fact]
    public void QuantumLoss_NormalizedState_WorksCorrectly()
    {
        // Arrange
        var quantumLoss = new QuantumLoss<double>();

        // Normalized state: |psi> = (0.6, 0.8)
        var predicted = new Vector<double>(new[] { 0.6, 0.0, 0.8, 0.0 });
        var expected = new Vector<double>(new[] { 0.6, 0.0, 0.8, 0.0 });

        // Act
        var loss = quantumLoss.CalculateLoss(predicted, expected);

        // Assert
        Assert.Equal(0.0, loss, Tolerance);
    }

    #endregion
}
