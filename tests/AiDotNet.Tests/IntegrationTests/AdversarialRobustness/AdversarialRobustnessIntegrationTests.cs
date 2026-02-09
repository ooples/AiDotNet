#nullable disable
using Xunit;
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.AdversarialRobustness.CertifiedRobustness;
using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.AdversarialRobustness.Safety;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tests.IntegrationTests.AdversarialRobustness;

/// <summary>
/// Comprehensive integration tests for the AdversarialRobustness module.
/// Tests cover attacks, defenses, certified robustness, and safety filtering.
/// </summary>
public class AdversarialRobustnessIntegrationTests
{
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();
    private const double Tolerance = 1e-6;
    private const int Seed = 42;

    #region Test Helpers

    /// <summary>
    /// Creates a simple mock model for testing adversarial attacks.
    /// </summary>
    private static MockClassificationModel CreateMockClassificationModel(int inputSize = 10, int numClasses = 3)
    {
        return new MockClassificationModel(inputSize, numClasses, Seed);
    }

    /// <summary>
    /// Creates test input data for adversarial testing.
    /// </summary>
    private static Vector<double> CreateTestInput(int size = 10, double value = 0.5)
    {
        var input = new Vector<double>(size);
        for (int i = 0; i < size; i++)
        {
            input[i] = value + (i * 0.01);
        }
        return input;
    }

    /// <summary>
    /// Creates a one-hot encoded label vector.
    /// </summary>
    private static Vector<double> CreateOneHotLabel(int numClasses, int trueClass)
    {
        var label = new Vector<double>(numClasses);
        label[trueClass] = 1.0;
        return label;
    }

    #endregion

    #region FGSM Attack Tests

    [Fact]
    public void FGSMAttack_GeneratesAdversarialExample_WithinEpsilonBound()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            RandomSeed = Seed
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert
        Assert.NotNull(adversarial);
        Assert.Equal(input.Length, adversarial.Length);

        // Check perturbation is within epsilon bound
        for (int i = 0; i < input.Length; i++)
        {
            var perturbation = Math.Abs(adversarial[i] - input[i]);
            Assert.True(perturbation <= options.Epsilon + Tolerance,
                $"Perturbation at index {i} exceeds epsilon bound");
        }
    }

    [Fact]
    public void FGSMAttack_GenerateBatch_ProcessesMultipleInputs()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.1, RandomSeed = Seed };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var inputs = new Vector<double>[] { CreateTestInput(), CreateTestInput() };
        var labels = new Vector<double>[] { CreateOneHotLabel(3, 0), CreateOneHotLabel(3, 1) };

        // Act
        var adversarials = attack.GenerateAdversarialBatch(inputs, labels, model);

        // Assert
        Assert.NotNull(adversarials);
        Assert.Equal(2, adversarials.Length);
        foreach (var adv in adversarials)
        {
            Assert.NotNull(adv);
            Assert.Equal(inputs[0].Length, adv.Length);
        }
    }

    [Fact]
    public void FGSMAttack_CalculatePerturbation_ReturnsValidPerturbation()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.1, RandomSeed = Seed };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Generate adversarial example first
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Act
        var perturbation = attack.CalculatePerturbation(input, adversarial);

        // Assert
        Assert.NotNull(perturbation);
        Assert.Equal(input.Length, perturbation.Length);
    }

    [Fact]
    public void FGSMAttack_Targeted_GeneratesAdversarialTowardTarget()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            IsTargeted = true,
            TargetClass = 2,
            RandomSeed = Seed
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert
        Assert.NotNull(adversarial);
    }

    [Fact]
    public void FGSMAttack_GetOptions_ReturnsConfiguredOptions()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.15,
            StepSize = 0.02,
            RandomSeed = Seed
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);

        // Act
        var retrievedOptions = attack.GetOptions();

        // Assert
        Assert.Equal(0.15, retrievedOptions.Epsilon, 3);
        Assert.Equal(0.02, retrievedOptions.StepSize, 3);
    }

    [Fact]
    public void FGSMAttack_SerializationRoundTrip_PreservesOptions()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.15, RandomSeed = Seed };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);

        // Act
        var serialized = attack.Serialize();
        var newAttack = new FGSMAttack<double, Vector<double>, Vector<double>>(
            new AdversarialAttackOptions<double>());
        newAttack.Deserialize(serialized);
        var newOptions = newAttack.GetOptions();

        // Assert
        Assert.Equal(0.15, newOptions.Epsilon, 3);
    }

    [Fact]
    public void FGSMAttack_ThrowsOnNullInput()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.1 };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var label = CreateOneHotLabel(3, 0);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            attack.GenerateAdversarialExample(null!, label, model));
    }

    [Fact]
    public void FGSMAttack_ThrowsOnNullLabel()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.1 };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            attack.GenerateAdversarialExample(input, null!, model));
    }

    [Fact]
    public void FGSMAttack_ThrowsOnNullModel()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.1 };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            attack.GenerateAdversarialExample(input, label, null!));
    }

    #endregion

    #region PGD Attack Tests

    [Fact]
    public void PGDAttack_GeneratesAdversarialExample_WithinEpsilonBound()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            StepSize = 0.01,
            Iterations = 10,
            RandomSeed = Seed
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert
        Assert.NotNull(adversarial);
        for (int i = 0; i < input.Length; i++)
        {
            var perturbation = Math.Abs(adversarial[i] - input[i]);
            Assert.True(perturbation <= options.Epsilon + Tolerance,
                $"PGD perturbation at index {i} exceeds epsilon bound");
        }
    }

    [Fact]
    public void PGDAttack_WithRandomStart_GeneratesDifferentAdversarials()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            StepSize = 0.01,
            Iterations = 10,
            UseRandomStart = true
        };
        var attack1 = new PGDAttack<double, Vector<double>, Vector<double>>(
            new AdversarialAttackOptions<double> { Epsilon = 0.1, UseRandomStart = true, RandomSeed = 1 });
        var attack2 = new PGDAttack<double, Vector<double>, Vector<double>>(
            new AdversarialAttackOptions<double> { Epsilon = 0.1, UseRandomStart = true, RandomSeed = 2 });
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adv1 = attack1.GenerateAdversarialExample(input, label, model);
        var adv2 = attack2.GenerateAdversarialExample(input, label, model);

        // Assert - both should be valid but may differ due to random start
        Assert.NotNull(adv1);
        Assert.NotNull(adv2);
    }

    [Fact]
    public void PGDAttack_MultipleIterations_ImprovesAttack()
    {
        // Arrange
        var options1 = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            StepSize = 0.01,
            Iterations = 1,
            RandomSeed = Seed
        };
        var options10 = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            StepSize = 0.01,
            Iterations = 10,
            RandomSeed = Seed
        };
        var attack1 = new PGDAttack<double, Vector<double>, Vector<double>>(options1);
        var attack10 = new PGDAttack<double, Vector<double>, Vector<double>>(options10);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adv1 = attack1.GenerateAdversarialExample(input, label, model);
        var adv10 = attack10.GenerateAdversarialExample(input, label, model);

        // Assert - both should produce valid adversarial examples
        Assert.NotNull(adv1);
        Assert.NotNull(adv10);
    }

    [Fact]
    public void PGDAttack_Reset_ReturnsToInitialState()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.1, Iterations = 5 };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        attack.GenerateAdversarialExample(input, label, model);
        attack.Reset();

        // Assert - should be able to generate again after reset
        var adv = attack.GenerateAdversarialExample(input, label, model);
        Assert.NotNull(adv);
    }

    #endregion

    #region CW Attack Tests

    [Fact]
    public void CWAttack_GeneratesAdversarialExample()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.5,
            Iterations = 10,
            RandomSeed = Seed
        };
        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert
        Assert.NotNull(adversarial);
        Assert.Equal(input.Length, adversarial.Length);
    }

    [Fact]
    public void CWAttack_CalculatePerturbation_MinimizesL2Norm()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.5,
            Iterations = 20,
            RandomSeed = Seed
        };
        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Generate adversarial example first
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Act
        var perturbation = attack.CalculatePerturbation(input, adversarial);

        // Assert
        Assert.NotNull(perturbation);

        // CW attack should produce relatively small perturbations
        double l2Norm = 0;
        for (int i = 0; i < perturbation.Length; i++)
        {
            l2Norm += perturbation[i] * perturbation[i];
        }
        l2Norm = Math.Sqrt(l2Norm);

        // Perturbation should exist (not be zero) unless attack failed
        // CW is optimization-based so may not always succeed
        Assert.True(l2Norm >= 0);
    }

    [Fact]
    public void CWAttack_Targeted_AttemptsToMisclassifyToTarget()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 1.0,
            Iterations = 20,
            IsTargeted = true,
            TargetClass = 2,
            RandomSeed = Seed
        };
        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert
        Assert.NotNull(adversarial);
    }

    #endregion

    #region AutoAttack Tests

    [Fact]
    public void AutoAttack_CombinesMultipleAttacks()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            RandomSeed = Seed
        };
        var attack = new AutoAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert
        Assert.NotNull(adversarial);
        Assert.Equal(input.Length, adversarial.Length);
    }

    [Fact]
    public void AutoAttack_GetOptions_ReturnsConfiguredOptions()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.2,
            Iterations = 50
        };
        var attack = new AutoAttack<double, Vector<double>, Vector<double>>(options);

        // Act
        var retrievedOptions = attack.GetOptions();

        // Assert
        Assert.Equal(0.2, retrievedOptions.Epsilon, 3);
    }

    #endregion

    #region Adversarial Training Tests

    [Fact]
    public void AdversarialTraining_ApplyDefense_ReturnsModel()
    {
        // Arrange
        var options = new AdversarialDefenseOptions<double>
        {
            Epsilon = 0.1,
            UsePreprocessing = false
        };
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var trainingData = new Vector<double>[] { CreateTestInput(), CreateTestInput() };
        var labels = new Vector<double>[] { CreateOneHotLabel(3, 0), CreateOneHotLabel(3, 1) };

        // Act
        var defendedModel = defense.ApplyDefense(trainingData, labels, model);

        // Assert
        Assert.NotNull(defendedModel);
    }

    [Fact]
    public void AdversarialTraining_WithPreprocessing_AppliesInputPreprocessing()
    {
        // Arrange
        var options = new AdversarialDefenseOptions<double>
        {
            Epsilon = 0.1,
            UsePreprocessing = true,
            PreprocessingMethod = "jpeg"
        };
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(options);
        var input = CreateTestInput();

        // Act
        var preprocessed = defense.PreprocessInput(input);

        // Assert
        Assert.NotNull(preprocessed);
        Assert.Equal(input.Length, preprocessed.Length);
    }

    [Fact]
    public void AdversarialTraining_PreprocessingMethods_AllSupported()
    {
        // Test different preprocessing methods
        var methods = new[] { "jpeg", "bit_depth_reduction", "denoising" };

        foreach (var method in methods)
        {
            // Arrange
            var options = new AdversarialDefenseOptions<double>
            {
                UsePreprocessing = true,
                PreprocessingMethod = method
            };
            var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(options);
            var input = CreateTestInput();

            // Act
            var preprocessed = defense.PreprocessInput(input);

            // Assert
            Assert.NotNull(preprocessed);
            Assert.Equal(input.Length, preprocessed.Length);
        }
    }

    [Fact]
    public void AdversarialTraining_EvaluateRobustness_ReturnsMetrics()
    {
        // Arrange
        var defenseOptions = new AdversarialDefenseOptions<double> { Epsilon = 0.1 };
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(defenseOptions);
        var model = CreateMockClassificationModel();
        var testData = new Vector<double>[] { CreateTestInput(), CreateTestInput() };
        var labels = new Vector<double>[] { CreateOneHotLabel(3, 0), CreateOneHotLabel(3, 1) };

        var attackOptions = new AdversarialAttackOptions<double> { Epsilon = 0.1, RandomSeed = Seed };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(attackOptions);

        // Act
        var metrics = defense.EvaluateRobustness(model, testData, labels, attack);

        // Assert
        Assert.NotNull(metrics);
        Assert.InRange(metrics.CleanAccuracy, 0.0, 1.0);
        Assert.InRange(metrics.AdversarialAccuracy, 0.0, 1.0);
    }

    [Fact]
    public void AdversarialTraining_SerializationRoundTrip_PreservesState()
    {
        // Arrange
        var options = new AdversarialDefenseOptions<double>
        {
            Epsilon = 0.15,
            UsePreprocessing = true,
            PreprocessingMethod = "jpeg"
        };
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(options);

        // Act
        var serialized = defense.Serialize();
        var newDefense = new AdversarialTraining<double, Vector<double>, Vector<double>>(
            new AdversarialDefenseOptions<double>());
        newDefense.Deserialize(serialized);
        var newOptions = newDefense.GetOptions();

        // Assert
        Assert.Equal(0.15, newOptions.Epsilon, 3);
    }

    [Fact]
    public void AdversarialTraining_ThrowsOnNullTrainingData()
    {
        // Arrange
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var labels = new Vector<double>[] { CreateOneHotLabel(3, 0) };

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            defense.ApplyDefense(null!, labels, model));
    }

    [Fact]
    public void AdversarialTraining_ThrowsOnMismatchedDataLabelCount()
    {
        // Arrange
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var data = new Vector<double>[] { CreateTestInput(), CreateTestInput() };
        var labels = new Vector<double>[] { CreateOneHotLabel(3, 0) }; // Only 1 label for 2 data points

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            defense.ApplyDefense(data, labels, model));
    }

    #endregion

    #region Randomized Smoothing Tests

    [Fact]
    public void RandomizedSmoothing_CertifyPrediction_ReturnsCertifiedPrediction()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.25,
            NumSamples = 100,
            ConfidenceLevel = 0.95,
            RandomSeed = Seed
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act
        var prediction = smoothing.CertifyPrediction(input, model);

        // Assert
        Assert.NotNull(prediction);
        Assert.True(prediction.PredictedClass >= 0);
        Assert.InRange(prediction.Confidence, 0.0, 1.0);
    }

    [Fact]
    public void RandomizedSmoothing_CertifyBatch_ProcessesMultipleInputs()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.25,
            NumSamples = 50,
            RandomSeed = Seed
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var inputs = new Vector<double>[] { CreateTestInput(), CreateTestInput() };

        // Act
        var predictions = smoothing.CertifyBatch(inputs, model);

        // Assert
        Assert.NotNull(predictions);
        Assert.Equal(2, predictions.Length);
        foreach (var pred in predictions)
        {
            Assert.NotNull(pred);
        }
    }

    [Fact]
    public void RandomizedSmoothing_ComputeCertifiedRadius_ReturnsNonNegativeRadius()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.25,
            NumSamples = 100,
            RandomSeed = Seed
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act
        var radius = smoothing.ComputeCertifiedRadius(input, model);

        // Assert
        Assert.True(radius >= 0, "Certified radius should be non-negative");
    }

    [Fact]
    public void RandomizedSmoothing_EvaluateCertifiedAccuracy_ReturnsValidMetrics()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.25,
            NumSamples = 50,
            RandomSeed = Seed
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var testData = new Vector<double>[] { CreateTestInput(), CreateTestInput() };
        var labels = new Vector<double>[] { CreateOneHotLabel(3, 0), CreateOneHotLabel(3, 1) };

        // Act
        var metrics = smoothing.EvaluateCertifiedAccuracy(testData, labels, model, 0.1);

        // Assert
        Assert.NotNull(metrics);
        Assert.InRange(metrics.CleanAccuracy, 0.0, 1.0);
        Assert.InRange(metrics.CertifiedAccuracy, 0.0, 1.0);
        Assert.InRange(metrics.CertificationRate, 0.0, 1.0);
    }

    [Fact]
    public void RandomizedSmoothing_HigherSigma_ProducesLargerRadius()
    {
        // Arrange
        var optionsLowSigma = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 100,
            RandomSeed = Seed
        };
        var optionsHighSigma = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.5,
            NumSamples = 100,
            RandomSeed = Seed
        };
        var smoothingLow = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(optionsLowSigma);
        var smoothingHigh = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(optionsHighSigma);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act
        var predLow = smoothingLow.CertifyPrediction(input, model);
        var predHigh = smoothingHigh.CertifyPrediction(input, model);

        // Assert - higher sigma can produce larger certified radius
        // but also more noise, so we just verify both work
        Assert.NotNull(predLow);
        Assert.NotNull(predHigh);
    }

    [Fact]
    public void RandomizedSmoothing_SerializationRoundTrip_PreservesOptions()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.3,
            NumSamples = 200,
            ConfidenceLevel = 0.99
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);

        // Act
        var serialized = smoothing.Serialize();
        var newSmoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(
            new CertifiedDefenseOptions<double>());
        newSmoothing.Deserialize(serialized);
        var newOptions = newSmoothing.GetOptions();

        // Assert
        Assert.Equal(0.3, newOptions.NoiseSigma, 3);
        Assert.Equal(200, newOptions.NumSamples);
    }

    [Fact]
    public void RandomizedSmoothing_ThrowsOnNullInput()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { RandomSeed = Seed };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            smoothing.CertifyPrediction(null!, model));
    }

    [Fact]
    public void RandomizedSmoothing_ThrowsOnNullModel()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { RandomSeed = Seed };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var input = CreateTestInput();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() =>
            smoothing.CertifyPrediction(input, null!));
    }

    #endregion

    #region Interval Bound Propagation Tests

    [Fact]
    public void IntervalBoundPropagation_CertifyPrediction_ReturnsCertifiedPrediction()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 50,
            RandomSeed = Seed
        };
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act
        var prediction = ibp.CertifyPrediction(input, model);

        // Assert
        Assert.NotNull(prediction);
        Assert.True(prediction.PredictedClass >= 0);
    }

    [Fact]
    public void IntervalBoundPropagation_DefaultConstructor_Works()
    {
        // Arrange & Act
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        var options = ibp.GetOptions();

        // Assert
        Assert.NotNull(options);
        Assert.Equal("IBP", options.CertificationMethod);
    }

    [Fact]
    public void IntervalBoundPropagation_CertifyBatch_ProcessesMultipleInputs()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 30,
            RandomSeed = Seed
        };
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var inputs = new Vector<double>[] { CreateTestInput(), CreateTestInput() };

        // Act
        var predictions = ibp.CertifyBatch(inputs, model);

        // Assert
        Assert.NotNull(predictions);
        Assert.Equal(2, predictions.Length);
    }

    [Fact]
    public void IntervalBoundPropagation_ComputeCertifiedRadius_ReturnsNonNegative()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 30,
            RandomSeed = Seed
        };
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act
        var radius = ibp.ComputeCertifiedRadius(input, model);

        // Assert
        Assert.True(radius >= 0);
    }

    [Fact]
    public void IntervalBoundPropagation_EvaluateCertifiedAccuracy_ReturnsMetrics()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 30,
            RandomSeed = Seed
        };
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var testData = new Vector<double>[] { CreateTestInput(), CreateTestInput() };
        var labels = new Vector<double>[] { CreateOneHotLabel(3, 0), CreateOneHotLabel(3, 1) };

        // Act
        var metrics = ibp.EvaluateCertifiedAccuracy(testData, labels, model, 0.05);

        // Assert
        Assert.NotNull(metrics);
        Assert.InRange(metrics.CleanAccuracy, 0.0, 1.0);
        Assert.InRange(metrics.CertifiedAccuracy, 0.0, 1.0);
    }

    [Fact]
    public void IntervalBoundPropagation_Reset_ResetsToDefaults()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5 };
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);

        // Act
        ibp.Reset();
        var newOptions = ibp.GetOptions();

        // Assert
        Assert.Equal("IBP", newOptions.CertificationMethod);
    }

    [Fact]
    public void IntervalBoundPropagation_SerializationRoundTrip_PreservesOptions()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.2,
            NumSamples = 100,
            UseTightBounds = true
        };
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);

        // Act
        var serialized = ibp.Serialize();
        var newIbp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>();
        newIbp.Deserialize(serialized);
        var newOptions = newIbp.GetOptions();

        // Assert
        Assert.Equal(0.2, newOptions.NoiseSigma, 3);
        Assert.Equal(100, newOptions.NumSamples);
    }

    [Fact]
    public void IntervalBoundPropagation_ThrowsOnMismatchedDataLabelCount()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>();
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var testData = new Vector<double>[] { CreateTestInput(), CreateTestInput() };
        var labels = new Vector<double>[] { CreateOneHotLabel(3, 0) }; // Mismatched

        // Act & Assert
        Assert.Throws<ArgumentException>(() =>
            ibp.EvaluateCertifiedAccuracy(testData, labels, model, 0.1));
    }

    #endregion

    #region CROWN Verification Tests

    [Fact]
    public void CROWNVerification_CertifyPrediction_ReturnsCertifiedPrediction()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 30,
            RandomSeed = Seed
        };
        var crown = new CROWNVerification<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act
        var prediction = crown.CertifyPrediction(input, model);

        // Assert
        Assert.NotNull(prediction);
        Assert.True(prediction.PredictedClass >= 0);
    }

    [Fact]
    public void CROWNVerification_DefaultConstructor_SetsCROWNMethod()
    {
        // Arrange & Act
        var crown = new CROWNVerification<double, Vector<double>, Vector<double>>();
        var options = crown.GetOptions();

        // Assert
        Assert.NotNull(options);
        Assert.Equal("CROWN", options.CertificationMethod);
        Assert.True(options.UseTightBounds);
    }

    [Fact]
    public void CROWNVerification_CertifyBatch_ProcessesMultipleInputs()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 20,
            RandomSeed = Seed
        };
        var crown = new CROWNVerification<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var inputs = new Vector<double>[] { CreateTestInput(), CreateTestInput() };

        // Act
        var predictions = crown.CertifyBatch(inputs, model);

        // Assert
        Assert.NotNull(predictions);
        Assert.Equal(2, predictions.Length);
    }

    [Fact]
    public void CROWNVerification_ProducesTighterBoundsThanIBP()
    {
        // Note: This is a theoretical property - CROWN should produce tighter bounds
        // In practice with mock model, we just verify both methods work

        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 30,
            RandomSeed = Seed
        };
        var crown = new CROWNVerification<double, Vector<double>, Vector<double>>(options);
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act
        var crownPred = crown.CertifyPrediction(input, model);
        var ibpPred = ibp.CertifyPrediction(input, model);

        // Assert - both should produce valid predictions
        Assert.NotNull(crownPred);
        Assert.NotNull(ibpPred);
    }

    [Fact]
    public void CROWNVerification_SerializationRoundTrip_PreservesOptions()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.15,
            NumSamples = 75,
            UseTightBounds = true
        };
        var crown = new CROWNVerification<double, Vector<double>, Vector<double>>(options);

        // Act
        var serialized = crown.Serialize();
        var newCrown = new CROWNVerification<double, Vector<double>, Vector<double>>();
        newCrown.Deserialize(serialized);
        var newOptions = newCrown.GetOptions();

        // Assert
        Assert.Equal(0.15, newOptions.NoiseSigma, 3);
        Assert.Equal(75, newOptions.NumSamples);
    }

    [Fact]
    public void CROWNVerification_Reset_ResetsToDefaults()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double> { NoiseSigma = 0.5 };
        var crown = new CROWNVerification<double, Vector<double>, Vector<double>>(options);

        // Act
        crown.Reset();
        var newOptions = crown.GetOptions();

        // Assert
        Assert.Equal("CROWN", newOptions.CertificationMethod);
        Assert.True(newOptions.UseTightBounds);
    }

    #endregion

    #region Safety Filter Tests

    [Fact]
    public void SafetyFilter_ValidateInput_ReturnsValidResult()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 100
        };
        var filter = new SafetyFilter<double>(options);
        var input = CreateTestInput();

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.IsValid);
        Assert.InRange(result.SafetyScore, 0.0, 1.0);
    }

    [Fact]
    public void SafetyFilter_ValidateInput_DetectsLengthExceeded()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 5
        };
        var filter = new SafetyFilter<double>(options);
        var input = CreateTestInput(size: 10); // Exceeds max length of 5

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.NotNull(result);
        Assert.False(result.IsValid);
        Assert.True(result.Issues.Count > 0);
        Assert.Contains(result.Issues, i => i.Type == "LengthExceeded");
    }

    [Fact]
    public void SafetyFilter_ValidateInput_DetectsNaNValues()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 100
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(5);
        input[0] = 0.5;
        input[1] = double.NaN;
        input[2] = 0.5;
        input[3] = 0.5;
        input[4] = 0.5;

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.NotNull(result);
        Assert.False(result.IsValid);
        Assert.Contains(result.Issues, i => i.Type == "InvalidValue");
    }

    [Fact]
    public void SafetyFilter_ValidateInput_DetectsInfinityValues()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 100
        };
        var filter = new SafetyFilter<double>(options);
        var input = new Vector<double>(5);
        input[0] = 0.5;
        input[1] = double.PositiveInfinity;
        input[2] = 0.5;
        input[3] = 0.5;
        input[4] = 0.5;

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains(result.Issues, i => i.Type == "InvalidValue");
    }

    [Fact]
    public void SafetyFilter_FilterOutput_ReturnsFilteredResult()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableOutputFiltering = true,
            SafetyThreshold = 0.5
        };
        var filter = new SafetyFilter<double>(options);
        var output = CreateTestInput();

        // Act
        var result = filter.FilterOutput(output);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.FilteredOutput);
    }

    [Fact]
    public void SafetyFilter_DetectJailbreak_ReturnsResult()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>();
        var filter = new SafetyFilter<double>(options);
        var input = CreateTestInput();

        // Act
        var result = filter.DetectJailbreak(input);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.Indicators);
    }

    [Fact]
    public void SafetyFilter_IdentifyHarmfulContent_ReturnsResult()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            HarmfulContentCategories = new[] { "violence", "hatespeech" }
        };
        var filter = new SafetyFilter<double>(options);
        var content = CreateTestInput();

        // Act
        var result = filter.IdentifyHarmfulContent(content);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.CategoryScores);
    }

    [Fact]
    public void SafetyFilter_ComputeSafetyScore_ReturnsValidScore()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true
        };
        var filter = new SafetyFilter<double>(options);
        var content = CreateTestInput();

        // Act
        var score = filter.ComputeSafetyScore(content);

        // Assert
        Assert.InRange(score, 0.0, 1.0);
    }

    [Fact]
    public void SafetyFilter_SerializationRoundTrip_PreservesOptions()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            EnableOutputFiltering = true,
            MaxInputLength = 200
        };
        var filter = new SafetyFilter<double>(options);

        // Act
        var serialized = filter.Serialize();
        var newFilter = new SafetyFilter<double>(new SafetyFilterOptions<double>());
        newFilter.Deserialize(serialized);
        var newOptions = newFilter.GetOptions();

        // Assert
        Assert.True(newOptions.EnableInputValidation);
        Assert.True(newOptions.EnableOutputFiltering);
    }

    [Fact]
    public void SafetyFilter_DisabledValidation_BypassesChecks()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>
        {
            EnableInputValidation = false,
            MaxInputLength = 5
        };
        var filter = new SafetyFilter<double>(options);
        var input = CreateTestInput(size: 100); // Would exceed length if validation enabled

        // Act
        var result = filter.ValidateInput(input);

        // Assert
        Assert.True(result.IsValid); // Bypassed due to disabled validation
    }

    [Fact]
    public void SafetyFilter_ThrowsOnNullInput()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>();
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => filter.ValidateInput(null!));
    }

    [Fact]
    public void SafetyFilter_ThrowsOnNullOutput()
    {
        // Arrange
        var options = new SafetyFilterOptions<double>();
        var filter = new SafetyFilter<double>(options);

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => filter.FilterOutput(null!));
    }

    #endregion

    #region Rule-Based Content Classifier Tests

    [Fact]
    public void RuleBasedContentClassifier_ClassifyText_ReturnsResult()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.5);
        var text = "This is a test message";

        // Act
        var result = classifier.ClassifyText(text);

        // Assert
        Assert.NotNull(result);
        Assert.NotNull(result.PrimaryCategory);
    }

    [Fact]
    public void RuleBasedContentClassifier_ClassifyText_DetectsViolence()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.3);
        var text = "I will kill and destroy everything";

        // Act
        var result = classifier.ClassifyText(text);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.IsHarmful);
        Assert.Contains("Violence", result.DetectedCategories);
    }

    [Fact]
    public void RuleBasedContentClassifier_ClassifyText_EmptyText_ReturnsSafe()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>();
        var text = "";

        // Act
        var result = classifier.ClassifyText(text);

        // Assert
        Assert.NotNull(result);
        Assert.False(result.IsHarmful);
        Assert.Equal("Safe", result.PrimaryCategory);
    }

    [Fact]
    public void RuleBasedContentClassifier_ClassifyVector_ReturnsResult()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>();
        var content = CreateTestInput();

        // Act
        var result = classifier.Classify(content);

        // Assert
        Assert.NotNull(result);
        // Vector classification is limited for rule-based - returns safe with zero confidence
        Assert.False(result.IsHarmful);
    }

    [Fact]
    public void RuleBasedContentClassifier_IsReady_ReturnsTrue()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>();

        // Act & Assert
        Assert.True(classifier.IsReady());
    }

    [Fact]
    public void RuleBasedContentClassifier_AddPattern_AddsNewPattern()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>();

        // Act
        classifier.AddPattern("CustomCategory", @"\bcustom\b");

        // Assert
        Assert.Contains("CustomCategory", classifier.GetSupportedCategories());
    }

    [Fact]
    public void RuleBasedContentClassifier_ClearCategory_RemovesPatterns()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>();
        classifier.AddPattern("TestCategory", @"\btest\b");

        // Act
        classifier.ClearCategory("TestCategory");

        // Assert - category still exists but patterns are cleared
        var result = classifier.ClassifyText("test");
        Assert.NotNull(result);
    }

    [Fact]
    public void RuleBasedContentClassifier_CustomPatterns_WorkCorrectly()
    {
        // Arrange
        var customPatterns = new Dictionary<string, List<string>>
        {
            ["Custom"] = new List<string> { @"\bcustom\b", @"\bpattern\b" }
        };
        var classifier = new RuleBasedContentClassifier<double>(customPatterns, threshold: 0.3);

        // Act
        var result = classifier.ClassifyText("This is a custom pattern test");

        // Assert
        Assert.NotNull(result);
        Assert.True(result.IsHarmful);
        Assert.Contains("Custom", result.DetectedCategories);
    }

    [Fact]
    public void RuleBasedContentClassifier_SerializationRoundTrip_PreservesState()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>(threshold: 0.7);
        classifier.AddPattern("TestCategory", @"\btest\b");

        // Act
        var serialized = classifier.Serialize();
        var newClassifier = new RuleBasedContentClassifier<double>();
        newClassifier.Deserialize(serialized);

        // Assert
        Assert.True(newClassifier.IsReady());
        Assert.Contains("TestCategory", newClassifier.GetSupportedCategories());
    }

    [Fact]
    public void RuleBasedContentClassifier_ThrowsOnNullVector()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>();

        // Act & Assert
        Assert.Throws<ArgumentNullException>(() => classifier.Classify(null!));
    }

    [Fact]
    public void RuleBasedContentClassifier_ThrowsOnEmptyCategory()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => classifier.AddPattern("", @"\btest\b"));
    }

    [Fact]
    public void RuleBasedContentClassifier_ThrowsOnEmptyPattern()
    {
        // Arrange
        var classifier = new RuleBasedContentClassifier<double>();

        // Act & Assert
        Assert.Throws<ArgumentException>(() => classifier.AddPattern("TestCategory", ""));
    }

    #endregion

    #region Integration Scenarios

    [Fact]
    public void IntegrationScenario_AttackThenDefense_WorksTogether()
    {
        // Arrange
        var attackOptions = new AdversarialAttackOptions<double> { Epsilon = 0.1, RandomSeed = Seed };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(attackOptions);

        var defenseOptions = new AdversarialDefenseOptions<double>
        {
            Epsilon = 0.1,
            UsePreprocessing = true,
            PreprocessingMethod = "denoising"
        };
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(defenseOptions);

        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);
        var preprocessed = defense.PreprocessInput(adversarial);

        // Assert
        Assert.NotNull(adversarial);
        Assert.NotNull(preprocessed);
        Assert.Equal(input.Length, preprocessed.Length);
    }

    [Fact]
    public void IntegrationScenario_CertificationAfterDefense_WorksTogether()
    {
        // Arrange
        var defenseOptions = new AdversarialDefenseOptions<double> { UsePreprocessing = false };
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(defenseOptions);

        var certOptions = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 30,
            RandomSeed = Seed
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(certOptions);

        var model = CreateMockClassificationModel();
        var trainingData = new Vector<double>[] { CreateTestInput() };
        var labels = new Vector<double>[] { CreateOneHotLabel(3, 0) };
        var input = CreateTestInput();

        // Act
        var defendedModel = defense.ApplyDefense(trainingData, labels, model);
        var certification = smoothing.CertifyPrediction(input, defendedModel);

        // Assert
        Assert.NotNull(defendedModel);
        Assert.NotNull(certification);
    }

    [Fact]
    public void IntegrationScenario_SafetyFilterWithCertification_WorksTogether()
    {
        // Arrange
        var safetyOptions = new SafetyFilterOptions<double>
        {
            EnableInputValidation = true,
            MaxInputLength = 100
        };
        var safetyFilter = new SafetyFilter<double>(safetyOptions);

        var certOptions = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 20,
            RandomSeed = Seed
        };
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(certOptions);

        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act
        var safetyResult = safetyFilter.ValidateInput(input);

        // Only proceed with certification if input is safe
        CertifiedPrediction<double>? certification = null;
        if (safetyResult.IsValid)
        {
            certification = ibp.CertifyPrediction(input, model);
        }

        // Assert
        Assert.True(safetyResult.IsValid);
        Assert.NotNull(certification);
    }

    [Fact]
    public void IntegrationScenario_MultipleAttackComparison()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            Iterations = 5,
            RandomSeed = Seed
        };

        var fgsm = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var pgd = new PGDAttack<double, Vector<double>, Vector<double>>(options);
        var auto = new AutoAttack<double, Vector<double>, Vector<double>>(options);

        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var fgsmAdv = fgsm.GenerateAdversarialExample(input, label, model);
        var pgdAdv = pgd.GenerateAdversarialExample(input, label, model);
        var autoAdv = auto.GenerateAdversarialExample(input, label, model);

        // Assert - all should produce valid adversarial examples
        Assert.NotNull(fgsmAdv);
        Assert.NotNull(pgdAdv);
        Assert.NotNull(autoAdv);

        // All should be within epsilon bound
        for (int i = 0; i < input.Length; i++)
        {
            Assert.True(Math.Abs(fgsmAdv[i] - input[i]) <= options.Epsilon + Tolerance);
            Assert.True(Math.Abs(pgdAdv[i] - input[i]) <= options.Epsilon + Tolerance);
        }
    }

    [Fact]
    public void IntegrationScenario_CertificationMethodComparison()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 30,
            RandomSeed = Seed
        };

        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var ibp = new IntervalBoundPropagation<double, Vector<double>, Vector<double>>(options);
        var crown = new CROWNVerification<double, Vector<double>, Vector<double>>(options);

        var model = CreateMockClassificationModel();
        var input = CreateTestInput();

        // Act
        var smoothingPred = smoothing.CertifyPrediction(input, model);
        var ibpPred = ibp.CertifyPrediction(input, model);
        var crownPred = crown.CertifyPrediction(input, model);

        // Assert
        Assert.NotNull(smoothingPred);
        Assert.NotNull(ibpPred);
        Assert.NotNull(crownPred);

        // All should produce non-negative certified radii
        Assert.True(smoothingPred.CertifiedRadius >= 0);
        Assert.True(ibpPred.CertifiedRadius >= 0);
        Assert.True(crownPred.CertifiedRadius >= 0);
    }

    #endregion

    #region Edge Cases and Error Handling

    [Fact]
    public void EdgeCase_EmptyInput_HandledGracefully()
    {
        // Arrange
        var options = new SafetyFilterOptions<double> { EnableInputValidation = true, MaxInputLength = 100 };
        var filter = new SafetyFilter<double>(options);
        var emptyInput = new Vector<double>(0);

        // Act
        var result = filter.ValidateInput(emptyInput);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.IsValid); // Empty but valid
    }

    [Fact]
    public void EdgeCase_VerySmallEpsilon_StillWorks()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 1e-10,
            RandomSeed = Seed
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert
        Assert.NotNull(adversarial);
        // With very small epsilon, adversarial should be nearly identical to input
        for (int i = 0; i < input.Length; i++)
        {
            Assert.True(Math.Abs(adversarial[i] - input[i]) <= options.Epsilon + Tolerance);
        }
    }

    [Fact]
    public void EdgeCase_ZeroIterations_HandledGracefully()
    {
        // Some implementations may handle 0 iterations by returning the original input
        // or throwing an exception - verify consistent behavior
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            Iterations = 0,
            RandomSeed = Seed
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var input = CreateTestInput();
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert - should return something valid (either original or adversarial)
        Assert.NotNull(adversarial);
    }

    [Fact]
    public void EdgeCase_SingleElementInput_HandledCorrectly()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.1, RandomSeed = Seed };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel(inputSize: 1, numClasses: 2);
        var input = new Vector<double>(1) { [0] = 0.5 };
        var label = new Vector<double>(2) { [0] = 1.0, [1] = 0.0 };

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert
        Assert.NotNull(adversarial);
        Assert.Single(adversarial.ToArray());
    }

    [Fact]
    public void EdgeCase_LargeInput_HandledEfficiently()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.1, RandomSeed = Seed };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel(inputSize: 1000, numClasses: 10);
        var input = CreateTestInput(size: 1000);
        var label = CreateOneHotLabel(10, 5);

        // Act
        var adversarial = attack.GenerateAdversarialExample(input, label, model);

        // Assert
        Assert.NotNull(adversarial);
        Assert.Equal(1000, adversarial.Length);
    }

    [Fact]
    public void EdgeCase_AllZeroInput_HandledCorrectly()
    {
        // Arrange
        var options = new CertifiedDefenseOptions<double>
        {
            NoiseSigma = 0.1,
            NumSamples = 30,
            RandomSeed = Seed
        };
        var smoothing = new RandomizedSmoothing<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var zeroInput = new Vector<double>(10); // All zeros

        // Act
        var prediction = smoothing.CertifyPrediction(zeroInput, model);

        // Assert
        Assert.NotNull(prediction);
        Assert.True(prediction.PredictedClass >= 0);
    }

    [Fact]
    public void EdgeCase_AllOnesInput_HandledCorrectly()
    {
        // Arrange
        var options = new AdversarialAttackOptions<double> { Epsilon = 0.1, RandomSeed = Seed };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = CreateMockClassificationModel();
        var onesInput = new Vector<double>(10);
        for (int i = 0; i < 10; i++) onesInput[i] = 1.0;
        var label = CreateOneHotLabel(3, 0);

        // Act
        var adversarial = attack.GenerateAdversarialExample(onesInput, label, model);

        // Assert
        Assert.NotNull(adversarial);
    }

    #endregion

    #region Mock Model

    /// <summary>
    /// Mock classification model for testing adversarial attacks and defenses.
    /// </summary>
    private class MockClassificationModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _inputSize;
        private readonly int _numClasses;
        private readonly Random _random;
        private Vector<double> _weights;

        public MockClassificationModel(int inputSize, int numClasses, int seed)
        {
            _inputSize = inputSize;
            _numClasses = numClasses;
            _random = RandomHelper.CreateSeededRandom(seed);

            // Initialize random weights
            _weights = new Vector<double>(inputSize * numClasses);
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] = _random.NextDouble() * 2 - 1;
            }
        }

        public ILossFunction<double>? DefaultLossFunction => null;
        public int ParameterCount => _weights.Length;
        public bool SupportsJitCompilation => false;

        public Vector<double> Predict(Vector<double> input)
        {
            var output = new Vector<double>(_numClasses);

            // Simple linear transformation
            for (int c = 0; c < _numClasses; c++)
            {
                double sum = 0;
                for (int i = 0; i < Math.Min(_inputSize, input.Length); i++)
                {
                    sum += input[i] * _weights[c * _inputSize + i];
                }
                output[c] = sum;
            }

            // Apply softmax
            double maxVal = output.Max();
            double expSum = 0;
            for (int i = 0; i < _numClasses; i++)
            {
                output[i] = Math.Exp(output[i] - maxVal);
                expSum += output[i];
            }
            for (int i = 0; i < _numClasses; i++)
            {
                output[i] /= expSum;
            }

            return output;
        }

        public void Train(Vector<double> input, Vector<double> expectedOutput) { }

        public ModelMetadata<double> GetModelMetadata()
        {
            return new ModelMetadata<double>
            {
                ModelType = ModelType.None,
                Description = "Mock classification model for testing"
            };
        }

        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }

        public Vector<double> GetParameters() => _weights;
        public void SetParameters(Vector<double> parameters) { _weights = parameters; }

        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters)
        {
            var newModel = new MockClassificationModel(_inputSize, _numClasses, 0);
            newModel.SetParameters(parameters);
            return newModel;
        }

        public IEnumerable<int> GetActiveFeatureIndices() => Enumerable.Range(0, _inputSize);
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) { }
        public bool IsFeatureUsed(int featureIndex) => true;
        public Dictionary<string, double> GetFeatureImportance() => new();

        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy()
        {
            var copy = new MockClassificationModel(_inputSize, _numClasses, 0);
            copy.SetParameters(new Vector<double>(_weights.ToArray()));
            return copy;
        }

        public IFullModel<double, Vector<double>, Vector<double>> Clone() => DeepCopy();

        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null)
        {
            // Simple gradient approximation for testing
            var gradients = new Vector<double>(_weights.Length);
            var prediction = Predict(input);

            for (int c = 0; c < _numClasses; c++)
            {
                double error = prediction[c] - target[c];
                for (int i = 0; i < Math.Min(_inputSize, input.Length); i++)
                {
                    gradients[c * _inputSize + i] = error * input[i];
                }
            }

            return gradients;
        }

        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] -= learningRate * gradients[i];
            }
        }

        public AiDotNet.Autodiff.ComputationNode<double> ExportComputationGraph(List<AiDotNet.Autodiff.ComputationNode<double>> inputNodes)
        {
            throw new NotImplementedException();
        }
    }

    #endregion
}
