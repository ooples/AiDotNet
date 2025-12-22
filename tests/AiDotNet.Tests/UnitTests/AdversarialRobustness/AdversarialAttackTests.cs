using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AdversarialRobustness;

/// <summary>
/// Tests for adversarial attack implementations (FGSM, PGD, C&W, AutoAttack).
/// </summary>
public class AdversarialAttackTests
{
    #region Helper Methods

    /// <summary>
    /// Creates a one-hot encoded label vector for a given class index.
    /// </summary>
    private static Vector<double> OneHotLabel(int classIndex, int numClasses = 3)
    {
        var label = new Vector<double>(numClasses);
        label[classIndex] = 1.0;
        return label;
    }

    #endregion

    #region Mock Model

    /// <summary>
    /// Mock predictive model for testing adversarial attacks.
    /// </summary>
    private class MockClassificationModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _numClasses;
        private readonly int _inputDim;
        private List<int> _activeFeatures;

        public MockClassificationModel(int numClasses = 3, int inputDim = 3)
        {
            _numClasses = numClasses;
            _inputDim = inputDim;
            _activeFeatures = Enumerable.Range(0, inputDim).ToList();
        }

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

        public Vector<double> Predict(Vector<double> input)
        {
            // Simple mock: output logits based on input sum
            var output = new Vector<double>(_numClasses);
            var sum = 0.0;
            for (int i = 0; i < input.Length; i++)
            {
                sum += input[i];
            }

            // Create class logits based on input sum
            for (int i = 0; i < _numClasses; i++)
            {
                output[i] = sum * (i + 1) * 0.1 + (i == 0 ? 1.0 : 0.0);
            }

            return output;
        }

        public void Train(Vector<double> input, Vector<double> expectedOutput) { }
        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public IFullModel<double, Vector<double>, Vector<double>> Clone() => new MockClassificationModel(_numClasses, _inputDim);
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() => new MockClassificationModel(_numClasses, _inputDim);
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(_inputDim);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }
        public Vector<double> GetParameters() => new Vector<double>(_inputDim * _numClasses);
        public void SetParameters(Vector<double> parameters) { }
        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters) => new MockClassificationModel(_numClasses, _inputDim);
        public int ParameterCount => _inputDim * _numClasses;
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => _activeFeatures = featureIndices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => Enumerable.Range(0, _inputDim).ToDictionary(i => $"Feature{i}", i => 1.0 / _inputDim);
        public bool SupportsJitCompilation => false;
        public ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes) => throw new NotSupportedException();
    }

    #endregion

    #region FGSM Attack Tests

    [Fact]
    public void FGSMAttack_Constructor_WithValidOptions_Initializes()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            NormType = "L-infinity"
        };

        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);

        Assert.NotNull(attack);
        Assert.Equal(0.1, attack.GetOptions().Epsilon);
    }

    [Fact]
    public void FGSMAttack_Constructor_WithNullOptions_ThrowsException()
    {
        Assert.Throws<ArgumentNullException>(() => new FGSMAttack<double, Vector<double>, Vector<double>>(null!));
    }

    [Fact]
    public void FGSMAttack_GenerateAdversarialExample_WithNullInput_ThrowsException()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();

        Assert.Throws<ArgumentNullException>(() => attack.GenerateAdversarialExample(null!, OneHotLabel(0), model));
    }

    [Fact]
    public void FGSMAttack_GenerateAdversarialExample_WithNullModel_ThrowsException()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        Assert.Throws<ArgumentNullException>(() => attack.GenerateAdversarialExample(input, OneHotLabel(0), null!));
    }

    [Fact]
    public void FGSMAttack_GenerateAdversarialExample_ReturnsValidAdversarial()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            NormType = "L-infinity"
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        Assert.NotNull(adversarial);
        Assert.Equal(input.Length, adversarial.Length);
    }

    [Fact]
    public void FGSMAttack_GenerateAdversarialExample_RespectsBounds()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.5
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.1, 0.9, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        // All values should be clipped to [0, 1]
        for (int i = 0; i < adversarial.Length; i++)
        {
            Assert.InRange(adversarial[i], 0.0, 1.0);
        }
    }

    [Fact]
    public void FGSMAttack_GenerateAdversarialExample_PerturbationWithinEpsilon()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);
        var perturbation = attack.CalculatePerturbation(input, adversarial);

        // Check L-infinity norm of perturbation
        var maxPerturbation = 0.0;
        for (int i = 0; i < perturbation.Length; i++)
        {
            maxPerturbation = Math.Max(maxPerturbation, Math.Abs(perturbation[i]));
        }

        Assert.True(maxPerturbation <= options.Epsilon + 1e-10);
    }

    [Fact]
    public void FGSMAttack_TargetedAttack_WorksCorrectly()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            IsTargeted = true,
            TargetClass = 2
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        Assert.NotNull(adversarial);
        Assert.Equal(input.Length, adversarial.Length);
    }

    #endregion

    #region PGD Attack Tests

    [Fact]
    public void PGDAttack_Constructor_WithValidOptions_Initializes()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            StepSize = 0.01,
            Iterations = 40
        };

        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);

        Assert.NotNull(attack);
        Assert.Equal(40, attack.GetOptions().Iterations);
    }

    [Fact]
    public void PGDAttack_GenerateAdversarialExample_ReturnsValidAdversarial()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            StepSize = 0.01,
            Iterations = 10
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        Assert.NotNull(adversarial);
        Assert.Equal(input.Length, adversarial.Length);
    }

    [Fact]
    public void PGDAttack_GenerateAdversarialExample_RespectsBounds()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.3,
            StepSize = 0.05,
            Iterations = 10
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.1, 0.9, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        for (int i = 0; i < adversarial.Length; i++)
        {
            Assert.InRange(adversarial[i], 0.0, 1.0);
        }
    }

    [Fact]
    public void PGDAttack_WithRandomStart_ProducesDifferentResults()
    {
        var options1 = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            StepSize = 0.01,
            Iterations = 10,
            UseRandomStart = true,
            RandomSeed = 1
        };
        var options2 = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            StepSize = 0.01,
            Iterations = 10,
            UseRandomStart = true,
            RandomSeed = 2
        };
        var attack1 = new PGDAttack<double, Vector<double>, Vector<double>>(options1);
        var attack2 = new PGDAttack<double, Vector<double>, Vector<double>>(options2);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial1 = attack1.GenerateAdversarialExample(input, OneHotLabel(0), model);
        var adversarial2 = attack2.GenerateAdversarialExample(input, OneHotLabel(0), model);

        // With different seeds, results should differ
        bool anyDifferent = false;
        for (int i = 0; i < adversarial1.Length; i++)
        {
            if (Math.Abs(adversarial1[i] - adversarial2[i]) > 1e-10)
            {
                anyDifferent = true;
                break;
            }
        }

        Assert.True(anyDifferent);
    }

    [Fact]
    public void PGDAttack_L2Norm_WorksCorrectly()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.5,
            StepSize = 0.1,
            Iterations = 10,
            NormType = "L2"
        };
        var attack = new PGDAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        Assert.NotNull(adversarial);
    }

    #endregion

    #region C&W Attack Tests

    [Fact]
    public void CWAttack_Constructor_WithValidOptions_Initializes()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1
        };

        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);

        Assert.NotNull(attack);
    }

    [Fact]
    public void CWAttack_GenerateAdversarialExample_ReturnsValidAdversarial()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            Iterations = 10 // Reduce for faster testing
        };
        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        Assert.NotNull(adversarial);
        Assert.Equal(input.Length, adversarial.Length);
    }

    [Fact]
    public void CWAttack_GenerateAdversarialExample_RespectsBounds()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.3,
            Iterations = 5
        };
        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.1, 0.9, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        for (int i = 0; i < adversarial.Length; i++)
        {
            Assert.InRange(adversarial[i], 0.0, 1.0);
        }
    }

    [Fact]
    public void CWAttack_TargetedAttack_WorksCorrectly()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            Iterations = 5,
            IsTargeted = true,
            TargetClass = 1
        };
        var attack = new CWAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        Assert.NotNull(adversarial);
    }

    #endregion

    #region AutoAttack Tests

    [Fact]
    public void AutoAttack_Constructor_WithValidOptions_Initializes()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1
        };

        var attack = new AutoAttack<double, Vector<double>, Vector<double>>(options);

        Assert.NotNull(attack);
    }

    [Fact]
    public void AutoAttack_GenerateAdversarialExample_ReturnsValidAdversarial()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1,
            Iterations = 5 // Reduce for faster testing
        };
        var attack = new AutoAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        Assert.NotNull(adversarial);
        Assert.Equal(input.Length, adversarial.Length);
    }

    [Fact]
    public void AutoAttack_GenerateAdversarialExample_RespectsBounds()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.3,
            Iterations = 3
        };
        var attack = new AutoAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.1, 0.9, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        for (int i = 0; i < adversarial.Length; i++)
        {
            Assert.InRange(adversarial[i], 0.0, 1.0);
        }
    }

    #endregion

    #region Batch Processing Tests

    [Fact]
    public void FGSMAttack_GenerateAdversarialBatch_ProcessesAllInputs()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel(3, 4);
        var inputs = new Vector<double>[3];
        for (int i = 0; i < 3; i++)
        {
            inputs[i] = new Vector<double>(new[] { 0.5, 0.5, 0.5, 0.5 });
        }
        var labels = new Vector<double>[] { OneHotLabel(0), OneHotLabel(1), OneHotLabel(2) };

        var adversarialBatch = attack.GenerateAdversarialBatch(inputs, labels, model);

        Assert.Equal(3, adversarialBatch.Length);
        Assert.Equal(4, adversarialBatch[0].Length);
    }

    [Fact]
    public void GenerateAdversarialBatch_NullInputs_ThrowsException()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var labels = new Vector<double>[] { OneHotLabel(0), OneHotLabel(1), OneHotLabel(2) };

        Assert.Throws<ArgumentNullException>(() => attack.GenerateAdversarialBatch(null!, labels, model));
    }

    [Fact]
    public void GenerateAdversarialBatch_NullLabels_ThrowsException()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var inputs = new Vector<double>[]
        {
            new Vector<double>(new[] { 0.5, 0.5, 0.5 }),
            new Vector<double>(new[] { 0.5, 0.5, 0.5 }),
            new Vector<double>(new[] { 0.5, 0.5, 0.5 })
        };

        Assert.Throws<ArgumentNullException>(() => attack.GenerateAdversarialBatch(inputs, null!, model));
    }

    [Fact]
    public void GenerateAdversarialBatch_MismatchedLabelCount_ThrowsException()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var inputs = new Vector<double>[]
        {
            new Vector<double>(new[] { 0.5, 0.5, 0.5 }),
            new Vector<double>(new[] { 0.5, 0.5, 0.5 }),
            new Vector<double>(new[] { 0.5, 0.5, 0.5 })
        };
        var labels = new Vector<double>[] { OneHotLabel(0), OneHotLabel(1) }; // Only 2 labels for 3 inputs

        Assert.Throws<ArgumentException>(() => attack.GenerateAdversarialBatch(inputs, labels, model));
    }

    #endregion

    #region Perturbation Calculation Tests

    [Fact]
    public void CalculatePerturbation_ReturnsCorrectDifference()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var original = new Vector<double>(new[] { 0.5, 0.6, 0.7 });
        var adversarial = new Vector<double>(new[] { 0.6, 0.5, 0.8 });

        var perturbation = attack.CalculatePerturbation(original, adversarial);

        Assert.Equal(0.1, perturbation[0], 5);
        Assert.Equal(-0.1, perturbation[1], 5);
        Assert.Equal(0.1, perturbation[2], 5);
    }

    [Fact]
    public void CalculatePerturbation_NullOriginal_ThrowsException()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var adversarial = new Vector<double>(new[] { 0.6, 0.5, 0.8 });

        Assert.Throws<ArgumentNullException>(() => attack.CalculatePerturbation(null!, adversarial));
    }

    [Fact]
    public void CalculatePerturbation_NullAdversarial_ThrowsException()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var original = new Vector<double>(new[] { 0.5, 0.6, 0.7 });

        Assert.Throws<ArgumentNullException>(() => attack.CalculatePerturbation(original, null!));
    }

    [Fact]
    public void CalculatePerturbation_DifferentLengths_ThrowsException()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var original = new Vector<double>(new[] { 0.5, 0.6, 0.7 });
        var adversarial = new Vector<double>(new[] { 0.6, 0.5 });

        Assert.Throws<ArgumentException>(() => attack.CalculatePerturbation(original, adversarial));
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void FGSMAttack_Serialize_ReturnsNonEmptyBytes()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.2,
            StepSize = 0.05
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);

        var bytes = attack.Serialize();

        Assert.NotNull(bytes);
        Assert.NotEmpty(bytes);
    }

    [Fact]
    public void FGSMAttack_Deserialize_RestoresOptions()
    {
        var originalOptions = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.25,
            StepSize = 0.03,
            Iterations = 50
        };
        var original = new FGSMAttack<double, Vector<double>, Vector<double>>(originalOptions);
        var bytes = original.Serialize();

        var restored = new FGSMAttack<double, Vector<double>, Vector<double>>(new AdversarialAttackOptions<double>());
        restored.Deserialize(bytes);

        Assert.Equal(0.25, restored.GetOptions().Epsilon);
        Assert.Equal(0.03, restored.GetOptions().StepSize);
        Assert.Equal(50, restored.GetOptions().Iterations);
    }

    [Fact]
    public void FGSMAttack_Deserialize_NullData_ThrowsException()
    {
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(new AdversarialAttackOptions<double>());

        Assert.Throws<ArgumentNullException>(() => attack.Deserialize(null!));
    }

    [Fact]
    public void FGSMAttack_SaveAndLoadModel_PreservesState()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"attack_test_{Guid.NewGuid()}.json");
        try
        {
            var originalOptions = new AdversarialAttackOptions<double>
            {
                Epsilon = 0.15,
                Iterations = 30
            };
            var original = new FGSMAttack<double, Vector<double>, Vector<double>>(originalOptions);
            original.SaveModel(tempPath);

            var loaded = new FGSMAttack<double, Vector<double>, Vector<double>>(new AdversarialAttackOptions<double>());
            loaded.LoadModel(tempPath);

            Assert.Equal(0.15, loaded.GetOptions().Epsilon);
            Assert.Equal(30, loaded.GetOptions().Iterations);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    [Fact]
    public void SaveModel_NullPath_ThrowsException()
    {
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(new AdversarialAttackOptions<double>());

        Assert.Throws<ArgumentException>(() => attack.SaveModel(null!));
    }

    [Fact]
    public void LoadModel_NonExistentFile_ThrowsException()
    {
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(new AdversarialAttackOptions<double>());

        Assert.Throws<FileNotFoundException>(() => attack.LoadModel("nonexistent_file.json"));
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_DoesNotThrow()
    {
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(new AdversarialAttackOptions<double>());

        var exception = Record.Exception(() => attack.Reset());

        Assert.Null(exception);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void FGSMAttack_ZeroEpsilon_ReturnsOriginalInput()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.0
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(input[i], adversarial[i], 5);
        }
    }

    [Fact]
    public void FGSMAttack_LargeEpsilon_ClipsToValidRange()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 10.0 // Very large epsilon
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        for (int i = 0; i < adversarial.Length; i++)
        {
            Assert.InRange(adversarial[i], 0.0, 1.0);
        }
    }

    [Fact]
    public void FGSMAttack_SingleDimensionInput_WorksCorrectly()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(new[] { 0.5 });

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        Assert.Single(adversarial.ToArray());
    }

    [Fact]
    public void FGSMAttack_LargeInput_HandlesCorrectly()
    {
        var options = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1
        };
        var attack = new FGSMAttack<double, Vector<double>, Vector<double>>(options);
        var model = new MockClassificationModel();
        var input = new Vector<double>(100);
        for (int i = 0; i < 100; i++)
        {
            input[i] = 0.5;
        }

        var adversarial = attack.GenerateAdversarialExample(input, OneHotLabel(0), model);

        Assert.Equal(100, adversarial.Length);
    }

    #endregion
}
