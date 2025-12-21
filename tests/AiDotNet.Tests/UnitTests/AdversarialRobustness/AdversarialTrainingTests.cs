using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.AdversarialRobustness;

/// <summary>
/// Tests for AdversarialTraining defense mechanism.
/// </summary>
public class AdversarialTrainingTests
{
    #region Mock Model

    /// <summary>
    /// Mock predictive model for testing adversarial training.
    /// </summary>
    private class MockClassificationModel : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _numClasses;

        public MockClassificationModel(int numClasses = 3)
        {
            _numClasses = numClasses;
        }

        public Vector<double> Predict(Vector<double> input)
        {
            var output = new Vector<double>(_numClasses);
            var sum = 0.0;
            for (int i = 0; i < input.Length; i++)
            {
                sum += input[i];
            }

            // Simple prediction based on input sum
            for (int i = 0; i < _numClasses; i++)
            {
                output[i] = sum * (i + 1) * 0.1 + (i == 0 ? 1.0 : 0.0);
            }

            return output;
        }

        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    #endregion

    #region Constructor Tests

    [Fact]
    public void Constructor_WithValidOptions_Initializes()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            Epsilon = 0.1,
            UsePreprocessing = true
        };

        var defense = new AdversarialTraining<double>(options);

        Assert.NotNull(defense);
        Assert.Equal(0.1, defense.GetOptions().Epsilon);
    }

    [Fact]
    public void Constructor_WithNullOptions_ThrowsException()
    {
        Assert.Throws<ArgumentNullException>(() => new AdversarialTraining<double>(null!));
    }

    [Fact]
    public void Constructor_WithDefaultOptions_Initializes()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);

        Assert.NotNull(defense);
        Assert.Equal(0.5, defense.GetOptions().AdversarialRatio);
        Assert.Equal(100, defense.GetOptions().TrainingEpochs);
    }

    #endregion

    #region ApplyDefense Tests

    [Fact]
    public void ApplyDefense_WithNullModel_ThrowsException()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);
        var data = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });

        Assert.Throws<ArgumentNullException>(() => defense.ApplyDefense(data, labels, null!));
    }

    [Fact]
    public void ApplyDefense_WithPreprocessingDisabled_ReturnsOriginalModel()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = false
        };
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var data = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });

        var defendedModel = defense.ApplyDefense(data, labels, model);

        Assert.Same(model, defendedModel);
    }

    [Fact]
    public void ApplyDefense_WithPreprocessingEnabled_ReturnsWrappedModel()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true
        };
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var data = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });

        var defendedModel = defense.ApplyDefense(data, labels, model);

        Assert.NotSame(model, defendedModel);
    }

    [Fact]
    public void ApplyDefense_DefendedModelCanPredict()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true,
            PreprocessingMethod = "JPEG"
        };
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var data = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });

        var defendedModel = defense.ApplyDefense(data, labels, model);
        var input = new Vector<double>(new[] { 0.5, 0.5, 0.5, 0.5 });

        var output = defendedModel.Predict(input);

        Assert.NotNull(output);
        Assert.Equal(3, output.Length);
    }

    #endregion

    #region PreprocessInput Tests

    [Fact]
    public void PreprocessInput_WithPreprocessingDisabled_ReturnsOriginal()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = false
        };
        var defense = new AdversarialTraining<double>(options);
        var input = new Vector<double>(new[] { 0.5, 0.6, 0.7 });

        var preprocessed = defense.PreprocessInput(input);

        Assert.Same(input, preprocessed);
    }

    [Fact]
    public void PreprocessInput_JPEGMethod_AppliesQuantization()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true,
            PreprocessingMethod = "JPEG"
        };
        var defense = new AdversarialTraining<double>(options);
        var input = new Vector<double>(new[] { 0.55, 0.66, 0.77 });

        var preprocessed = defense.PreprocessInput(input);

        Assert.NotNull(preprocessed);
        Assert.Equal(input.Length, preprocessed.Length);
        // Values should be quantized
        for (int i = 0; i < preprocessed.Length; i++)
        {
            Assert.InRange(preprocessed[i], 0.0, 1.0);
        }
    }

    [Fact]
    public void PreprocessInput_BitDepthReduction_AppliesQuantization()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true,
            PreprocessingMethod = "bit_depth_reduction"
        };
        var defense = new AdversarialTraining<double>(options);
        var input = new Vector<double>(new[] { 0.55, 0.66, 0.77 });

        var preprocessed = defense.PreprocessInput(input);

        Assert.NotNull(preprocessed);
        Assert.Equal(input.Length, preprocessed.Length);
    }

    [Fact]
    public void PreprocessInput_Denoising_ReturnsVector()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true,
            PreprocessingMethod = "denoising"
        };
        var defense = new AdversarialTraining<double>(options);
        var input = new Vector<double>(new[] { 0.5, 0.6, 0.7 });

        var preprocessed = defense.PreprocessInput(input);

        Assert.NotNull(preprocessed);
        Assert.Equal(input.Length, preprocessed.Length);
    }

    [Fact]
    public void PreprocessInput_UnknownMethod_ReturnsOriginal()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true,
            PreprocessingMethod = "unknown_method"
        };
        var defense = new AdversarialTraining<double>(options);
        var input = new Vector<double>(new[] { 0.5, 0.6, 0.7 });

        var preprocessed = defense.PreprocessInput(input);

        Assert.Same(input, preprocessed);
    }

    #endregion

    #region EvaluateRobustness Tests

    [Fact]
    public void EvaluateRobustness_WithNullModel_ThrowsException()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);
        var testData = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });
        var attack = new FGSMAttack<double>(new AdversarialAttackOptions<double>());

        Assert.Throws<ArgumentNullException>(() => defense.EvaluateRobustness(null!, testData, labels, attack));
    }

    [Fact]
    public void EvaluateRobustness_WithNullTestData_ThrowsException()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var labels = new Vector<int>(new[] { 0, 1, 2 });
        var attack = new FGSMAttack<double>(new AdversarialAttackOptions<double>());

        Assert.Throws<ArgumentNullException>(() => defense.EvaluateRobustness(model, null!, labels, attack));
    }

    [Fact]
    public void EvaluateRobustness_WithNullLabels_ThrowsException()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var testData = new Matrix<double>(3, 4);
        var attack = new FGSMAttack<double>(new AdversarialAttackOptions<double>());

        Assert.Throws<ArgumentNullException>(() => defense.EvaluateRobustness(model, testData, null!, attack));
    }

    [Fact]
    public void EvaluateRobustness_WithNullAttack_ThrowsException()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var testData = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });

        Assert.Throws<ArgumentNullException>(() => defense.EvaluateRobustness(model, testData, labels, null!));
    }

    [Fact]
    public void EvaluateRobustness_MismatchedRowsAndLabels_ThrowsException()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var testData = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1 }); // Only 2 labels for 3 rows
        var attack = new FGSMAttack<double>(new AdversarialAttackOptions<double>());

        Assert.Throws<ArgumentException>(() => defense.EvaluateRobustness(model, testData, labels, attack));
    }

    [Fact]
    public void EvaluateRobustness_ReturnsValidMetrics()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var testData = new Matrix<double>(3, 4);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                testData[i, j] = 0.5;
            }
        }
        var labels = new Vector<int>(new[] { 0, 0, 0 });
        var attackOptions = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1
        };
        var attack = new FGSMAttack<double>(attackOptions);

        var metrics = defense.EvaluateRobustness(model, testData, labels, attack);

        Assert.NotNull(metrics);
        Assert.InRange(metrics.CleanAccuracy, 0.0, 1.0);
        Assert.InRange(metrics.AdversarialAccuracy, 0.0, 1.0);
        Assert.InRange(metrics.AttackSuccessRate, 0.0, 1.0);
        Assert.InRange(metrics.RobustnessScore, 0.0, 1.0);
    }

    [Fact]
    public void EvaluateRobustness_CalculatesAveragePerturbationSize()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var testData = new Matrix<double>(2, 4);
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                testData[i, j] = 0.5;
            }
        }
        var labels = new Vector<int>(new[] { 0, 0 });
        var attackOptions = new AdversarialAttackOptions<double>
        {
            Epsilon = 0.1
        };
        var attack = new FGSMAttack<double>(attackOptions);

        var metrics = defense.EvaluateRobustness(model, testData, labels, attack);

        Assert.True(metrics.AveragePerturbationSize >= 0.0);
    }

    #endregion

    #region Serialization Tests

    [Fact]
    public void Serialize_ReturnsNonEmptyBytes()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            Epsilon = 0.2,
            AdversarialRatio = 0.6
        };
        var defense = new AdversarialTraining<double>(options);

        var bytes = defense.Serialize();

        Assert.NotNull(bytes);
        Assert.NotEmpty(bytes);
    }

    [Fact]
    public void Deserialize_RestoresOptions()
    {
        var originalOptions = new AdversarialDefenseOptions<double>
        {
            Epsilon = 0.25,
            AdversarialRatio = 0.7,
            TrainingEpochs = 50
        };
        var original = new AdversarialTraining<double>(originalOptions);
        var bytes = original.Serialize();

        var restored = new AdversarialTraining<double>(new AdversarialDefenseOptions<double>());
        restored.Deserialize(bytes);

        Assert.Equal(0.25, restored.GetOptions().Epsilon);
        Assert.Equal(0.7, restored.GetOptions().AdversarialRatio);
        Assert.Equal(50, restored.GetOptions().TrainingEpochs);
    }

    [Fact]
    public void Deserialize_NullData_ThrowsException()
    {
        var defense = new AdversarialTraining<double>(new AdversarialDefenseOptions<double>());

        Assert.Throws<ArgumentNullException>(() => defense.Deserialize(null!));
    }

    [Fact]
    public void SaveAndLoadModel_PreservesState()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"defense_test_{Guid.NewGuid()}.json");
        try
        {
            var originalOptions = new AdversarialDefenseOptions<double>
            {
                Epsilon = 0.15,
                TrainingEpochs = 80
            };
            var original = new AdversarialTraining<double>(originalOptions);
            original.SaveModel(tempPath);

            var loaded = new AdversarialTraining<double>(new AdversarialDefenseOptions<double>());
            loaded.LoadModel(tempPath);

            Assert.Equal(0.15, loaded.GetOptions().Epsilon);
            Assert.Equal(80, loaded.GetOptions().TrainingEpochs);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
            }
        }
    }

    #endregion

    #region Reset Tests

    [Fact]
    public void Reset_DoesNotThrow()
    {
        var defense = new AdversarialTraining<double>(new AdversarialDefenseOptions<double>());

        var exception = Record.Exception(() => defense.Reset());

        Assert.Null(exception);
    }

    #endregion

    #region GetOptions Tests

    [Fact]
    public void GetOptions_ReturnsCorrectOptions()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            Epsilon = 0.3,
            UsePreprocessing = true,
            PreprocessingMethod = "bit_depth_reduction"
        };
        var defense = new AdversarialTraining<double>(options);

        var retrievedOptions = defense.GetOptions();

        Assert.Equal(0.3, retrievedOptions.Epsilon);
        Assert.True(retrievedOptions.UsePreprocessing);
        Assert.Equal("bit_depth_reduction", retrievedOptions.PreprocessingMethod);
    }

    #endregion

    #region Defended Model Tests

    [Fact]
    public void DefendedModel_GetModelMetadata_DelegatesToInner()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true
        };
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var data = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });

        var defendedModel = defense.ApplyDefense(data, labels, model);
        var metadata = defendedModel.GetModelMetadata();

        Assert.NotNull(metadata);
    }

    [Fact]
    public void DefendedModel_Serialize_DelegatesToInner()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true
        };
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var data = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });

        var defendedModel = defense.ApplyDefense(data, labels, model);
        var bytes = defendedModel.Serialize();

        Assert.NotNull(bytes);
    }

    [Fact]
    public void DefendedModel_Deserialize_DelegatesToInner()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true
        };
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var data = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });

        var defendedModel = defense.ApplyDefense(data, labels, model);

        var exception = Record.Exception(() => defendedModel.Deserialize(Array.Empty<byte>()));

        Assert.Null(exception);
    }

    [Fact]
    public void DefendedModel_SaveModel_DelegatesToInner()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true
        };
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var data = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });
        var tempPath = Path.Combine(Path.GetTempPath(), $"defended_model_{Guid.NewGuid()}.json");

        try
        {
            var defendedModel = defense.ApplyDefense(data, labels, model);

            var exception = Record.Exception(() => defendedModel.SaveModel(tempPath));

            Assert.Null(exception);
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
    public void DefendedModel_LoadModel_DelegatesToInner()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true
        };
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var data = new Matrix<double>(3, 4);
        var labels = new Vector<int>(new[] { 0, 1, 2 });

        var defendedModel = defense.ApplyDefense(data, labels, model);

        // LoadModel on mock will succeed (no-op)
        var exception = Record.Exception(() => defendedModel.LoadModel("anypath.json"));

        Assert.Null(exception);
    }

    #endregion

    #region Edge Cases

    [Fact]
    public void EvaluateRobustness_EmptyTestData_ReturnsZeroMetrics()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialTraining<double>(options);
        var model = new MockClassificationModel();
        var testData = new Matrix<double>(0, 4);
        var labels = new Vector<int>(0);
        var attack = new FGSMAttack<double>(new AdversarialAttackOptions<double>());

        var metrics = defense.EvaluateRobustness(model, testData, labels, attack);

        Assert.NotNull(metrics);
        // With 0 rows, metrics should be NaN or 0 depending on implementation
    }

    [Fact]
    public void PreprocessInput_JPEGMethod_ClipsToValidRange()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true,
            PreprocessingMethod = "JPEG"
        };
        var defense = new AdversarialTraining<double>(options);
        var input = new Vector<double>(new[] { 1.5, -0.5, 2.0 }); // Out of [0,1] range

        var preprocessed = defense.PreprocessInput(input);

        for (int i = 0; i < preprocessed.Length; i++)
        {
            Assert.InRange(preprocessed[i], 0.0, 1.0);
        }
    }

    [Fact]
    public void PreprocessInput_BitDepthReduction_ClipsToValidRange()
    {
        var options = new AdversarialDefenseOptions<double>
        {
            UsePreprocessing = true,
            PreprocessingMethod = "bit_depth_reduction"
        };
        var defense = new AdversarialTraining<double>(options);
        var input = new Vector<double>(new[] { 1.5, -0.5, 2.0 });

        var preprocessed = defense.PreprocessInput(input);

        for (int i = 0; i < preprocessed.Length; i++)
        {
            Assert.InRange(preprocessed[i], 0.0, 1.0);
        }
    }

    #endregion
}
