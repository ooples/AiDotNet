#nullable disable
using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Safety;
using AiDotNet.Safety.Adversarial;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Safety;

/// <summary>
/// Integration tests for adversarial robustness modules.
/// Tests AdversarialRobustnessEvaluator, AdversarialImageEvaluator,
/// ViTAdversarialAttack, AdaptiveRandomizedSmoothing, AdversarialPromptDefense,
/// and AdversarialPreferenceAlignment with mock models and gradient-based training verification.
/// </summary>
public class AdversarialRobustnessIntegrationTests
{
    private static readonly INumericOperations<double> NumOps = MathHelper.GetNumericOperations<double>();

    #region AdversarialRobustnessEvaluator Tests

    [Fact]
    public void Evaluator_Homoglyphs_DetectsAttack()
    {
        var evaluator = new AdversarialRobustnessEvaluator<double>();
        var findings = evaluator.EvaluateText("H\u0435ll\u043E w\u043Erld th\u0456s \u0456s \u0430 t\u0435st");

        Assert.NotEmpty(findings);
        Assert.Contains(findings, f => f.Category == SafetyCategory.PromptInjection);
    }

    [Fact]
    public void Evaluator_InvisibleChars_DetectsAttack()
    {
        var evaluator = new AdversarialRobustnessEvaluator<double>();
        var findings = evaluator.EvaluateText(
            "Hello\u200B\u200Bworld\u200C\u200Dtest\u200B\u200Binvisible");

        Assert.NotEmpty(findings);
    }

    [Fact]
    public void Evaluator_NormalText_NoFindings()
    {
        var evaluator = new AdversarialRobustnessEvaluator<double>();
        var findings = evaluator.EvaluateText("This is a normal English sentence with no tricks.");

        Assert.Empty(findings);
    }

    [Fact]
    public void Evaluator_CustomThreshold_Works()
    {
        var strict = new AdversarialRobustnessEvaluator<double>(threshold: 0.1);
        var lenient = new AdversarialRobustnessEvaluator<double>(threshold: 0.9);
        var text = "H\u0435llo";

        var strictFindings = strict.EvaluateText(text);
        var lenientFindings = lenient.EvaluateText(text);

        Assert.True(strictFindings.Count >= lenientFindings.Count);
    }

    [Fact]
    public void Evaluator_EmptyText_NoFindings()
    {
        var evaluator = new AdversarialRobustnessEvaluator<double>();
        var findings = evaluator.EvaluateText("");

        Assert.Empty(findings);
    }

    #endregion

    #region AdversarialImageEvaluator Tests

    [Fact]
    public void ImageEvaluator_RandomTensor_ProcessesWithoutError()
    {
        var evaluator = new AdversarialImageEvaluator<double>();
        var data = new double[3 * 32 * 32];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble();

        var tensor = new Tensor<double>(data, new[] { 3, 32, 32 });
        var findings = evaluator.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void ImageEvaluator_SmallTensor_HandlesGracefully()
    {
        var evaluator = new AdversarialImageEvaluator<double>();
        var data = new double[3 * 8 * 8];
        for (int i = 0; i < data.Length; i++) data[i] = 0.5;
        var tensor = new Tensor<double>(data, new[] { 3, 8, 8 });
        var findings = evaluator.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    [Fact]
    public void ImageEvaluator_CustomThreshold_Works()
    {
        var evaluator = new AdversarialImageEvaluator<double>(threshold: 0.3);
        var data = new double[3 * 16 * 16];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble();

        var tensor = new Tensor<double>(data, new[] { 3, 16, 16 });
        var findings = evaluator.EvaluateImage(tensor);

        Assert.NotNull(findings);
    }

    #endregion

    #region AdversarialPreferenceAlignment Tests

    [Fact]
    public void PreferenceAlignment_NonTrainableModel_ReturnsWrappedModel()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new AdversarialPreferenceAlignment<double>(options);

        var model = new MockPredictiveModel();
        var feedbackData = CreateMockFeedbackData(5, 4);

        var alignedModel = alignment.AlignModel(model, feedbackData);

        Assert.NotNull(alignedModel);
        // Non-trainable model should be wrapped (different instance)
        Assert.NotSame(model, alignedModel);
    }

    [Fact]
    public void PreferenceAlignment_TrainableModel_ModifiesInPlace()
    {
        var options = new AlignmentMethodOptions<double>
        {
            LearningRate = 0.01,
            TrainingIterations = 3
        };
        var alignment = new AdversarialPreferenceAlignment<double>(options);

        var model = new MockTrainableModel(inputDim: 4, outputDim: 4);
        var feedbackData = CreateMockFeedbackData(5, 4);

        var paramsBefore = CopyVector(model.GetParameters());
        var alignedModel = alignment.AlignModel(model, feedbackData);

        // Trainable model should be returned as-is (modified in place via gradients)
        Assert.Same(model, alignedModel);

        // Parameters should have changed
        var paramsAfter = model.GetParameters();
        bool anyChanged = false;
        for (int i = 0; i < paramsBefore.Length; i++)
        {
            if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-15)
            {
                anyChanged = true;
                break;
            }
        }

        Assert.True(anyChanged, "Gradient-based training should modify model parameters");
    }

    [Fact]
    public void PreferenceAlignment_ConstitutionalPrinciples_NonTrainable_ReturnsWrapped()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new AdversarialPreferenceAlignment<double>(options);

        var model = new MockPredictiveModel();
        var principles = new[] { "Be helpful", "Avoid harm", "Be honest" };

        var result = alignment.ApplyConstitutionalPrinciples(model, principles);

        Assert.NotNull(result);
        Assert.NotSame(model, result);
    }

    [Fact]
    public void PreferenceAlignment_ConstitutionalPrinciples_Trainable_ModifiesModel()
    {
        var options = new AlignmentMethodOptions<double>
        {
            LearningRate = 0.01
        };
        // Use larger perturbation budget so probe inputs produce outputs with higher variance
        var alignment = new AdversarialPreferenceAlignment<double>(options, perturbationBudget: 5.0);

        // Use larger weight scale so outputs have higher variance, triggering compliance checks
        var model = new MockTrainableModel(inputDim: 4, outputDim: 4, weightScale: 2.0);
        var principles = new[] { "Avoid harm", "Be safe" };

        var paramsBefore = CopyVector(model.GetParameters());
        var result = alignment.ApplyConstitutionalPrinciples(model, principles);

        Assert.Same(model, result);

        var paramsAfter = model.GetParameters();
        bool anyChanged = false;
        for (int i = 0; i < paramsBefore.Length; i++)
        {
            if (Math.Abs(paramsBefore[i] - paramsAfter[i]) > 1e-15)
            {
                anyChanged = true;
                break;
            }
        }

        Assert.True(anyChanged, "Constitutional training should modify model parameters");
    }

    [Fact]
    public void PreferenceAlignment_EvaluateAlignment_ProducesMetrics()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new AdversarialPreferenceAlignment<double>(options);

        var model = new MockPredictiveModel();
        var evalData = CreateMockEvaluationData(5, 4);

        var metrics = alignment.EvaluateAlignment(model, evalData);

        Assert.True(metrics.HelpfulnessScore >= 0 && metrics.HelpfulnessScore <= 1);
        Assert.True(metrics.HarmlessnessScore >= 0 && metrics.HarmlessnessScore <= 1);
        Assert.True(metrics.HonestyScore >= 0 && metrics.HonestyScore <= 1);
    }

    [Fact]
    public void PreferenceAlignment_RedTeaming_ProducesResults()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new AdversarialPreferenceAlignment<double>(options);

        // Train the reward model first
        var model = new MockPredictiveModel();
        var feedbackData = CreateMockFeedbackData(5, 4);
        alignment.AlignModel(model, feedbackData);

        // Now red-team
        var prompts = CreateRandomMatrix(3, 4);
        var results = alignment.PerformRedTeaming(model, prompts);

        Assert.NotNull(results);
        Assert.True(results.SuccessRate >= 0 && results.SuccessRate <= 1);
    }

    [Fact]
    public void PreferenceAlignment_SerializeDeserialize_Works()
    {
        var options = new AlignmentMethodOptions<double> { LearningRate = 0.001 };
        var alignment = new AdversarialPreferenceAlignment<double>(options);

        var bytes = alignment.Serialize();
        Assert.NotNull(bytes);
        Assert.True(bytes.Length > 0);

        var newAlignment = new AdversarialPreferenceAlignment<double>(
            new AlignmentMethodOptions<double>());
        newAlignment.Deserialize(bytes);

        Assert.Equal(options.LearningRate, newAlignment.GetOptions().LearningRate);
    }

    [Fact]
    public void PreferenceAlignment_Reset_ClearsRewardModel()
    {
        var options = new AlignmentMethodOptions<double>();
        var alignment = new AdversarialPreferenceAlignment<double>(options);

        var model = new MockPredictiveModel();
        var feedbackData = CreateMockFeedbackData(3, 4);
        alignment.AlignModel(model, feedbackData);

        alignment.Reset();

        // After reset, red teaming should use default reward (0.5)
        var prompts = CreateRandomMatrix(2, 4);
        var results = alignment.PerformRedTeaming(model, prompts);
        Assert.NotNull(results);
    }

    #endregion

    #region ViTAdversarialAttack Tests

    [Fact]
    public void ViTAttack_Instantiation_Works()
    {
        var options = new AdversarialAttackOptions<double>();
        var attack = new ViTAdversarialAttack<double, Vector<double>, Vector<double>>(options);

        Assert.NotNull(attack);
    }

    #endregion

    #region AdaptiveRandomizedSmoothing Tests

    [Fact]
    public void AdaptiveSmoothing_Instantiation_Works()
    {
        var options = new CertifiedDefenseOptions<double>();
        var defense = new AdaptiveRandomizedSmoothing<double, Vector<double>, Vector<double>>(options);

        Assert.NotNull(defense);
    }

    #endregion

    #region AdversarialPromptDefense Tests

    [Fact]
    public void PromptDefense_Instantiation_Works()
    {
        var options = new AdversarialDefenseOptions<double>();
        var defense = new AdversarialPromptDefense<double, Vector<double>, Vector<double>>(options);

        Assert.NotNull(defense);
    }

    #endregion

    #region Mock Classes

    private sealed class MockPredictiveModel : IPredictiveModel<double, Vector<double>, Vector<double>>
    {
        public Vector<double> Predict(Vector<double> input)
        {
            var result = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                result[i] = Math.Tanh(input[i]);
            }

            return new Vector<double>(result);
        }

        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    private sealed class MockTrainableModel
        : IPredictiveModel<double, Vector<double>, Vector<double>>,
          IGradientComputable<double, Vector<double>, Vector<double>>,
          IParameterizable<double, Vector<double>, Vector<double>>
    {
        private double[] _weights;
        private readonly int _inputDim;
        private readonly int _outputDim;

        public MockTrainableModel(int inputDim, int outputDim, double weightScale = 0.1)
        {
            _inputDim = inputDim;
            _outputDim = outputDim;
            _weights = new double[inputDim * outputDim];
            var rng = RandomHelper.CreateSeededRandom(42);
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] = (rng.NextDouble() - 0.5) * weightScale;
            }
        }

        public Vector<double> Predict(Vector<double> input)
        {
            var result = new double[_outputDim];
            for (int o = 0; o < _outputDim; o++)
            {
                double sum = 0;
                for (int i = 0; i < Math.Min(input.Length, _inputDim); i++)
                {
                    sum += input[i] * _weights[i * _outputDim + o];
                }

                result[o] = Math.Tanh(sum);
            }

            return new Vector<double>(result);
        }

        public Vector<double> ComputeGradients(
            Vector<double> input, Vector<double> target,
            AiDotNet.Interfaces.ILossFunction<double> lossFunction = null)
        {
            var output = Predict(input);
            var gradients = new double[_weights.Length];

            for (int o = 0; o < _outputDim; o++)
            {
                double error = output[o] - (o < target.Length ? target[o] : 0);
                double tanhDeriv = 1.0 - output[o] * output[o];

                for (int i = 0; i < Math.Min(input.Length, _inputDim); i++)
                {
                    gradients[i * _outputDim + o] = error * tanhDeriv * input[i];
                }
            }

            return new Vector<double>(gradients);
        }

        public void ApplyGradients(Vector<double> gradients, double learningRate)
        {
            for (int i = 0; i < Math.Min(gradients.Length, _weights.Length); i++)
            {
                _weights[i] -= learningRate * gradients[i];
            }
        }

        public Vector<double> GetParameters() => new Vector<double>((double[])_weights.Clone());

        public void SetParameters(Vector<double> parameters)
        {
            for (int i = 0; i < Math.Min(parameters.Length, _weights.Length); i++)
            {
                _weights[i] = parameters[i];
            }
        }

        public int ParameterCount => _weights.Length;

        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(
            Vector<double> parameters)
        {
            throw new NotSupportedException("Mock does not support WithParameters");
        }

        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
    }

    #endregion

    #region Helpers

    private static AlignmentFeedbackData<double> CreateMockFeedbackData(int numSamples, int dim)
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        var inputMatrix = new Matrix<double>(numSamples, dim);
        var outputMatrix = new Matrix<double>(numSamples, dim);

        for (int r = 0; r < numSamples; r++)
        {
            for (int c = 0; c < dim; c++)
            {
                inputMatrix[r, c] = (rng.NextDouble() - 0.5) * 2;
                outputMatrix[r, c] = (rng.NextDouble() - 0.5) * 2;
            }
        }

        return new AlignmentFeedbackData<double>
        {
            Inputs = inputMatrix,
            Outputs = outputMatrix,
            Preferences = numSamples >= 2
                ? new[] { (0, 1), (0, 2 < numSamples ? 2 : 1) }
                : Array.Empty<(int, int)>(),
            Ratings = Enumerable.Range(0, numSamples)
                .Select(_ => 0.5 + rng.NextDouble() * 0.5)
                .ToArray()
        };
    }

    private static AlignmentEvaluationData<double> CreateMockEvaluationData(int numSamples, int dim)
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        var inputMatrix = new Matrix<double>(numSamples, dim);
        var outputMatrix = new Matrix<double>(numSamples, dim);

        for (int r = 0; r < numSamples; r++)
        {
            for (int c = 0; c < dim; c++)
            {
                inputMatrix[r, c] = (rng.NextDouble() - 0.5) * 2;
                outputMatrix[r, c] = (rng.NextDouble() - 0.5) * 2;
            }
        }

        return new AlignmentEvaluationData<double>
        {
            TestInputs = inputMatrix,
            ExpectedOutputs = outputMatrix,
            ReferenceScores = Enumerable.Range(0, numSamples)
                .Select(_ => 0.5 + rng.NextDouble() * 0.5)
                .ToArray()
        };
    }

    private static Matrix<double> CreateRandomMatrix(int rows, int cols)
    {
        var rng = RandomHelper.CreateSeededRandom(42);
        var matrix = new Matrix<double>(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                matrix[r, c] = (rng.NextDouble() - 0.5) * 2;
            }
        }

        return matrix;
    }

    private static Vector<double> CopyVector(Vector<double> source)
    {
        var data = new double[source.Length];
        for (int i = 0; i < source.Length; i++)
        {
            data[i] = source[i];
        }

        return new Vector<double>(data);
    }

    #endregion
}
