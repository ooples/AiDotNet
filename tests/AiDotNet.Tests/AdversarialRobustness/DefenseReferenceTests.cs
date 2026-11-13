using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.AdversarialRobustness;

/// <summary>
/// Reference tests for the input-transformation and prompt defenses: JPEG compression (Das et al. 2017,
/// ISO/IEC 10918-1), bit-depth reduction (Xu et al. 2018 feature squeezing) and visual prompting
/// (Chen et al. 2023).
/// </summary>
public class DefenseReferenceTests
{
    private sealed class LengthCheckingModel : IFullModel<double, Vector<double>, Vector<double>>
    {
        private readonly int _inputs;
        private List<int> _activeFeatures;

        public LengthCheckingModel(int inputs)
        {
            _inputs = inputs;
            _activeFeatures = Enumerable.Range(0, inputs).ToList();
        }

        public ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();

        public Vector<double> Predict(Vector<double> input)
        {
            Assert.Equal(_inputs, input.Length);
            double sum = input.ToArray().Sum();
            return new Vector<double>(new[] { sum, -sum });
        }

        public void Train(Vector<double> input, Vector<double> expectedOutput) { }
        public ModelMetadata<double> GetModelMetadata() => new();
        public byte[] Serialize() => Array.Empty<byte>();
        public void Deserialize(byte[] data) { }
        public void SaveModel(string filePath) { }
        public void LoadModel(string filePath) { }
        public void SaveState(Stream stream) { }
        public void LoadState(Stream stream) { }
        public IFullModel<double, Vector<double>, Vector<double>> Clone() => new LengthCheckingModel(_inputs);
        public IFullModel<double, Vector<double>, Vector<double>> DeepCopy() => new LengthCheckingModel(_inputs);
        public Vector<double> ComputeGradients(Vector<double> input, Vector<double> target, ILossFunction<double>? lossFunction = null) => new Vector<double>(_inputs);
        public void ApplyGradients(Vector<double> gradients, double learningRate) { }
        public Vector<double> GetParameters() => new Vector<double>(0);
        public void SetParameters(Vector<double> parameters) { }
        public IFullModel<double, Vector<double>, Vector<double>> WithParameters(Vector<double> parameters) => new LengthCheckingModel(_inputs);
        public long ParameterCount => 0;
        public bool SupportsParameterInitialization => false;
        public IEnumerable<int> GetActiveFeatureIndices() => _activeFeatures;
        public void SetActiveFeatureIndices(IEnumerable<int> featureIndices) => _activeFeatures = featureIndices.ToList();
        public bool IsFeatureUsed(int featureIndex) => _activeFeatures.Contains(featureIndex);
        public Dictionary<string, double> GetFeatureImportance() => Enumerable.Range(0, _inputs).ToDictionary(i => $"Feature{i}", i => 1.0 / _inputs);
        public Vector<double> SanitizeParameters(Vector<double> parameters) => parameters;
        public void Dispose() { }
    }

    [Fact(Timeout = 60000)]
    public async Task Jpeg_PreservesASmoothImage()
    {
        await Task.Yield();

        // A horizontal ramp from 0.2 to 0.8 over one 8x8 block. JPEG's table is defined for 8-bit samples
        // level-shifted by 128; quantizing [0, 1] intensities against it rounded every coefficient to zero and
        // flattened the block to its minimum (0.2). Real JPEG keeps a smooth ramp close to the original.
        var pixels = new double[64];
        for (int i = 0; i < 64; i++)
        {
            pixels[i] = 0.2 + 0.6 * (i % 8) / 7.0;
        }

        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(
            new AdversarialDefenseOptions<double> { UsePreprocessing = true, PreprocessingMethod = "JPEG" });
        var output = defense.PreprocessInput(new Vector<double>(pixels)).ToArray();

        Assert.Equal(64, output.Length);
        Assert.Equal(0.5, output.Average(), 0.02);
        Assert.True(output.Max() - output.Min() > 0.4);
        for (int i = 0; i < 64; i++)
        {
            Assert.Equal(pixels[i], output[i], 0.1);
        }
    }

    [Fact(Timeout = 60000)]
    public async Task BitDepthReduction_KeepsSixteenLevels()
    {
        await Task.Yield();

        // Xu et al. (2018): 4-bit depth maps x to round(15 x) / 15. Multiplying by 16 kept 17 levels.
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(
            new AdversarialDefenseOptions<double> { UsePreprocessing = true, PreprocessingMethod = "bit_depth_reduction" });
        var output = defense.PreprocessInput(new Vector<double>(new[] { 0.2, 0.0, 1.0, 0.52 })).ToArray();

        Assert.Equal(3.0 / 15, output[0], 1e-12);
        Assert.Equal(0.0, output[1], 1e-12);
        Assert.Equal(1.0, output[2], 1e-12);
        Assert.Equal(8.0 / 15, output[3], 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task PromptDefense_KeepsTheInputSize()
    {
        await Task.Yield();

        // Chen et al. (2023) add the visual prompt to the input; prepending it changed the length the frozen
        // model receives.
        var defense = new AdversarialPromptDefense<double, Vector<double>, Vector<double>>(
            new AdversarialDefenseOptions<double> { Epsilon = 0.05 }, promptLength: 4, promptOptimizationSteps: 1);
        var model = new LengthCheckingModel(10);
        var inputs = new[] { new Vector<double>(Enumerable.Repeat(0.5, 10).ToArray()) };
        var labels = new[] { new Vector<double>(new[] { 1.0, 0.0 }) };

        defense.ApplyDefense(inputs, labels, model);
        var defended = defense.PreprocessInput(inputs[0]);

        Assert.Equal(10, defended.Length);
        var delta = defended.ToArray().Select((v, i) => v - inputs[0][i]).ToArray();

        // The prompt frames the input: entries 0, 1 and 8, 9 move, the middle is untouched.
        for (int i = 2; i < 8; i++)
        {
            Assert.Equal(0.0, delta[i], 1e-12);
        }
    }
}
