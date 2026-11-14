using AiDotNet.AdversarialRobustness.Attacks;
using AiDotNet.AdversarialRobustness.Defenses;
using AiDotNet.AdversarialRobustness.Safety;
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
    [Fact(Timeout = 60000)]
    public async Task HalfPreferredCentroid_DoesNotOverflowItsSum()
    {
        await Task.Yield();

        // Three preferred rows of (30000, 0.5): the centroid is (30000, 0.5), but the running sum of the first
        // entry (90000) passed Half's 65504 and the centroid became infinite.
        var outputs = new Matrix<Half>(4, 2);
        for (int r = 0; r < 4; r++)
        {
            outputs[r, 0] = (Half)(r < 3 ? 30000.0 : 0.0);
            outputs[r, 1] = (Half)0.5;
        }

        var feedback = new AiDotNet.Models.AlignmentFeedbackData<Half>
        {
            Outputs = outputs,
            Preferences = new[] { (0, 3), (1, 3), (2, 3) }
        };

        var centroid = AdversarialPreferenceAlignment<Half>.PreferredCentroid(feedback);

        Assert.Equal(30000.0, (double)centroid[0], 0);
        Assert.Equal(0.5, (double)centroid[1], 6);
    }

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
            new AdversarialDefenseOptions<double> { Epsilon = 0.05, Seed = 3 }, promptLength: 4, promptOptimizationSteps: 1);
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

        // A PreprocessInput that returned the input unchanged would pass everything above.
        Assert.Contains(new[] { 0, 1, 8, 9 }, i => Math.Abs(delta[i]) > 0.0);
    }

    [Fact(Timeout = 60000)]
    public async Task Jpeg_ShortInputKeepsItsObservedRangeLikeThe2DPath()
    {
        await Task.Yield();

        // Five values have no 8 x 8 factorisation, so JPEG falls back to the 1D DCT path. The 2D path clamps
        // its reconstruction to the observed [min, max] of the input; clamping the fallback to [0, 1] instead
        // saturated -0.5 to 0 and 1.5 to 1, so the output range depended on the input length.
        var input = new[] { -0.5, 0.0, 0.5, 1.0, 1.5 };
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(
            new AdversarialDefenseOptions<double> { UsePreprocessing = true, PreprocessingMethod = "JPEG" });
        var output = defense.PreprocessInput(new Vector<double>(input)).ToArray();

        Assert.Equal(5, output.Length);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.InRange(output[i], -0.5, 1.5);
            Assert.Equal(input[i], output[i], 0.15);
        }

        Assert.True(output.Min() < -0.3, $"min {output.Min()}");
        Assert.True(output.Max() > 1.3, $"max {output.Max()}");
    }

    /// <summary>Exposes the character-frequency features of the content classifier.</summary>
    private sealed class FrequencyProbe<TNum> : RuleBasedContentClassifier<TNum>
    {
        public Vector<TNum> Frequencies(string text) => TextToVector(text);
    }

    [Fact(Timeout = 60000)]
    public async Task ContentClassifier_HalfFrequenciesAreDividedBeforeTheCast()
    {
        await Task.Yield();

        // 2049 'a' and one 'b': the frequency of 'a' is 2049 / 2050, which Half holds as 1 - 2^-11. Casting the
        // counts first made it Half(2048) / Half(2050) = 1 - 2^-10, and a text longer than 65504 divided by
        // infinity.
        var probe = new FrequencyProbe<Half>();
        var frequencies = probe.Frequencies(new string('a', 2049) + "b");
        Assert.Equal((double)(Half)(2049.0 / 2050.0), (double)frequencies['a'], 1e-12);

        var longText = probe.Frequencies(new string('a', 70000));
        Assert.Equal(1.0, (double)longText['a'], 1e-12);
    }

    [Fact(Timeout = 120000)]
    public async Task AdversarialTraining_EvaluateRobustnessPropagatesAFailingAttack()
    {
        await Task.Yield();

        // A targeted AutoAttack naming class 5 of a two-class model throws. Counting the failure as a successful
        // defense reported an adversarial accuracy of 1.
        var defense = new AdversarialTraining<double, Vector<double>, Vector<double>>(
            new AdversarialDefenseOptions<double> { UsePreprocessing = false });
        var attack = new AutoAttack<double, Vector<double>, Vector<double>>(
            new AdversarialAttackOptions<double> { Epsilon = 0.1, IsTargeted = true, TargetClass = 5 });
        var model = new LengthCheckingModel(2);
        var inputs = new[] { new Vector<double>(new[] { 0.5, 0.25 }) };
        var labels = new[] { new Vector<double>(new[] { 1.0, 0.0 }) };

        Assert.Throws<ArgumentOutOfRangeException>(() => defense.EvaluateRobustness(model, inputs, labels, attack));
    }

    [Fact(Timeout = 120000)]
    public async Task PromptDefense_UsesTheOptionsSeed()
    {
        await Task.Yield();

        // The prompt initialization and mini-batches came from a fixed seed (42) whatever the options said.
        double[] Prompt(int seed)
        {
            var defense = new AdversarialPromptDefense<double, Vector<double>, Vector<double>>(
                new AdversarialDefenseOptions<double> { Epsilon = 0.05, Seed = seed }, promptLength: 4, promptOptimizationSteps: 1);
            var inputs = new[] { new Vector<double>(new double[10]) };
            var labels = new[] { new Vector<double>(new[] { 1.0, 0.0 }) };
            defense.ApplyDefense(inputs, labels, new LengthCheckingModel(10));
            return defense.PreprocessInput(inputs[0]).ToArray();
        }

        Assert.Equal(Prompt(1), Prompt(1));
        Assert.NotEqual(Prompt(1), Prompt(2));
    }
}
