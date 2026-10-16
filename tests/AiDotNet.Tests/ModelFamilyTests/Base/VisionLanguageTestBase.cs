using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using Xunit;
using System.Threading.Tasks;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// Base test class for vision-language models (CLIP, BLIP, LLaVA, etc.).
/// Inherits all NN invariant tests and adds VL-specific invariants:
/// image-only input, embedding diversity, bounded norms, and zero-image handling.
/// </summary>
public abstract class VisionLanguageTestBase<T> : NeuralNetworkModelTestBase<T>
{
    /// <summary>The caption the grounding supervision below names boxes for.</summary>
    protected const string GroundingCaption = "cat . remote control .";

    /// <summary>
    /// Synthetic grounding supervision for a model that trains on grounded phrases (<see cref="AiDotNet.VisionLanguage.Interfaces.IGroundingTrainable{T}"/>):
    /// the input image, <see cref="GroundingCaption"/>, and one box per phrase placed and sized by the target's first eight
    /// values (through tanh, so any target gives a box inside the image). A different target moves the boxes, and the same
    /// target gives the same supervision, so the memorization and target-dependence invariants have something to learn.
    /// </summary>
    protected static (Tensor<T> Image, IReadOnlyList<AiDotNet.VisionLanguage.Grounding.GroundingBox> Boxes) GroundingSupervision(
        Tensor<T> input, Tensor<T> target)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double U(int i) => target.Length == 0 ? 0.5 : 0.5 + 0.5 * Math.Tanh(ops.ToDouble(target[i % target.Length]));
        AiDotNet.VisionLanguage.Grounding.GroundingBox Box(string phrase, int at)
        {
            double cx = 0.25 + 0.5 * U(at), cy = 0.25 + 0.5 * U(at + 1), w = 0.1 + 0.3 * U(at + 2), h = 0.1 + 0.3 * U(at + 3);
            return new AiDotNet.VisionLanguage.Grounding.GroundingBox(phrase, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2);
        }
        var image = input.Rank == 4 && input.Shape[0] == 1 ? new Tensor<T>(input.Shape.ToArray().Skip(1).ToArray(), input.ToVector()) : input;
        return (image, new[] { Box("cat", 0), Box("remote control", 4) });
    }

    /// <summary>A grounding model trains on grounded phrases; every invariant's step goes through its typed entry point.</summary>
    protected override void TrainOn(INeuralNetworkModel<T> network, Tensor<T> input, Tensor<T> target)
    {
        if (network is AiDotNet.VisionLanguage.Interfaces.IGroundingTrainable<T> grounding)
        {
            var (image, boxes) = GroundingSupervision(input, target);
            grounding.TrainGrounding(image, GroundingCaption, boxes);
            return;
        }
        base.TrainOn(network, input, target);
    }

    /// <summary>A grounding model is measured on the loss it trains, with the same synthetic supervision.</summary>
    protected override double MeasureLoss(INeuralNetworkModel<T> network, Tensor<T> input, Tensor<T> output, Tensor<T> target)
    {
        if (network is AiDotNet.VisionLanguage.Interfaces.IGroundingTrainable<T> grounding)
        {
            var (image, boxes) = GroundingSupervision(input, target);
            return ConvertToDouble(grounding.EvaluateGroundingLoss(image, GroundingCaption, boxes));
        }
        return base.MeasureLoss(network, input, output, target);
    }

    /// <inheritdoc />
    protected override void RunRecalibrationPass(AiDotNet.NeuralNetworks.NeuralNetworkBase<T> network, Tensor<T> input, Tensor<T>? target)
    {
        if (target is not null && network is AiDotNet.VisionLanguage.Interfaces.IGroundingTrainable<T> grounding)
        {
            var (image, boxes) = GroundingSupervision(input, target);
            grounding.EvaluateGroundingLoss(image, GroundingCaption, boxes);
            return;
        }
        base.RunRecalibrationPass(network, input, target);
    }

    // =====================================================
    // VISION-LANGUAGE INVARIANT: Image Input → Finite Output
    // The model should produce finite output from image-only input.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ImageOnly_ShouldProduceOutput()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);
        Assert.True(output.Length > 0, "VL model produced empty output for image input.");
        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            Assert.True(!double.IsNaN(v) && !double.IsInfinity(v),
                $"Output[{i}] is not finite — VL model failed on image input.");
        }
    }

    // =====================================================
    // VISION-LANGUAGE INVARIANT: Different Images → Different Embeddings
    // Distinct visual inputs must produce distinct representations.
    // A model mapping everything to the same embedding is collapsed.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task DifferentImages_DifferentEmbeddings()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();

        var img1 = CreateConstantTensor(InputShape, 0.1);
        var img2 = CreateConstantTensor(InputShape, 0.9);

        var emb1 = network.Predict(img1);
        var emb2 = network.Predict(img2);

        bool anyDifferent = false;
        int minLen = Math.Min(emb1.Length, emb2.Length);
        for (int i = 0; i < minLen; i++)
        {
            if (Math.Abs(ConvertToDouble(emb1[i]) - ConvertToDouble(emb2[i])) > 1e-12)
            {
                anyDifferent = true;
                break;
            }
        }
        Assert.True(anyDifferent,
            "VL model produces identical embeddings for different images — representation has collapsed.");
    }

    // =====================================================
    // VISION-LANGUAGE INVARIANT: Output Norm Should Be Bounded
    // Embedding L2 norm should be reasonable. Extreme norms
    // indicate numerical instability or unbounded growth.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task OutputNorm_ShouldBeBounded()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var rng = ModelTestHelpers.CreateSeededRandom();
        var network = CreateNetwork();
        var input = CreateRandomTensor(InputShape, rng);

        var output = network.Predict(input);

        double normSq = 0;
        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            normSq += v * v;
        }
        double norm = Math.Sqrt(normSq);

        Assert.True(norm < 1e4,
            $"Embedding L2 norm = {norm:E4} exceeds bound of 1e4. " +
            "VL model embedding is not well-bounded.");
    }

    // =====================================================
    // VISION-LANGUAGE INVARIANT: Zero Image Should Not Crash
    // All-black image is a valid edge case.
    // =====================================================

    [Fact(Timeout = 120000)]
    public async Task ZeroImage_ShouldNotCrash()
    {
        await Task.Yield();
        using var _arena = TensorArena.Create();
        var network = CreateNetwork();
        var blackImage = CreateConstantTensor(InputShape, 0.0);

        var output = network.Predict(blackImage);
        Assert.True(output.Length > 0, "VL model produced empty output for black image.");
        for (int i = 0; i < output.Length; i++)
        {
            double v = ConvertToDouble(output[i]);
            Assert.True(!double.IsNaN(v) && !double.IsInfinity(v),
                $"Output[{i}] is not finite for all-zero image input.");
        }
    }
}

/// <summary>
/// Double-precision binding of <see cref="VisionLanguageTestBase{T}"/>. Existing
/// VL test classes extend this non-generic name and run in double unchanged; large
/// (&gt;1B-param) VL models that cannot fit a double forward in the CPU test budget
/// extend <c>VisionLanguageTestBase&lt;float&gt;</c> instead.
/// </summary>
public abstract class VisionLanguageTestBase : VisionLanguageTestBase<double> { }
