using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.SyntheticData;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.SyntheticData;

/// <summary>
/// Verifies that <see cref="MedGANGenerator{T}"/> implements medGAN's actual mechanisms
/// (Choi et al., arXiv:1703.06490) rather than a generic GAN wearing its name.
/// </summary>
/// <remarks>
/// The paper names three efficiency devices — minibatch averaging, batch normalization and shortcut
/// connections — plus a two-stage structure in which a pre-trained decoder, not the generator,
/// produces the record. Each of those is asserted here directly. Without these tests the rebuild
/// would be asserted only by its comments, which is exactly the failure mode that made this model
/// need rebuilding.
/// </remarks>
public class MedGANMechanismTests
{
    private const int Width = 12;
    private const int Embedding = 8;

    private static NeuralNetworkArchitecture<double> Arch(int width) =>
        new(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: width, outputSize: width);

    private static MedGANOptions<double> Options(Action<MedGANOptions<double>>? tweak = null)
    {
        var o = new MedGANOptions<double>
        {
            Seed = 7,
            EmbeddingDimension = Embedding,
            GeneratorDimensions = [Embedding, Embedding],
            DiscriminatorDimensions = [16, 8],
            BatchSize = 8,
        };
        tweak?.Invoke(o);
        return o;
    }

    private static MedGANGenerator<double> Model(Action<MedGANOptions<double>>? tweak = null) =>
        new(Arch(Width), Options(tweak));

    private static Tensor<double> Batch(int rows, int cols, int seed = 3)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([rows, cols]);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble();
        return t;
    }

    [Fact]
    public void PaperHyperparametersAreTheDefaults()
    {
        var o = new MedGANOptions<double>();
        Assert.Equal(128, o.EmbeddingDimension);
        Assert.Equal([128, 128], o.GeneratorDimensions);
        Assert.Equal([256, 128], o.DiscriminatorDimensions);
        Assert.Equal(2, o.DiscriminatorSteps);          // "k = 2"
        Assert.Equal(1000, o.BatchSize);
        Assert.Equal(1000, o.Epochs);
        Assert.Equal(1e-3, o.LearningRate);
        Assert.Equal(0.99, o.BatchNormDecay);
        Assert.True(o.UseMinibatchAveraging);

        // The two additions that are NOT in the paper must be inert by default, so the default
        // configuration is medGAN and nothing else.
        Assert.False(o.EnablePrivacy);
        Assert.Equal(0.0, o.ConstraintWeight);
    }

    [Fact]
    public void ShortcutConnectionRequiresMatchingWidths_AndIsRejectedLoudly()
    {
        // x_k = ReLU(BN_k(W_k x_(k-1))) + x_(k-1) cannot be evaluated when the widths differ.
        // Silently dropping the shortcut would remove one of the paper's three contributions.
        var ex = Assert.Throws<ArgumentException>(() =>
            new MedGANGenerator<double>(Arch(Width), Options(o => o.GeneratorDimensions = [Embedding, Embedding + 1])));
        Assert.Contains("shortcut", ex.Message, StringComparison.OrdinalIgnoreCase);

        var ex2 = Assert.Throws<ArgumentException>(() =>
            new MedGANGenerator<double>(Arch(Width), Options(o => o.NoiseDimension = Embedding + 4)));
        Assert.Contains("shortcut", ex2.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void MinibatchAveraging_ConcatenatesTheBatchMeanOntoEverySample()
    {
        var model = Model();
        var records = Batch(rows: 5, cols: Width);

        var augmented = model.ApplyMinibatchAveraging(records);

        // Width doubles: [x_i ; xbar].
        Assert.Equal([5, Width * 2], augmented.Shape.ToArray());
        Assert.Equal(Width * 2, model.DiscriminatorInputWidth);

        for (int j = 0; j < Width; j++)
        {
            double expectedMean = 0.0;
            for (int i = 0; i < 5; i++) expectedMean += records[i, j];
            expectedMean /= 5;

            for (int i = 0; i < 5; i++)
            {
                Assert.Equal(records[i, j], augmented[i, j], 12);            // the sample itself
                Assert.Equal(expectedMean, augmented[i, Width + j], 12);     // the batch average
            }
        }
    }

    [Fact]
    public void MinibatchAveraging_IsComputedOverTheBatchPresent_NotAConstant()
    {
        // The mechanism only works because the average reflects THIS batch. If a collapsed
        // generator's batch produced the same companion vector as a diverse one, the discriminator
        // would learn nothing from it.
        var model = Model();

        var diverse = Batch(rows: 4, cols: Width, seed: 11);
        var collapsed = new Tensor<double>([4, Width]);
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < Width; j++) collapsed[i, j] = diverse[0, j];

        var a = model.ApplyMinibatchAveraging(diverse);
        var b = model.ApplyMinibatchAveraging(collapsed);

        bool differs = false;
        for (int j = 0; j < Width; j++)
        {
            if (Math.Abs(a[0, Width + j] - b[0, Width + j]) > 1e-9) { differs = true; break; }
        }
        Assert.True(differs,
            "A mode-collapsed batch must produce a different minibatch average than a diverse one; " +
            "that difference is the entire signal medGAN uses to detect mode collapse.");

        // A collapsed batch's average equals its (identical) rows exactly.
        for (int j = 0; j < Width; j++) Assert.Equal(collapsed[0, j], b[0, Width + j], 12);
    }

    [Fact]
    public void MinibatchAveraging_CanBeDisabled_AndThenTheDiscriminatorSeesTheRawWidth()
    {
        var model = Model(o => o.UseMinibatchAveraging = false);
        var records = Batch(rows: 5, cols: Width);

        Assert.Equal(Width, model.DiscriminatorInputWidth);
        Assert.Equal([5, Width], model.ApplyMinibatchAveraging(records).Shape.ToArray());
    }

    [Fact]
    public void Generator_MapsThePriorIntoTheEmbeddingSpace_NotTheRecordSpace()
    {
        // The defining structural choice: G never emits a record. It emits a latent point, and the
        // pre-trained decoder expands it. If G produced records directly this would be Width.
        var model = Model();
        var z = Batch(rows: 6, cols: model.NoiseDimension);

        var latent = model.GeneratorForwardBatched(z, isTraining: false);

        Assert.Equal([6, Embedding], latent.Shape.ToArray());
        Assert.Equal(Embedding, model.NoiseDimension);
    }

    [Fact]
    public void Synthesis_IsDecoderOfGenerator_NotDecoderOfNoise()
    {
        var model = Model();
        var z = Batch(rows: 6, cols: model.NoiseDimension);

        var viaComposition = model.DecoderForwardBatched(
            model.GeneratorForwardBatched(z, isTraining: false), applyOutputActivation: true);
        var synthesized = model.SynthesizeForDiscriminator(z, isTraining: false);

        Assert.Equal([6, Width], synthesized.Shape.ToArray());
        for (int i = 0; i < synthesized.Length; i++)
        {
            Assert.Equal(viaComposition[i], synthesized[i], 12);
        }

        // And Dec(G(z)) is NOT Dec(z): the generator is a real network in the path, not a no-op.
        var decoderOfNoiseDirectly = model.DecoderForwardBatched(z, applyOutputActivation: true);
        bool differs = Enumerable.Range(0, synthesized.Length)
            .Any(i => Math.Abs(synthesized[i] - decoderOfNoiseDirectly[i]) > 1e-9);
        Assert.True(differs, "Dec(G(z)) must differ from Dec(z) — otherwise G is not in the path.");
    }

    [Fact]
    public void Generator_ShortcutConnection_PassesItsInputThrough()
    {
        // With every parameter zeroed, each generator layer contributes ReLU(BN(0)) = 0, so the
        // shortcut addition alone determines the output: G(z) = z. Without the shortcut the output
        // would be 0. This distinguishes the two directly.
        var model = Model();
        var zeros = new Vector<double>(model.GetParameters().Length);
        model.UpdateParameters(zeros);

        var z = Batch(rows: 4, cols: model.NoiseDimension, seed: 21);
        var latent = model.GeneratorForwardBatched(z, isTraining: false);

        for (int i = 0; i < latent.Length; i++)
        {
            Assert.Equal(z[i], latent[i], 10);
        }
    }

    [Fact]
    public void Discriminator_HasNoBatchNormAndNoShortcuts_AndEmitsOneLogitPerSample()
    {
        // The paper states the discriminator has neither. A BatchNorm in the critic would make its
        // verdict depend on the batch through a SECOND channel besides minibatch averaging; a
        // shortcut would force its hidden widths to match its input width, which 256 -> 128 does not.
        var model = Model();
        var records = Batch(rows: 5, cols: Width);

        var scores = model.DiscriminatorForwardBatched(records);
        Assert.Equal([5, 1], scores.Shape.ToArray());

        // No BatchNorm: with parameters zeroed the logits are exactly zero. A BatchNorm would
        // normalize the zero pre-activations by their (zero) variance and produce something else.
        model.UpdateParameters(new Vector<double>(model.GetParameters().Length));
        var zeroed = model.DiscriminatorForwardBatched(records);
        for (int i = 0; i < zeroed.Length; i++) Assert.Equal(0.0, zeroed[i], 12);
    }

    [Fact]
    public void ReconstructionLoss_Binary_IsTheCrossEntropyOfEquation3()
    {
        // Eq. 3: sum(x log x' + (1-x) log(1-x')), which as a minimized loss over logits is
        // softplus(z) - x*z. Checked against a straight scalar evaluation of that identity.
        var model = Model(o => o.DataType = MedGANDataType.Binary);

        var logits = new Tensor<double>([2, 3]);
        double[] zv = [0.5, -1.25, 2.0, -0.75, 0.0, 3.5];
        for (int i = 0; i < zv.Length; i++) logits[i] = zv[i];
        var target = new Tensor<double>([2, 3]);
        double[] xv = [1, 0, 1, 1, 0, 0];
        for (int i = 0; i < xv.Length; i++) target[i] = xv[i];

        double expected = 0.0;
        for (int row = 0; row < 2; row++)
        {
            double rowSum = 0.0;
            for (int col = 0; col < 3; col++)
            {
                double z = zv[row * 3 + col], x = xv[row * 3 + col];
                double p = 1.0 / (1.0 + Math.Exp(-z));
                rowSum += -(x * Math.Log(p) + (1 - x) * Math.Log(1 - p));
            }
            expected += rowSum;
        }
        expected /= 2;   // sum over features, mean over batch

        var loss = model.ReconstructionLoss(logits, target);
        Assert.Equal(expected, loss[0], 9);
    }

    [Fact]
    public void ReconstructionLoss_Count_IsTheSquaredErrorOfEquation2()
    {
        var model = Model(o => o.DataType = MedGANDataType.Count);

        var logits = new Tensor<double>([2, 3]);
        double[] zv = [0.5, -1.25, 2.0, -0.75, 0.0, 3.5];
        for (int i = 0; i < zv.Length; i++) logits[i] = zv[i];
        var target = new Tensor<double>([2, 3]);
        double[] xv = [1, 0, 3, 2, 0, 1];
        for (int i = 0; i < xv.Length; i++) target[i] = xv[i];

        double expected = 0.0;
        for (int row = 0; row < 2; row++)
        {
            for (int col = 0; col < 3; col++)
            {
                double z = zv[row * 3 + col], x = xv[row * 3 + col];
                double recon = Math.Max(0.0, z);          // ReLU decoder for counts
                expected += (recon - x) * (recon - x);
            }
        }
        expected /= 2;

        var loss = model.ReconstructionLoss(logits, target);
        Assert.Equal(expected, loss[0], 9);
    }

    [Fact]
    public void AutoencoderActivations_FollowTheDataType()
    {
        // Binary: sigmoid decoder output, so every value lies in (0, 1).
        var binary = Model(o => o.DataType = MedGANDataType.Binary);
        var binaryOut = binary.DecoderForwardBatched(
            Batch(rows: 4, cols: Embedding, seed: 5), applyOutputActivation: true);
        for (int i = 0; i < binaryOut.Length; i++)
        {
            Assert.InRange(binaryOut[i], 0.0, 1.0);
        }

        // Count: ReLU decoder output, so nothing is negative.
        var count = Model(o => o.DataType = MedGANDataType.Count);
        var countOut = count.DecoderForwardBatched(
            Batch(rows: 4, cols: Embedding, seed: 5), applyOutputActivation: true);
        for (int i = 0; i < countOut.Length; i++)
        {
            Assert.True(countOut[i] >= 0.0, $"ReLU decoder produced {countOut[i]}");
        }
    }

    [Fact]
    public void FitRunsBothStages_AndGeneratesThroughTheDecoder()
    {
        var (data, columns) = SmallTable(rows: 40, cols: 4, seed: 13);
        var model = new MedGANGenerator<double>(Arch(4), Options(o =>
        {
            o.BatchSize = 20;
            o.VGMModes = 2;
            o.AutoencoderPretrainEpochs = 1;
        }));

        model.Fit(data, columns, epochs: 3);
        Assert.True(model.IsFitted);

        var generated = model.Generate(6);
        Assert.Equal(6, generated.Rows);
        Assert.Equal(4, generated.Columns);
        for (int i = 0; i < generated.Rows; i++)
        {
            for (int j = 0; j < generated.Columns; j++)
            {
                Assert.False(double.IsNaN(generated[i, j]) || double.IsInfinity(generated[i, j]),
                    $"Generated [{i},{j}] was {generated[i, j]}");
            }
        }
    }

    [Fact]
    public void SeedMakesGenerationReproducible()
    {
        var (data, columns) = SmallTable(rows: 40, cols: 4, seed: 13);

        Matrix<double> Run()
        {
            var m = new MedGANGenerator<double>(Arch(4), Options(o =>
            {
                o.BatchSize = 20;
                o.VGMModes = 2;
                o.AutoencoderPretrainEpochs = 1;
            }));
            m.Fit(data, columns, epochs: 2);
            return m.Generate(5);
        }

        var first = Run();
        var second = Run();
        for (int i = 0; i < first.Rows; i++)
        {
            for (int j = 0; j < first.Columns; j++) Assert.Equal(first[i, j], second[i, j], 10);
        }
    }

    private static (Matrix<double> Data, ColumnMetadata[] Columns) SmallTable(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var data = new Matrix<double>(rows, cols);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) data[i, j] = rng.NextDouble() * 10.0;
        }
        var columns = new ColumnMetadata[cols];
        for (int j = 0; j < cols; j++)
        {
            columns[j] = new ColumnMetadata($"c{j}", ColumnDataType.Continuous, columnIndex: j);
        }
        return (data, columns);
    }
}
