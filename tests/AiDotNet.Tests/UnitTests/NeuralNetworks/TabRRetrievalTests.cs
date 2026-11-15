using System;
using System.Linq;
using AiDotNet.Distributions;
using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Tabular;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// TabR's retrieval module (Gorishniy et al. 2024): L2 key similarity, label-conditioned values, leave-one-out
/// candidates, the tape training step, and the neighbour-spread confidence interval.
/// </summary>
public class TabRRetrievalTests
{
    private static TabROptions<double> SmallOptions(int neighbors) => new TabROptions<double>
    {
        // The reference architecture at its smallest: one encoder block and one predictor block (the old fixture's
        // one encoder layer and one context layer), d_block = d_main, and no normalization layers (the old
        // UseLayerNorm = false).
        EmbeddingDimension = 2,
        BlockWidthMultiplier = 1.0,
        EncoderNumBlocks = 1,
        PredictorNumBlocks = 1,
        NumNeighbors = neighbors,
        Normalization = TabRNormalization.Identity,
        NormalizeEmbeddings = false,
        DropoutRate = 0,
        ContextDropoutRate = 0
    };

    private static Tensor<double> Features() =>
        new Tensor<double>(new[] { 3, 2 }, new Vector<double>(new[] { 0.4, 0.7, 1.1, 0.2, 0.6, 0.9 }));

    private static void SetDistinctParameters(TabRBase<double> model)
    {
        // Positive, distinct values keep the ReLUs away from their kink and break attention symmetry.
        var initial = model.GetParameters();
        for (int i = 0; i < initial.Length; i++) initial[i] = 0.3 + 0.04 * (i % 7);
        model.SetParameters(initial);
    }

    [Fact]
    public void LeaveOneOut_NeverRetrievesTheTargetItself()
    {
        var model = new TabRRegression<double>(2, 1, SmallOptions(neighbors: 3));
        var features = Features();
        model.BuildIndex(features, new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.1, 0.5, 0.9 })));

        _ = model.Forward(features, new Vector<int>(new[] { 0, 1, 2 }));

        // Three candidates, one excluded per row: k = 2, and row b never holds b.
        var neighbors = model.GetRetrievedNeighborIndices();
        Assert.NotNull(neighbors);
        Assert.Equal(2, neighbors.Columns);
        for (int b = 0; b < 3; b++)
        {
            Assert.NotEqual(b, neighbors[b, 0]);
            Assert.NotEqual(b, neighbors[b, 1]);
            Assert.NotEqual(neighbors[b, 0], neighbors[b, 1]);
        }
    }

    [Fact]
    public void NeighbourLabels_ChangeThePrediction()
    {
        var features = Features();
        var model = new TabRClassifier<double>(2, 2, SmallOptions(neighbors: 2));
        SetDistinctParameters(model);

        model.BuildIndex(features, new[] { 0, 1, 1 });
        var withLabels = model.PredictProbabilities(features).ToArray();
        model.BuildIndex(features, new[] { 1, 0, 0 });
        var withFlippedLabels = model.PredictProbabilities(features).ToArray();
        model.BuildIndex(features);
        var withoutLabels = model.PredictProbabilities(features).ToArray();

        Assert.False(withLabels.SequenceEqual(withFlippedLabels), "The label embedding W_Y(y_i) must reach the output.");
        Assert.False(withLabels.SequenceEqual(withoutLabels), "Indexing labels must add the label term.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TrainStep_WholeModelUpdateMatchesFiniteDifferences(bool classification)
    {
        var features = Features();
        var sampleIndices = new Vector<int>(new[] { 0, 1, 2 });
        var targets = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.3, 0.8, 0.5 }));
        var labels = new[] { 0, 1, 1 };

        // Every row retrieves both other samples, so the neighbour set cannot change under a perturbation.
        TabRBase<double> model = classification
            ? new TabRClassifier<double>(2, 2, SmallOptions(neighbors: 2))
            : new TabRRegression<double>(2, 1, SmallOptions(neighbors: 2));
        if (model is TabRClassifier<double> indexedClassifier) indexedClassifier.BuildIndex(features, labels);
        else ((TabRRegression<double>)model).BuildIndex(features, targets);
        SetDistinctParameters(model);
        var initial = model.GetParameters();

        double Loss() => model is TabRClassifier<double> classifier
            ? classifier.ComputeCrossEntropyLoss(
                Softmax(classifier.Forward(features, sampleIndices)), labels)
            : ((TabRRegression<double>)model).ComputeMSELoss(
                ((TabRRegression<double>)model).Forward(features, sampleIndices), targets);

        double beforeLoss = Loss();
        const double epsilon = 1e-5;
        var numerical = new double[initial.Length];
        for (int i = 0; i < initial.Length; i++)
        {
            var perturbed = new Vector<double>(initial.ToArray());
            perturbed[i] += epsilon;
            model.SetParameters(perturbed);
            double plus = Loss();
            perturbed[i] -= 2 * epsilon;
            model.SetParameters(perturbed);
            numerical[i] = (plus - Loss()) / (2 * epsilon);
        }

        model.SetParameters(initial);
        const double rate = 0.01;
        double reported = model is TabRClassifier<double> c
            ? c.TrainStep(features, labels, rate, sampleIndices)
            : ((TabRRegression<double>)model).TrainStep(features, targets, rate, sampleIndices);
        Assert.Equal(beforeLoss, reported, 10);

        var updated = model.GetParameters();
        for (int i = 0; i < initial.Length; i++)
        {
            double actual = (initial[i] - updated[i]) / rate;
            Assert.True(Math.Abs(numerical[i] - actual) < 1e-6 + 1e-4 * Math.Abs(numerical[i]),
                $"Parameter {i}: finite difference {numerical[i]:R}, training gradient {actual:R}.");
        }

        Assert.True(Loss() < beforeLoss, "One SGD step should lower this smooth objective.");
    }

    [Fact]
    public void PredictWithConfidence_IsTheRetrievalWeightedSpreadOfNeighbourTargets()
    {
        // Two candidates with targets 0 and 2, both retrieved: var = w0 (0 - m)^2 + w1 (2 - m)^2 with
        // m = 2 w1, which is 4 w0 w1; the 95% half-width is z * 2 sqrt(w0 w1) with z the two-sided normal
        // quantile. The model takes z from the library's shared normal inverse CDF, the Abramowitz and Stegun
        // 26.2.23 rational approximation (absolute error below 4.5e-4), so z is held to the exact
        // 1.959963984540054 within that bound and the interval to 9 decimals with the quantile the library gives.
        var index = new Tensor<double>(new[] { 2, 2 }, new Vector<double>(new[] { 0.4, 0.7, 1.1, 0.2 }));
        var model = new TabRRegression<double>(2, 1, SmallOptions(neighbors: 2));
        model.BuildIndex(index, new Tensor<double>(new[] { 2, 1 }, new Vector<double>(new[] { 0.0, 2.0 })));
        SetDistinctParameters(model);

        var query = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.6, 0.9 }));
        var (predictions, lower, upper) = model.PredictWithConfidence(query, 0.95);

        var weights = model.GetAttentionWeights();
        Assert.NotNull(weights);
        double w0 = weights[0];
        double w1 = weights[1];
        Assert.Equal(1.0, w0 + w1, 12);
        double z = new NormalDistribution<double>().InverseCdf(0.975);
        Assert.InRange(z, 1.959963984540054 - 4.5e-4, 1.959963984540054 + 4.5e-4);
        double halfWidth = z * 2.0 * Math.Sqrt(w0 * w1);
        Assert.True(halfWidth > 0.1, "Both neighbours must carry weight for the interval to test anything.");
        Assert.Equal(predictions[0] - halfWidth, lower[0], 9);
        Assert.Equal(predictions[0] + halfWidth, upper[0], 9);
    }

    [Fact]
    public void PredictWithConfidence_IsZeroWidthWhenNeighboursAgree()
    {
        var model = new TabRRegression<double>(2, 1, SmallOptions(neighbors: 3));
        model.BuildIndex(Features(), new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 1.5, 1.5, 1.5 })));

        var query = new Tensor<double>(new[] { 1, 2 }, new Vector<double>(new[] { 0.6, 0.9 }));
        var (predictions, lower, upper) = model.PredictWithConfidence(query);

        Assert.Equal(predictions[0], lower[0], 12);
        Assert.Equal(predictions[0], upper[0], 12);
    }

    [Fact]
    public void PredictWithConfidence_WithoutIndexedTargets_Throws()
    {
        var model = new TabRRegression<double>(2, 1, SmallOptions(neighbors: 2));
        model.BuildIndex(Features());

        Assert.Throws<InvalidOperationException>(() => model.PredictWithConfidence(Features()));
    }

    [Fact]
    public void Architecture_MatchesTheReferenceModelLayerForLayer()
    {
        // tabr.py Model with d_in = 3, d_main = 4, d_multiplier = 1.5 (d_block = int(6.0) = 6), encoder_n_blocks = 2,
        // predictor_n_blocks = 1, mixer_normalization 'auto' (on, since encoder_n_blocks > 0), LayerNorm, 3 classes:
        //   linear          Linear(3, 4)                               3*4 + 4             = 16
        //   blocks0[0]      Linear(4, 6), Linear(6, 4)  (prenorm off)  (24 + 6) + (24 + 4) = 58
        //   blocks0[1]      LayerNorm(4), Linear(4, 6), Linear(6, 4)   8 + 58              = 66
        //   normalization   LayerNorm(4)                                                   = 8
        //   K               Linear(4, 4)                               16 + 4              = 20
        //   T               Linear(4, 6), Linear(6, 4, bias=False)     30 + 24             = 54
        //   label_encoder   Embedding(3, 4)                            3*4                 = 12
        //   blocks1[0]      LayerNorm(4), Linear(4, 6), Linear(6, 4)                       = 66
        //   head            LayerNorm(4), ReLU, Linear(4, 3)           8 + (12 + 3)        = 23
        //                                                                          total   = 323
        var options = new TabROptions<double>
        {
            EmbeddingDimension = 4,
            BlockWidthMultiplier = 1.5,
            EncoderNumBlocks = 2,
            PredictorNumBlocks = 1,
            NumNeighbors = 2
        };
        var model = new TabRClassifier<double>(3, 3, options);

        Assert.Equal(6, model.BlockDimension);
        AssertLinear(model.InputProjection, 3, 4, bias: true);

        Assert.Equal(2, model.EncoderBlocks.Count);
        Assert.Null(model.EncoderBlocks[0].Normalization);
        AssertBlock(model.EncoderBlocks[0]);
        Assert.IsType<LayerNormalizationLayer<double>>(model.EncoderBlocks[1].Normalization);
        AssertBlock(model.EncoderBlocks[1]);

        Assert.IsType<LayerNormalizationLayer<double>>(model.MixerNormalizationLayer);
        AssertLinear(model.KeyProjection, 4, 4, bias: true);
        AssertLinear(model.ValueHiddenProjection, 4, 6, bias: true);
        AssertLinear(model.ValueOutputProjection, 6, 4, bias: false);
        Assert.NotNull(model.LabelEmbedding);
        AssertLinear(model.LabelEmbedding, 3, 4, bias: false);

        Assert.Single(model.PredictorBlocks);
        Assert.IsType<LayerNormalizationLayer<double>>(model.PredictorBlocks[0].Normalization);
        AssertBlock(model.PredictorBlocks[0]);
        Assert.IsType<LayerNormalizationLayer<double>>(model.HeadNormalization);

        Assert.Equal(323L, model.ParameterCount);
        Assert.Equal(323, model.GetParameters().Length);

        static void AssertBlock(TabRResidualBlock<double> block)
        {
            AssertLinear(block.Expand, 4, 6, bias: true);
            AssertLinear(block.Project, 6, 4, bias: true);
        }
    }

    [Fact]
    public void DefaultOptions_AreTheTabRSConfigurationWithItsReferenceParameterCount()
    {
        // Gorishniy et al. (2023), default TabR-S: d_main = 265, d_block = 2 d_main = 530, encoder_n_blocks = 0,
        // predictor_n_blocks = 1, mixer normalization off ('auto' with no encoder blocks), LayerNorm, ReLU,
        // dropout1 = 0. For a 5-feature, single-output regressor:
        //   linear          Linear(5, 265)                                  1325 + 265            = 1590
        //   K               Linear(265, 265)                                70225 + 265           = 70490
        //   T               Linear(265, 530), Linear(530, 265, bias=False)  140980 + 140450       = 281430
        //   label_encoder   Linear(1, 265)                                  265 + 265             = 530
        //   blocks1[0]      LayerNorm(265), Linear(265, 530), Linear(530, 265)  530 + 140980 + 140715 = 282225
        //   head            LayerNorm(265), ReLU, Linear(265, 1)            530 + 266             = 796
        //                                                                                 total   = 637061
        var options = new TabROptions<double>();
        Assert.Equal(265, options.EmbeddingDimension);
        Assert.Equal(2.0, options.BlockWidthMultiplier);
        Assert.Equal(0, options.EncoderNumBlocks);
        Assert.Equal(1, options.PredictorNumBlocks);
        Assert.Null(options.MixerNormalization);
        Assert.Equal(TabRNormalization.LayerNorm, options.Normalization);
        Assert.Equal(ActivationFunction.ReLU, options.Activation);
        Assert.Equal(0.0, options.BlockOutputDropoutRate);

        var model = new TabRRegression<double>(5, 1, options);

        Assert.Equal(530, model.BlockDimension);
        AssertLinear(model.InputProjection, 5, 265, bias: true);
        Assert.Empty(model.EncoderBlocks);
        Assert.Null(model.MixerNormalizationLayer);
        Assert.NotNull(model.LabelEmbedding);
        AssertLinear(model.LabelEmbedding, 1, 265, bias: true);
        Assert.Single(model.PredictorBlocks);
        Assert.Equal(637061L, model.ParameterCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ResidualBlock_MatchesTheReferenceMakeBlock(bool prenorm)
    {
        // make_block(prenorm) run as x = x + block(x): x + Linear(d_block, d_main)(ReLU(Linear(d_main, d_block)(n)))
        // with n = LayerNorm(x) (eps 1e-5, unit gain, zero shift) when prenorm and n = x otherwise. Both dropouts are
        // set but inactive, since this is not a training step. The encoder's first block is the prenorm = False one;
        // every predictor block is prenorm = True.
        const int dMain = 3;
        const int dBlock = 6;
        var options = new TabROptions<double>
        {
            EmbeddingDimension = dMain,
            BlockWidthMultiplier = 2.0,
            EncoderNumBlocks = 1,
            PredictorNumBlocks = 1,
            NumNeighbors = 2,
            DropoutRate = 0.5,
            BlockOutputDropoutRate = 0.5
        };
        var model = new TabRRegression<double>(2, 1, options);
        var block = prenorm ? model.PredictorBlocks[0] : model.EncoderBlocks[0];
        Assert.Equal(prenorm, block.Normalization is not null);

        var w1 = new double[dBlock, dMain];
        var b1 = new double[dBlock];
        var w2 = new double[dMain, dBlock];
        var b2 = new double[dMain];
        for (int j = 0; j < dBlock; j++)
        {
            b1[j] = 0.05 * (j - 2);
            for (int i = 0; i < dMain; i++) w1[j, i] = 0.3 * Math.Sin(1.7 * j + 0.9 * i + 0.4);
        }

        for (int i = 0; i < dMain; i++)
        {
            b2[i] = 0.1 * (i + 1) - 0.15;
            for (int j = 0; j < dBlock; j++) w2[i, j] = 0.25 * Math.Cos(1.1 * i + 0.6 * j + 0.2);
        }

        SetLinear(block.Expand, w1, b1);
        SetLinear(block.Project, w2, b2);

        double[,] x = { { 0.7, -1.2, 0.4 }, { 1.5, 0.3, -0.8 } };
        var input = new Tensor<double>(new[] { 2, dMain }, new Vector<double>(x.Cast<double>().ToArray()));
        var actual = model.ForwardResidualBlock(block, input).ToArray();

        Assert.Equal(2 * dMain, actual.Length);
        bool clipped = false;
        for (int r = 0; r < 2; r++)
        {
            var n = new double[dMain];
            for (int i = 0; i < dMain; i++) n[i] = x[r, i];
            if (prenorm)
            {
                double mean = n.Average();
                double variance = n.Select(v => (v - mean) * (v - mean)).Average();
                for (int i = 0; i < dMain; i++) n[i] = (n[i] - mean) / Math.Sqrt(variance + 1e-5);
            }

            var hidden = new double[dBlock];
            for (int j = 0; j < dBlock; j++)
            {
                double pre = b1[j];
                for (int i = 0; i < dMain; i++) pre += w1[j, i] * n[i];
                clipped |= pre < 0;
                hidden[j] = Math.Max(0.0, pre);
            }

            for (int i = 0; i < dMain; i++)
            {
                double expected = x[r, i] + b2[i];
                for (int j = 0; j < dBlock; j++) expected += w2[i, j] * hidden[j];
                Assert.Equal(expected, actual[r * dMain + i], 12);
            }
        }

        Assert.True(clipped, "Some pre-activation must be negative so the reference's ReLU is exercised.");
    }

    [Fact]
    public void BlockOutputDropout_IsATrainingStepMaskDrawnThroughCompiledStepRandom()
    {
        // tabr.py make_block ends in Dropout(dropout1), before the residual add. With dropout0 and context_dropout at
        // zero, a training step draws exactly one dropout1 mask per block application: the query encoder block, the
        // candidate encoder block and the predictor block. A prediction draws none.
        var options = SmallOptions(neighbors: 2);
        options.EmbeddingDimension = 4;
        options.BlockOutputDropoutRate = 0.25;
        var model = new TabRRegression<double>(2, 1, options);
        var features = Features();
        var targets = new Tensor<double>(new[] { 3, 1 }, new Vector<double>(new[] { 0.1, 0.5, 0.9 }));
        model.BuildIndex(features, targets);

        int[][] stepShapes;
        using (var recording = AiDotNet.Training.CompiledStepRandom<double>.BeginRecording())
        {
            _ = model.TrainStep(features, targets, 0.01, new Vector<int>(new[] { 0, 1, 2 }));
            stepShapes = recording.Draws.Select(draw => draw.Tensor.Shape.ToArray()).ToArray();
        }

        int predictDraws;
        using (var recording = AiDotNet.Training.CompiledStepRandom<double>.BeginRecording())
        {
            _ = model.Predict(features);
            predictDraws = recording.Draws.Count;
        }

        Assert.Equal(new[] { new[] { 3, 4 }, new[] { 3, 4 }, new[] { 3, 4 } }, stepShapes);
        Assert.Equal(0, predictDraws);
    }

    [Fact]
    public void MixerNormalization_AutoIsOnExactlyWhenTheEncoderHasBlocks()
    {
        // tabr.py: mixer_normalization = encoder_n_blocks > 0 for 'auto', and assert not mixer_normalization when
        // encoder_n_blocks == 0.
        TabROptions<double> Options(int encoderBlocks, bool? mixer) => new TabROptions<double>
        {
            EmbeddingDimension = 4,
            EncoderNumBlocks = encoderBlocks,
            MixerNormalization = mixer,
            NumNeighbors = 2
        };

        Assert.Null(new TabRRegression<double>(2, 1, Options(0, null)).MixerNormalizationLayer);
        Assert.IsType<LayerNormalizationLayer<double>>(new TabRRegression<double>(2, 1, Options(1, null)).MixerNormalizationLayer);
        Assert.Null(new TabRRegression<double>(2, 1, Options(1, false)).MixerNormalizationLayer);
        Assert.Throws<ArgumentException>(() => new TabRRegression<double>(2, 1, Options(0, true)));
    }

    [Fact]
    public void LabelEncoder_IsTheReferenceEmbeddingForClassesAndLinearForTargets()
    {
        // tabr.py: nn.Embedding(n_classes, d_main) initialized uniform on [-1, 1] for classes (bias-free over a one-hot
        // label), nn.Linear(1, d_main) with weight and bias uniform on [-1/sqrt(2), 1/sqrt(2)] for regression. The
        // matrices are large enough that the bounds are reached, not just respected: 4 x 256 = 1024 draws on [-1, 1] all
        // stay inside 0.9 with probability 0.9^1024 (about 1e-47), and 2 x 256 = 512 draws on +-1/sqrt(2) all stay
        // inside 0.6 with probability 0.85^512 (about 1e-36), so a +-1/sqrt(2) class init or a narrower target init
        // fails for any seed.
        var classifier = new TabRClassifier<double>(2, 4, new TabROptions<double> { EmbeddingDimension = 256, NumNeighbors = 2 });
        var embedding = classifier.LabelEmbedding;
        Assert.NotNull(embedding);
        AssertLinear(embedding, 4, 256, bias: false);
        var classWeights = embedding.GetParameters().ToArray();
        Assert.All(classWeights, w => Assert.InRange(w, -1.0, 1.0));
        Assert.True(classWeights.Max(v => Math.Abs(v)) > 0.9, "The class embedding must be drawn from [-1, 1], not +-1/sqrt(2).");

        var regressor = new TabRRegression<double>(2, 1, new TabROptions<double> { EmbeddingDimension = 256, NumNeighbors = 2 });
        var linear = regressor.LabelEmbedding;
        Assert.NotNull(linear);
        AssertLinear(linear, 1, 256, bias: true);
        double bound = 1.0 / Math.Sqrt(2.0);
        var targetWeights = linear.GetParameters().ToArray();
        Assert.All(targetWeights, w => Assert.InRange(w, -bound, bound));
        Assert.True(targetWeights.Max(v => Math.Abs(v)) > 0.6, "The target encoder must be drawn from +-1/sqrt(2).");
    }

    [Theory]
    [InlineData(true, 57L, 556L)]
    [InlineData(false, 147L, 646L)]
    public void NumericEmbedding_PlrMatchesTheReferenceShapesAndParameterCount(bool lite, long embeddingCount, long total)
    {
        // lib.deep.PLREmbeddings(n_features = 3, n_frequencies = 4, frequency_scale = 0.01, d_embedding = 5, lite)
        // feeding a regressor with d_main = 6, d_block = 12, no encoder blocks and one predictor block:
        //   frequencies     [3, 4]                                                       12
        //   lite            Linear(8, 5)                     40 + 5 = 45    -> embedding 57
        //   not lite        NLinear(3, 8, 5)                 120 + 15 = 135 -> embedding 147
        //   linear          Linear(3 * 5, 6)                 90 + 6                    = 96
        //   K               Linear(6, 6)                     36 + 6                    = 42
        //   T               Linear(6, 12), Linear(12, 6, bias=False)   84 + 72         = 156
        //   label_encoder   Linear(1, 6)                     6 + 6                     = 12
        //   blocks1[0]      LayerNorm(6), Linear(6, 12), Linear(12, 6)  12 + 84 + 78   = 174
        //   head            LayerNorm(6), ReLU, Linear(6, 1)            12 + 7         = 19
        //   total: 57 + 499 = 556 (lite), 147 + 499 = 646 (not lite)
        var options = new TabROptions<double>
        {
            EmbeddingDimension = 6,
            NumNeighbors = 2,
            NumericEmbedding = TabRNumericEmbedding.PeriodicLinearReLU,
            NumFrequencies = 4,
            FrequencyInitScale = 0.01,
            NumericEmbeddingDimension = 5,
            LiteNumericEmbedding = lite
        };
        var model = new TabRRegression<double>(3, 1, options);

        var embedding = model.NumericEmbeddingModule;
        Assert.NotNull(embedding);
        Assert.Equal(new[] { 3, 4 }, embedding.Frequencies.Shape.ToArray());
        Assert.Equal(lite ? new[] { 8, 5 } : new[] { 3, 8, 5 }, embedding.Weight.Shape.ToArray());
        Assert.Equal(lite ? new[] { 5 } : new[] { 3, 5 }, embedding.Bias.Shape.ToArray());
        Assert.Equal(embeddingCount, embedding.ParameterCount);
        AssertLinear(model.InputProjection, 15, 6, bias: true);
        Assert.Equal(total, model.ParameterCount);
        Assert.Equal(total, (long)model.GetParameters().Length);

        // Frequencies start as N(0, 0.01^2) draws (all inside ten standard deviations), the linear map uniform in
        // +-1/sqrt(2k)
        var frequencies = embedding.Frequencies.ToArray();
        Assert.All(frequencies, c => Assert.InRange(c, -0.1, 0.1));
        Assert.Contains(frequencies, c => c != 0.0);
        double bound = 1.0 / Math.Sqrt(8.0);
        Assert.All(embedding.Weight.ToArray().Concat(embedding.Bias.ToArray()), w => Assert.InRange(w, -bound, bound));

        var embedded = embedding.Forward(new Tensor<double>(new[] { 2, 3 }, new Vector<double>(new[] { 0.4, -1.1, 0.7, 1.3, 0.2, -0.5 })));
        Assert.Equal(new[] { 2, 15 }, embedded.Shape.ToArray());
    }

    [Fact]
    public void NumericEmbedding_IsNoneByDefault()
    {
        // TabR-S has no numeric embedding; the PLR defaults are rtdl_num_embeddings' (n_frequencies 48,
        // frequency_init_scale 0.01, d_embedding 24) with TabR's lite = true
        var options = new TabROptions<double>();
        Assert.Equal(TabRNumericEmbedding.None, options.NumericEmbedding);
        Assert.Equal(48, options.NumFrequencies);
        Assert.Equal(0.01, options.FrequencyInitScale);
        Assert.Equal(24, options.NumericEmbeddingDimension);
        Assert.True(options.LiteNumericEmbedding);

        var model = new TabRRegression<double>(3, 1, new TabROptions<double> { EmbeddingDimension = 6, NumNeighbors = 2 });
        Assert.Null(model.NumericEmbeddingModule);
        AssertLinear(model.InputProjection, 3, 6, bias: true);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void NumericEmbedding_ForwardMatchesAHandWrittenPeriodicLinearReLU(bool lite)
    {
        // periodic(x)_f = [cos(2 pi c_f x_f), sin(2 pi c_f x_f)] -> Linear (shared when lite, per feature otherwise)
        // -> ReLU, the feature embeddings concatenated along the row
        const int features = 3;
        const int k = 2;
        const int d = 3;
        var model = new TabRRegression<double>(features, 1, new TabROptions<double>
        {
            EmbeddingDimension = 4,
            NumNeighbors = 2,
            NumericEmbedding = TabRNumericEmbedding.PeriodicLinearReLU,
            NumFrequencies = k,
            NumericEmbeddingDimension = d,
            LiteNumericEmbedding = lite
        });
        var embedding = model.NumericEmbeddingModule;
        Assert.NotNull(embedding);

        int weightGroups = lite ? 1 : features;
        var c = new double[features, k];
        var w = new double[weightGroups, 2 * k, d];
        var b = new double[weightGroups, d];
        for (int f = 0; f < features; f++)
            for (int j = 0; j < k; j++) c[f, j] = 0.3 * (f + 1) - 0.2 * j + 0.05;
        for (int g = 0; g < weightGroups; g++)
        {
            for (int j = 0; j < 2 * k; j++)
                for (int e = 0; e < d; e++) w[g, j, e] = 0.4 * Math.Sin(1.3 * g + 0.7 * j + 1.9 * e + 0.3);
            for (int e = 0; e < d; e++) b[g, e] = 0.1 * (e - 1) + 0.05 * g;
        }

        // Parameter order: frequencies, weight ([2k, d] in -> out, or [F, 2k, d]), bias
        embedding.SetParameters(new Vector<double>(c.Cast<double>().Concat(w.Cast<double>()).Concat(b.Cast<double>()).ToArray()));
        Assert.Equal(c.Cast<double>().ToArray(), embedding.Frequencies.ToArray());
        Assert.Equal(w.Cast<double>().ToArray(), embedding.Weight.ToArray());

        double[,] x = { { 0.4, -1.1, 0.7 }, { 1.3, 0.2, -0.5 } };
        var actual = embedding.Forward(new Tensor<double>(new[] { 2, features }, new Vector<double>(x.Cast<double>().ToArray()))).ToArray();

        Assert.Equal(2 * features * d, actual.Length);
        int clipped = 0;
        int passed = 0;
        for (int r = 0; r < 2; r++)
        {
            for (int f = 0; f < features; f++)
            {
                var periodic = new double[2 * k];
                for (int j = 0; j < k; j++)
                {
                    double phase = 2.0 * Math.PI * c[f, j] * x[r, f];
                    periodic[j] = Math.Cos(phase);
                    periodic[k + j] = Math.Sin(phase);
                }

                int g = lite ? 0 : f;
                for (int e = 0; e < d; e++)
                {
                    double pre = b[g, e];
                    for (int j = 0; j < 2 * k; j++) pre += periodic[j] * w[g, j, e];
                    if (pre < 0) clipped++;
                    else passed++;
                    Assert.Equal(Math.Max(0.0, pre), actual[(r * features + f) * d + e], 12);
                }
            }
        }

        // Both sides of the ReLU are exercised (10 / 8 lite, 6 / 12 per feature)
        Assert.True(clipped > 0 && passed > 0, "The fixture must exercise both sides of the ReLU.");
    }

    private static void AssertLinear(FullyConnectedLayer<double> layer, int inputs, int outputs, bool bias)
    {
        var weights = layer.GetWeights();
        Assert.NotNull(weights);
        Assert.Equal(new[] { outputs, inputs }, weights.Shape.ToArray());
        Assert.Equal(bias, layer.UseBias);
        Assert.Equal((long)(inputs * outputs + (bias ? outputs : 0)), layer.ParameterCount);
    }

    private static void SetLinear(FullyConnectedLayer<double> layer, double[,] weights, double[] biases)
    {
        // Row-major [outputs, inputs] weights, then the biases: the layer's parameter order, checked by reading back
        layer.SetParameters(new Vector<double>(weights.Cast<double>().Concat(biases).ToArray()));
        var stored = layer.GetWeights();
        Assert.NotNull(stored);
        Assert.Equal(weights.Cast<double>().ToArray(), stored.ToArray());
    }

    private static Tensor<double> Softmax(Tensor<double> logits)
    {
        int rows = logits.Shape[0];
        int cols = logits.Shape[1];
        var result = new Tensor<double>(new[] { rows, cols });
        for (int r = 0; r < rows; r++)
        {
            double max = double.NegativeInfinity;
            for (int c = 0; c < cols; c++) max = Math.Max(max, logits[r, c]);
            double sum = 0;
            for (int c = 0; c < cols; c++) sum += Math.Exp(logits[r, c] - max);
            for (int c = 0; c < cols; c++) result[r, c] = Math.Exp(logits[r, c] - max) / sum;
        }

        return result;
    }
}
