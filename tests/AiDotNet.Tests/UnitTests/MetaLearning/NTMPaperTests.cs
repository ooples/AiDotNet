using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Data.Structures;
using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Algorithms;
using AiDotNet.MetaLearning.Options;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.IntegrationTests.MetaLearning;
using Moq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.MetaLearning;

/// <summary>
/// Reference tests for <see cref="NTMAlgorithm{T, TInput, TOutput}"/> against Graves, Wayne and Danihelka,
/// "Neural Turing Machines" (2014), section 3, and its meta-learning use in Santoro et al., "Meta-Learning with
/// Memory-Augmented Neural Networks" (ICML 2016). Expected values were computed with NumPy from the paper's
/// equations.
/// </summary>
public class NTMPaperTests
{
    private const double Tolerance = 1e-12;

    private static Tensor<double> Vec(params double[] values) => new Tensor<double>(values, new[] { values.Length });

    private static readonly double[][] MemoryRows =
    {
        new[] { 1.0, 0.0, 0.0 },
        new[] { 0.0, 1.0, 0.0 },
        new[] { 0.0, 0.0, 1.0 },
        new[] { 1.0, 1.0, 0.0 },
    };

    /// <summary>A 4 x 3 memory holding <see cref="MemoryRows"/>, loaded by full erase-and-add writes of each row.</summary>
    private static NTMMemory<double> ReferenceMemory()
    {
        var memory = new NTMMemory<double>(4, 3, randomSeed: 1);
        for (int i = 0; i < MemoryRows.Length; i++)
        {
            var oneHot = new Tensor<double>(new[] { 4 });
            oneHot[i] = 1.0;
            memory.Write(oneHot, Vec(1.0, 1.0, 1.0), Vec(MemoryRows[i]));
        }

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++)
                Assert.Equal(MemoryRows[i][j], memory.GetValue(i, j), Tolerance);
        return memory;
    }

    private static void AssertClose(double[] expected, Tensor<double> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], Tolerance);
    }

    private static readonly Tensor<double> Key = Vec(1.0, 0.5, 0.0);
    private static readonly Tensor<double> PreviousWeights = Vec(0.1, 0.2, 0.3, 0.4);
    private static readonly Tensor<double> NoShift = Vec(0.0, 1.0, 0.0);
    private static readonly Tensor<double> Shift = Vec(0.2, 0.7, 0.1); // s(-1), s(0), s(+1)

    // NumPy: wc = softmax(2 * cos(k, M_i)).
    private static readonly double[] ContentWeights =
        { 0.3716626836205617, 0.15195121075609763, 0.06212399433088155, 0.414262111292459 };

    // NumPy: wg = 0.7 * wc + 0.3 * w_prev.
    private static readonly double[] GatedWeights =
        { 0.2901638785343932, 0.16636584752926836, 0.1334867960316171, 0.40998347790472134 };

    // NumPy: wt(i) = sum_j wg(j) s((i - j) mod 4).
    private static readonly double[] ShiftedWeights =
        { 0.27738623227040105, 0.17216984033025057, 0.19207403755600305, 0.3583698898433453 };

    // NumPy: w = wt^2 / sum(wt^2).
    private static readonly double[] SharpenedWeights =
        { 0.28297588804230245, 0.1090168883137659, 0.13568035141121002, 0.4723268722327215 };

    /// <summary>Eq. 5-6: content addressing is the softmax of the key strength times the cosine similarity.</summary>
    [Fact]
    public void ContentAddressing_IsSoftmaxOfKeyStrengthTimesCosineSimilarity()
    {
        var parameters = new NTMAddressingParameters<double>(Key, Vec(2.0), Vec(1.0), NoShift, Vec(1.0));
        AssertClose(ContentWeights, ReferenceMemory().Address(parameters, PreviousWeights));
    }

    /// <summary>Eq. 7: the gate interpolates the content weighting with the previous weighting.</summary>
    [Fact]
    public void Interpolation_BlendsContentAndPreviousWeightsByTheGate()
    {
        var parameters = new NTMAddressingParameters<double>(Key, Vec(2.0), Vec(0.7), NoShift, Vec(1.0));
        AssertClose(GatedWeights, ReferenceMemory().Address(parameters, PreviousWeights));
    }

    /// <summary>Eq. 8: the shift weighting is applied by circular convolution.</summary>
    [Fact]
    public void ConvolutionalShift_IsCircularConvolutionWithTheShiftWeighting()
    {
        var parameters = new NTMAddressingParameters<double>(Key, Vec(2.0), Vec(0.7), Shift, Vec(1.0));
        AssertClose(ShiftedWeights, ReferenceMemory().Address(parameters, PreviousWeights));
    }

    /// <summary>Eq. 5-9 in full: one addressing step ending with sharpening by gamma = 2.</summary>
    [Fact]
    public void AddressingStep_MatchesGravesEquations5To9()
    {
        var parameters = new NTMAddressingParameters<double>(Key, Vec(2.0), Vec(0.7), Shift, Vec(2.0));
        AssertClose(SharpenedWeights, ReferenceMemory().Address(parameters, PreviousWeights));
    }

    /// <summary>A head carries its weighting to the next step as w_{t-1}.</summary>
    [Fact]
    public void ReadHead_UsesItsPreviousWeightingForInterpolation()
    {
        var memory = ReferenceMemory();
        var head = new NTMReadHead<double>(3, 4, 0);

        // A zero gate keeps w_{t-1} only; the first step's w_{t-1} is zero, so a pure content step seeds it.
        head.ComputeReadWeights(new NTMAddressingParameters<double>(Key, Vec(2.0), Vec(1.0), NoShift, Vec(1.0)), memory);
        var held = head.ComputeReadWeights(
            new NTMAddressingParameters<double>(Vec(0.0, 0.0, 1.0), Vec(5.0), Vec(0.0), NoShift, Vec(1.0)), memory);

        AssertClose(ContentWeights, held);
    }

    /// <summary>Eq. 2-4: reading is the weighted sum of rows; writing erases then adds.</summary>
    [Fact]
    public void ReadAndWrite_MatchGravesEquations2To4()
    {
        var memory = ReferenceMemory();
        var weights = Vec(0.1, 0.2, 0.3, 0.4);

        // r = sum_i w(i) M(i)
        AssertClose(new[] { 0.5, 0.6, 0.3 }, memory.Read(weights));

        // M' = M * (1 - w e^T) + w a^T
        memory.Write(weights, Vec(0.5, 0.0, 1.0), Vec(0.2, -0.1, 0.3));
        double[] expected =
        {
            0.97, -0.01, 0.03,
            0.04000000000000001, 0.98, 0.06,
            0.06, -0.03, 0.7899999999999999,
            0.8800000000000001, 0.96, 0.12,
        };
        AssertClose(expected, memory.Contents);
    }

    /// <summary>Section 3.3: beta &gt;= 0, g in (0, 1), gamma &gt;= 1 and s a distribution over -R..R.</summary>
    [Theory]
    [InlineData(NTMControllerType.LSTM)]
    [InlineData(NTMControllerType.MLP)]
    public void ControllerAddressing_SatisfiesThePaperConstraints(NTMControllerType type)
    {
        var options = new NTMOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
        {
            MemoryWidth = 3, NumReadHeads = 2, ControllerHiddenSize = 4, NumClasses = 2, ShiftRange = 2, RandomSeed = 3,
        };
        INTMController<double> controller = type == NTMControllerType.LSTM
            ? new LSTMNTMController<double, Matrix<double>, Tensor<double>>(options)
            : new MLPNTMController<double, Matrix<double>, Tensor<double>>(options);

        var output = controller.Forward(Vec(0.3, -0.7, 0.2, 0.9, -0.4, 0.1, 0.5, -0.2), new List<Tensor<double>>());
        var heads = controller.GenerateReadAddressing(output).Append(controller.GenerateWriteAddressing(output)).ToList();

        Assert.Equal(3, heads.Count);
        foreach (var head in heads)
        {
            Assert.Equal(3, head.Key.Length);
            Assert.True(head.KeyStrength[0] >= 0);
            Assert.InRange(head.Gate[0], 0.0, 1.0);
            Assert.True(head.Sharpening[0] >= 1);
            Assert.Equal(5, head.Shift.Length);
            double total = 0;
            for (int i = 0; i < head.Shift.Length; i++)
            {
                Assert.True(head.Shift[i] >= 0);
                total += head.Shift[i];
            }

            Assert.Equal(1.0, total, 1e-12);
        }
    }

    private static MetaLearningTask<double, Matrix<double>, Tensor<double>> Task()
        => new()
        {
            SupportSetX = new Matrix<double>(new[,] { { 0.5, -1.0 }, { 1.0, 0.5 }, { -0.5, 0.25 } }),
            SupportSetY = Vec(0.0, 1.0, 0.0),
            QuerySetX = new Matrix<double>(new[,] { { 0.75, -0.5 }, { 0.9, 0.4 } }),
            QuerySetY = Vec(0.0, 1.0),
            NumWays = 2, NumShots = 1, NumQueryPerClass = 1,
        };

    private static NTMAlgorithm<double, Matrix<double>, Tensor<double>> Algorithm(NTMControllerType type, double learningRate)
        => new(new NTMOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
        {
            MemorySize = 4, MemoryWidth = 3, NumReadHeads = 1, NumClasses = 2, ControllerHiddenSize = 5,
            ControllerType = type, InitializeMemory = false, GradientClipThreshold = null,
            OuterLearningRate = learningRate, RandomSeed = 11,
        });

    /// <summary>Santoro et al. 2016 eq. 24: the adapted model emits a class distribution per query example.</summary>
    [Fact]
    public void AdaptedModel_PredictsOneProbabilityRowPerExample()
    {
        var task = Task();
        var predictions = Algorithm(NTMControllerType.LSTM, 0.01).Adapt(task).Predict(task.QuerySetX);

        Assert.Equal(new[] { 2, 2 }, predictions.Shape.ToArray());
        for (int r = 0; r < 2; r++)
        {
            Assert.InRange(predictions[r * 2], 0.0, 1.0);
            Assert.InRange(predictions[r * 2 + 1], 0.0, 1.0);
            Assert.Equal(1.0, predictions[r * 2] + predictions[r * 2 + 1], 1e-12);
        }
    }

    /// <summary>
    /// Training follows the tape gradient of the episode loss: with a deterministic episode (zero memory reset each
    /// time) a small plain-SGD step along it must lower the same episode's loss.
    /// </summary>
    [Theory]
    [InlineData(NTMControllerType.LSTM)]
    [InlineData(NTMControllerType.MLP)]
    public void MetaTrain_StepsDownTheTapeGradient(NTMControllerType type)
    {
        var algorithm = Algorithm(type, 1e-3);
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { Task() });

        double before = algorithm.MetaTrain(batch);
        double after = algorithm.MetaTrain(batch);

        Assert.False(double.IsNaN(before));
        Assert.True(after < before, $"Episode loss rose from {before} to {after} after a gradient step.");
    }

    /// <summary>
    /// Santoro et al. 2016 section 3.1: each input arrives with the previous step's one-hot label; the last support
    /// label rides with the first query, and unknown query labels are presented as zeros.
    /// </summary>
    [Fact]
    public void Controller_SeesEachInputWithThePreviousLabel()
    {
        var options = new NTMOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
        {
            MemorySize = 2, MemoryWidth = 2, NumReadHeads = 1, NumClasses = 3, ControllerHiddenSize = 2,
        };
        var controller = new RecordingController(options.MemoryWidth, options.NumClasses);
        var model = new NTMModel<double, Matrix<double>, Tensor<double>>(
            controller,
            new NTMMemory<double>(options.MemorySize, options.MemoryWidth, randomSeed: 1),
            new List<NTMReadHead<double>> { new(options.MemoryWidth, options.MemorySize, 0) },
            new NTMWriteHead<double>(options.MemoryWidth, options.MemorySize),
            options);

        model.ProcessTimestepInternal(Vec(0.1, 0.2), Vec(0.0, 0.0, 1.0));
        model.ProcessTimestepInternal(Vec(0.3, 0.4), Vec(1.0, 0.0, 0.0));
        model.Predict(new Matrix<double>(new[,] { { 0.5, 0.6 }, { 0.7, 0.8 } }));

        // Input layout: 2 features, then the 3-class label slot, then one 2-wide read vector.
        double[][] labelSlots = controller.Inputs.Select(input => new[] { input[2], input[3], input[4] }).ToArray();
        Assert.Equal(4, labelSlots.Length);
        Assert.Equal(new[] { 0.0, 0.0, 0.0 }, labelSlots[0]);
        Assert.Equal(new[] { 0.0, 0.0, 1.0 }, labelSlots[1]);
        Assert.Equal(new[] { 1.0, 0.0, 0.0 }, labelSlots[2]);
        Assert.Equal(new[] { 0.0, 0.0, 0.0 }, labelSlots[3]);
        Assert.All(controller.Inputs, input => Assert.Equal(7, input.Length));
    }

    private static Tensor<Half> HalfVec(params double[] values)
        => new Tensor<Half>(values.Select(v => (Half)v).ToArray(), new[] { values.Length });

    /// <summary>A 2 x 2 Half memory holding diag(300, 300), loaded by full erase-and-add writes.</summary>
    private static NTMMemory<Half> LargeHalfMemory()
    {
        var memory = new NTMMemory<Half>(2, 2, randomSeed: 1);
        memory.Write(HalfVec(1.0, 0.0), HalfVec(1.0, 1.0), HalfVec(300.0, 0.0));
        memory.Write(HalfVec(0.0, 1.0), HalfVec(1.0, 1.0), HalfVec(0.0, 300.0));
        Assert.Equal(300.0, (double)memory.GetValue(0, 0));
        Assert.Equal(300.0, (double)memory.GetValue(1, 1));
        return memory;
    }

    /// <summary>
    /// Eq. 6 on an all-zero memory in Half: every cosine similarity is zero, so the content weighting is uniform.
    /// The 1e-8 norm floor flushed to zero in Half and the similarity was 0/0 = NaN.
    /// </summary>
    [Fact]
    public void HalfContentAddressing_OnZeroMemory_IsUniform()
    {
        var memory = new NTMMemory<Half>(4, 3, randomSeed: 1);
        var weights = memory.Address(NTMAddressingParameters<Half>.ContentOnly(HalfVec(1.0, 0.0, 0.0)), HalfVec(0, 0, 0, 0));

        for (int i = 0; i < 4; i++)
            Assert.Equal(0.25, (double)weights[i], 1e-3);
    }

    /// <summary>
    /// Eq. 5-6 in Half with entries of 300: cos(k, M_0) = 1 and cos(k, M_1) = 0, so w = softmax([1, 0]). A row's
    /// squared norm, 90000, overflowed Half's 65504 and the similarity collapsed to 0, giving [0.5, 0.5].
    /// </summary>
    [Fact]
    public void HalfContentAddressing_WithLargeEntries_DoesNotOverflow()
    {
        var weights = LargeHalfMemory().Address(NTMAddressingParameters<Half>.ContentOnly(HalfVec(1.0, 0.0)), HalfVec(0, 0));

        double expected = Math.E / (1.0 + Math.E);
        Assert.Equal(expected, (double)weights[0], 2e-3);
        Assert.Equal(1.0 - expected, (double)weights[1], 2e-3);
    }

    /// <summary>
    /// The usage penalty is the summed row norms over Size * Width: (300 + 300) / 4 = 150 for diag(300, 300). The
    /// squared row norms overflowed Half to infinity.
    /// </summary>
    [Fact]
    public void HalfUsagePenalty_WithLargeEntries_DoesNotOverflow()
    {
        Assert.Equal(150.0, (double)LargeHalfMemory().ComputeUsagePenalty(), 1e-9);
    }

    /// <summary>
    /// Eq. 9 with gamma = 1000 on w~ = [0.1, 0.2, 0.3, 0.4]: the largest entry dominates, w = [0, 0, 0, 1]. Every
    /// w~(i)^1000 underflowed to zero in double and the normalization was 0/0 = NaN.
    /// </summary>
    [Fact]
    public void Sharpening_WithLargeGamma_DoesNotUnderflow()
    {
        // A zero gate keeps w_{t-1} as w~, unshifted; the zero memory makes the content weighting irrelevant.
        var memory = new NTMMemory<double>(4, 3, randomSeed: 1);
        var parameters = new NTMAddressingParameters<double>(Key, Vec(1.0), Vec(0.0), NoShift, Vec(1000.0));

        AssertClose(new[] { 0.0, 0.0, 0.0, 1.0 }, memory.Address(parameters, PreviousWeights));
    }

    /// <summary>
    /// Seeded random memory is (u - 0.5) * 0.1 per element, computed in double and rounded to T once, as it always
    /// was. Rounding u to float first and scaling in float changed seeded float memories.
    /// </summary>
    [Fact]
    public void SeededRandomMemory_InFloat_IsTheDoubleValueRoundedOnce()
    {
        var memory = new NTMMemory<float>(8, 8, randomSeed: 5);
        memory.InitializeRandom();

        var random = RandomHelper.CreateSeededRandom(5);
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++)
                Assert.Equal((float)((random.NextDouble() - 0.5) * 0.1), memory.GetValue(i, j));
    }

    /// <summary>
    /// Seeded controller weights are (2u - 1) * scale per weight, computed in double and rounded to T once, as they
    /// always were. Rounding u to float first and scaling in float changed seeded float controllers.
    /// </summary>
    [Theory]
    [InlineData(NTMControllerType.LSTM)]
    [InlineData(NTMControllerType.MLP)]
    public void SeededControllerWeights_InFloat_AreTheDoubleValuesRoundedOnce(NTMControllerType type)
    {
        var options = new NTMOptions<float, Matrix<float>, Tensor<float>>(
            new Mock<IFullModel<float, Matrix<float>, Tensor<float>>>().Object)
        {
            MemoryWidth = 3, NumReadHeads = 1, ControllerHiddenSize = 4, NumClasses = 2, RandomSeed = 9,
        };
        INTMController<float> controller = type == NTMControllerType.LSTM
            ? new LSTMNTMController<float, Matrix<float>, Tensor<float>>(options)
            : new MLPNTMController<float, Matrix<float>, Tensor<float>>(options);

        // The first tensor: the LSTM input gates [16, 6] (Xavier) or the MLP first layer [4, 6] (He), where 6 is
        // the constructor's estimate of the input width (3 + 1 read head * 3).
        int rows = type == NTMControllerType.LSTM ? 16 : 4;
        double scale = type == NTMControllerType.LSTM ? Math.Sqrt(2.0 / (6 + 4)) : Math.Sqrt(2.0 / 6);
        var parameters = controller.GetParameters();
        var random = RandomHelper.CreateSeededRandom(9);
        for (int i = 0; i < rows * 6; i++)
            Assert.Equal((float)((random.NextDouble() * 2.0 - 1.0) * scale), parameters[i]);
    }

    private static MetaLearningTask<double, Matrix<double>, Tensor<double>> OtherTask()
        => new()
        {
            SupportSetX = new Matrix<double>(new[,] { { -0.8, 0.3 }, { 0.2, -0.9 }, { 0.6, 0.6 } }),
            SupportSetY = Vec(1.0, 0.0, 1.0),
            QuerySetX = new Matrix<double>(new[,] { { -0.4, 0.7 }, { 0.1, -0.3 } }),
            QuerySetY = Vec(1.0, 0.0),
            NumWays = 2, NumShots = 1, NumQueryPerClass = 1,
        };

    private static void AssertSamePredictions(Tensor<double> expected, Tensor<double> actual)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i]);
    }

    /// <summary>
    /// Santoro et al. 2016 wipe the memory between episodes, so adapting to a task starts from the initial memory,
    /// fresh head weightings and a reset controller: adapting to the same task twice predicts the same. The LSTM
    /// controller used to continue from the hidden state the previous adapted model left behind.
    /// </summary>
    [Fact]
    public void Adapt_ToTheSameTaskTwice_PredictsTheSame()
    {
        var algorithm = Algorithm(NTMControllerType.LSTM, 0.01);
        var task = Task();

        var first = algorithm.Adapt(task).Predict(task.QuerySetX);
        var second = algorithm.Adapt(task).Predict(task.QuerySetX);

        AssertSamePredictions(first, second);
    }

    /// <summary>
    /// After meta-training, adaptation still starts a fresh episode: it predicts exactly what an untrained
    /// algorithm given the same controller weights predicts. It used to clone the last training episode's
    /// written memory and carry the heads' last weightings into the adapted model.
    /// </summary>
    [Fact]
    public void Adapt_AfterMetaTraining_StartsFromAFreshEpisode()
    {
        var task = Task();
        var trained = Algorithm(NTMControllerType.LSTM, 0.01);
        trained.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { OtherTask() }));

        // The same weights in an algorithm that has never trained (its first Adapt resolves the input width).
        var untrained = Algorithm(NTMControllerType.LSTM, 0.01);
        untrained.Adapt(task);
        untrained.Controller.SetParameters(trained.Controller.GetParameters());

        AssertSamePredictions(
            untrained.Adapt(task).Predict(task.QuerySetX),
            trained.Adapt(task).Predict(task.QuerySetX));
    }

    /// <summary>
    /// Adapted models share the controller's weights but not its recurrent state: adapting a second task does not
    /// change what the first adapted model predicts.
    /// </summary>
    [Fact]
    public void AdaptedModels_KeepTheirOwnControllerState()
    {
        var task = Task();
        var expected = Algorithm(NTMControllerType.LSTM, 0.01).Adapt(task).Predict(task.QuerySetX);

        var algorithm = Algorithm(NTMControllerType.LSTM, 0.01);
        var first = algorithm.Adapt(task);
        algorithm.Adapt(OtherTask()).Predict(OtherTask().QuerySetX);

        AssertSamePredictions(expected, first.Predict(task.QuerySetX));
    }

    /// <summary>
    /// Graves et al. 2014 section 4 train by backpropagation through time: the update is exactly -lr times the
    /// gradient of the episode loss (cross-entropy over the query steps), which a central finite difference of
    /// that same loss reproduces. The forward finite difference of a squared-error loss used before does not.
    /// </summary>
    [Theory]
    [InlineData(NTMControllerType.LSTM)]
    [InlineData(NTMControllerType.MLP)]
    public void MetaTrain_UpdateIsTheEpisodeLossGradient(NTMControllerType type)
    {
        const double learningRate = 1e-3;
        const double step = 1e-5;
        var task = Task();
        var batch = new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { task });
        var algorithm = Algorithm(type, learningRate);
        algorithm.Adapt(task); // resolves the controller's input width without changing any weight

        var initial = algorithm.Controller.GetParameters();
        algorithm.MetaTrain(batch);
        var updated = algorithm.Controller.GetParameters();
        Assert.Equal(initial.Length, updated.Length);

        double EpisodeLoss(int index, double offset)
        {
            var shifted = initial.Clone();
            shifted[index] += offset;
            algorithm.Controller.SetParameters(shifted);
            return algorithm.MetaTrain(batch); // the loss before its own update
        }

        for (int i = 0; i < initial.Length; i += 5)
        {
            double gradient = (initial[i] - updated[i]) / learningRate;
            double numeric = (EpisodeLoss(i, step) - EpisodeLoss(i, -step)) / (2 * step);
            Assert.True(Math.Abs(gradient - numeric) <= 1e-6 + 1e-4 * Math.Abs(numeric),
                $"Parameter {i}: update gives gradient {gradient}, central difference {numeric}.");
        }
    }

    /// <summary>
    /// The controller and memory are the whole learner (Graves et al. 2014; Santoro et al. 2016): meta-training
    /// leaves the supplied meta-model's parameters untouched. Every outer step used to copy the leading controller
    /// weights over them.
    /// </summary>
    [Fact]
    public void MetaTrain_LeavesTheSuppliedMetaModelUntouched()
    {
        var metaModel = new TensorEmbeddingModel(2, 2);
        var before = metaModel.GetParameters().Clone();
        var algorithm = new NTMAlgorithm<double, Matrix<double>, Tensor<double>>(
            new NTMOptions<double, Matrix<double>, Tensor<double>>(metaModel)
            {
                MemorySize = 4, MemoryWidth = 3, NumReadHeads = 1, NumClasses = 2, ControllerHiddenSize = 5,
                ControllerType = NTMControllerType.LSTM, InitializeMemory = false, OuterLearningRate = 0.01,
                RandomSeed = 11,
            });

        algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { Task() }));

        var after = metaModel.GetParameters();
        Assert.Equal(before.Length, after.Length);
        for (int i = 0; i < before.Length; i++)
            Assert.Equal(before[i], after[i]);
    }

    /// <summary>
    /// The algorithm's parameters are the controller's: GetParameters reads them and SetParameters writes them
    /// back. They used to be the supplied meta-model's, which the NTM never trains.
    /// </summary>
    [Theory]
    [InlineData(NTMControllerType.LSTM)]
    [InlineData(NTMControllerType.MLP)]
    public void Parameters_RoundTripTheController(NTMControllerType type)
    {
        var algorithm = Algorithm(type, 0.01);
        algorithm.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { Task() }));

        var controllerParameters = algorithm.Controller.GetParameters();
        var parameters = algorithm.GetParameters();
        Assert.Equal(controllerParameters.Length, parameters.Length);
        Assert.Equal(controllerParameters.Length, algorithm.ParameterCount);
        for (int i = 0; i < parameters.Length; i++)
            Assert.Equal(controllerParameters[i], parameters[i]);

        var shifted = parameters.Clone();
        for (int i = 0; i < shifted.Length; i++)
            shifted[i] += 0.25;
        algorithm.SetParameters(shifted);

        var restored = algorithm.Controller.GetParameters();
        for (int i = 0; i < shifted.Length; i++)
            Assert.Equal(shifted[i], restored[i]);
    }

    /// <summary>
    /// A saved NTM keeps what it trains: a trained controller saved and loaded into a freshly built algorithm has
    /// the same parameters and adapts and predicts identically. Save used to write only the supplied meta-model
    /// (and threw when it could not serialize itself), so the controller was lost.
    /// </summary>
    [Theory]
    [InlineData(NTMControllerType.LSTM)]
    [InlineData(NTMControllerType.MLP)]
    public void SaveAndLoad_RestoreTheTrainedController(NTMControllerType type)
    {
        var task = Task();
        var trained = Algorithm(type, 0.01);
        trained.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { OtherTask() }));

        string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"ntm-{Guid.NewGuid():N}.bin");
        try
        {
            trained.Save(path);
            var loaded = Algorithm(type, 0.01);
            loaded.Load(path);

            var expected = trained.Controller.GetParameters();
            var actual = loaded.Controller.GetParameters();
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
                Assert.Equal(expected[i], actual[i]);

            AssertSamePredictions(
                trained.Adapt(task).Predict(task.QuerySetX),
                loaded.Adapt(task).Predict(task.QuerySetX));
        }
        finally
        {
            System.IO.File.Delete(path);
        }
    }

    /// <summary>A checkpoint is only loaded into an algorithm of the same controller shape.</summary>
    [Fact]
    public void Load_RejectsADifferentlyShapedController()
    {
        var trained = Algorithm(NTMControllerType.LSTM, 0.01);
        trained.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { Task() }));

        string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"ntm-{Guid.NewGuid():N}.bin");
        try
        {
            trained.Save(path);
            var other = new NTMAlgorithm<double, Matrix<double>, Tensor<double>>(
                new NTMOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
                {
                    MemorySize = 4, MemoryWidth = 3, NumReadHeads = 1, NumClasses = 2, ControllerHiddenSize = 6,
                    ControllerType = NTMControllerType.LSTM, InitializeMemory = false, RandomSeed = 11,
                });

            Assert.Throws<System.IO.InvalidDataException>(() => other.Load(path));
        }
        finally
        {
            System.IO.File.Delete(path);
        }
    }

    /// <summary>
    /// The byte-array path keeps the controller too: a trained controller serialized and deserialized into a
    /// freshly built algorithm has the same parameters and adapts and predicts identically. The base wrote the
    /// flat parameter vector, which a fresh LSTM algorithm could not take back because its controller had not
    /// resolved its input width.
    /// </summary>
    [Theory]
    [InlineData(NTMControllerType.LSTM)]
    [InlineData(NTMControllerType.MLP)]
    public void SerializeAndDeserialize_RestoreTheTrainedController(NTMControllerType type)
    {
        var task = Task();
        var trained = Algorithm(type, 0.01);
        trained.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { OtherTask() }));

        var loaded = Algorithm(type, 0.01);
        loaded.Deserialize(trained.Serialize());

        var expected = trained.Controller.GetParameters();
        var actual = loaded.Controller.GetParameters();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i]);

        AssertSamePredictions(
            trained.Adapt(task).Predict(task.QuerySetX),
            loaded.Adapt(task).Predict(task.QuerySetX));
    }

    /// <summary>Serialized bytes are only restored into an algorithm of the same controller shape.</summary>
    [Fact]
    public void Deserialize_RejectsADifferentlyShapedController()
    {
        var trained = Algorithm(NTMControllerType.LSTM, 0.01);
        trained.MetaTrain(new TaskBatch<double, Matrix<double>, Tensor<double>>(new[] { Task() }));

        var other = new NTMAlgorithm<double, Matrix<double>, Tensor<double>>(
            new NTMOptions<double, Matrix<double>, Tensor<double>>(new TensorEmbeddingModel(2, 2))
            {
                MemorySize = 4, MemoryWidth = 3, NumReadHeads = 1, NumClasses = 2, ControllerHiddenSize = 6,
                ControllerType = NTMControllerType.LSTM, InitializeMemory = false, RandomSeed = 11,
            });

        Assert.Throws<System.IO.InvalidDataException>(() => other.Deserialize(trained.Serialize()));
    }

    /// <summary>A controller that records every input and addresses by content only.</summary>
    private sealed class RecordingController : INTMController<double>
    {
        private readonly int _memoryWidth;
        private readonly int _numClasses;

        public RecordingController(int memoryWidth, int numClasses)
        {
            _memoryWidth = memoryWidth;
            _numClasses = numClasses;
        }

        public List<Tensor<double>> Inputs { get; } = new();

        public Tensor<double> Forward(Tensor<double> input, List<Tensor<double>> readContents)
        {
            Inputs.Add(input.Clone());
            return new Tensor<double>(new[] { 2 });
        }

        public List<Tensor<double>> GenerateReadKeys(Tensor<double> output)
            => new() { Vec(1.0, 0.0) };

        public Tensor<double> GenerateWriteKey(Tensor<double> output) => Vec(0.0, 1.0);

        public List<NTMAddressingParameters<double>> GenerateReadAddressing(Tensor<double> output)
            => GenerateReadKeys(output).Select(key => NTMAddressingParameters<double>.ContentOnly(key)).ToList();

        public NTMAddressingParameters<double> GenerateWriteAddressing(Tensor<double> output)
            => NTMAddressingParameters<double>.ContentOnly(GenerateWriteKey(output));

        public Tensor<double> GenerateEraseVector(Tensor<double> output) => new(new[] { _memoryWidth });

        public Tensor<double> GenerateAddVector(Tensor<double> output) => Vec(0.5, -0.5);

        public Tensor<double> GenerateOutput(Tensor<double> output, List<Tensor<double>> readContents)
            => new(new[] { _numClasses });

        public Vector<double> GetParameters() => new(0);

        public IReadOnlyList<Tensor<double>> GetParameterTensors() => new List<Tensor<double>>();

        public void SetParameters(Vector<double> parameters)
        {
        }

        public void Reset()
        {
        }
    }
}
