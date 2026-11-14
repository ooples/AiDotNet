using AiDotNet.LossFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.SpeechRecognition.ConformerFamily;
using AiDotNet.Helpers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tests.ModelFamilyTests.Base;
using Xunit;

namespace AiDotNet.Tests.UnitTests.LossFunctions;

public class CTCLossTapeTests
{
    [Fact]
    public void ComputeTapeLoss_AcceptsBatchFirstLogProbabilities()
    {
        const int timeSteps = 8;
        const int classes = 4;
        double logUniform = Math.Log(1.0 / classes);
        var logProbs = new Tensor<double>([1, timeSteps, classes]);
        for (int i = 0; i < logProbs.Length; i++)
            logProbs[i] = logUniform;

        var encodedTarget = new Tensor<double>(new double[] { 1, 2, 1, 2 }, [4]);
        var loss = new CTCLoss<double>(classes, blankIndex: 0, inputsAreLogProbs: true);

        using var tape = new GradientTape<double>();
        var value = loss.ComputeTapeLoss(logProbs, encodedTarget);
        var gradients = tape.ComputeGradients(value, [logProbs]);

        Assert.True(IsFinite(value[0]), $"Expected finite CTC loss, got {value[0]}.");
        Assert.True(value[0] > 0.0);
        Assert.True(gradients.TryGetValue(logProbs, out var gradient));
        Assert.NotNull(gradient);
        Assert.All(gradient.ToArray(), item => Assert.True(IsFinite(item)));
    }

    [Fact]
    public void EfficientConformer_ProducesFiniteNormalizedCtcInput()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceToSequence,
            inputHeight: 64,
            inputWidth: 32,
            inputDepth: 1,
            outputSize: 4);
        var options = new EfficientConformerOptions
        {
            EncoderDim = 32,
            NumEncoderLayers = 3,
            NumAttentionHeads = 2,
            FeedForwardExpansionFactor = 2,
            ConvKernelSize = 5,
            InitialAttentionGroupSize = 2,
            DownsamplingFactor = 8,
            NumMels = 32,
            VocabSize = 4,
            DropoutRate = 0.0,
            UseLayerNormalization = true,
        };

        EfficientConformer<float> CreateModel()
        {
            LayerInitializationSeedScope.AmbientFallbackSeed = 1337;
            try
            {
                return new EfficientConformer<float>(
                    architecture, new EfficientConformerOptions(options));
            }
            finally
            {
                LayerInitializationSeedScope.AmbientFallbackSeed = null;
            }
        }

        void ResetPerTestState()
        {
            AiDotNet.Tensors.Helpers.BlasProvider.SetDeterministicMode(true);
            NeuralNetworkArchitecture<float>.DefaultRandomSeedOverride = 1234;
            if (AiDotNet.Tensors.Engines.AiDotNetEngine.Current is not AiDotNet.Tensors.Engines.CpuEngine)
                AiDotNet.Tests.TestInfrastructure.TestEngines.EnsureCpu();
            AiDotNet.Training.CompiledTapeTrainingStep<float>.Invalidate();
            WeightRegistry.Reset();
        }

        (Tensor<float> Input, Tensor<float> Target) CreateTrainingExample()
        {
            var rng = RandomHelper.CreateSeededRandom(42);
            var input = new Tensor<float>([1, 64, 32]);
            for (int i = 0; i < input.Length; i++)
                input[i] = (float)(-1.0 + 2.0 * rng.NextDouble());

            // The generated invariant first creates a continuous [1,8,4] target and then
            // projects it to CTC's encoded-label representation using the same RNG.
            for (int i = 0; i < 1 * 8 * 4; i++)
                rng.NextDouble();

            var encoded = new Tensor<float>([4]);
            encoded[0] = 1;
            encoded[1] = 2;
            int previous = 0;
            for (int position = 0; position < 2; position++)
            {
                int label;
                do
                {
                    label = rng.Next(4);
                }
                while (label == 0 || label == previous);

                encoded[2 + position] = label;
                previous = label;
            }

            return (input, encoded);
        }

        // Reproduce the generated-suite lifecycle: the previous invariant trains and disposes
        // an equivalent model before the next test creates a fresh arena and model.
        ResetPerTestState();
        using (var priorArena = TensorArena.Create())
        using (var priorModel = CreateModel())
        {
            var (priorInput, priorTarget) = CreateTrainingExample();
            // ShapeCheckedOutputShape performs one separately-scoped warm-up prediction the first
            // time a generated fixture family is touched, before its first training call.
            using (var shapeArena = TensorArena.Create())
            using (var shapeModel = CreateModel())
                shapeModel.Predict(priorInput);

            priorModel.Train(priorInput, priorTarget);
            var priorLoss1 = new CTCLoss<float>(4, blankIndex: 0, inputsAreLogProbs: true)
                .ComputeTapeLoss(priorModel.Predict(priorInput), priorTarget);
            priorModel.Train(priorInput, priorTarget);
            var priorLoss2 = new CTCLoss<float>(4, blankIndex: 0, inputsAreLogProbs: true)
                .ComputeTapeLoss(priorModel.Predict(priorInput), priorTarget);
            GC.KeepAlive(priorLoss1);
            GC.KeepAlive(priorLoss2);
        }
        ModelFamilyTestGcGate.ReclaimBetweenTests();
        ResetPerTestState();

        using var arena = TensorArena.Create();
        var model = CreateModel();
        using var _ = model;
        var (input, target) = CreateTrainingExample();
        var logProbs = model.Predict(input);

        model.SetTrainingMode(false);
        Tensor<float> current = input;
        var layers = ((ILayeredModel<float>)model).Layers;
        for (int layerIndex = 0; layerIndex < layers.Count; layerIndex++)
        {
            current = layers[layerIndex].Forward(current);
            for (int valueIndex = 0; valueIndex < current.Length; valueIndex++)
            {
                Assert.True(
                    IsFinite(current[valueIndex]),
                    $"Layer {layerIndex} ({layers[layerIndex].GetType().Name}) produced " +
                    $"non-finite output {current[valueIndex]} at index {valueIndex}.");
            }
        }

        Assert.Equal(new[] { 1, 8, 4 }, logProbs.Shape.ToArray());
        for (int time = 0; time < 8; time++)
        {
            double probabilityMass = 0.0;
            for (int @class = 0; @class < 4; @class++)
            {
                double value = logProbs[0, time, @class];
                Assert.True(IsFinite(value), $"logProbs[0,{time},{@class}] was {value}.");
                probabilityMass += Math.Exp(value);
            }

            Assert.InRange(probabilityMass, 1.0 - 1e-5, 1.0 + 1e-5);
        }

        var loss = new CTCLoss<float>(4, blankIndex: 0, inputsAreLogProbs: true)
            .ComputeTapeLoss(logProbs, target);
        Assert.True(IsFinite(loss[0]), $"Expected finite CTC loss, got {loss[0]}.");
    }

    /// <summary>
    /// Target [1, 2] over 3 frames with blank 0 has exactly five alignments (120, 112, 122, 012, 102), so
    /// p(target) = 0.027 + 0.09 + 0.054 + 0.15 + 0.036 = 0.357 and the loss is -ln(0.357) = 1.030019497202498.
    /// torch.nn.functional.ctc_loss (torch 2.12.1, reduction "sum", float64) returns the same value. A skip
    /// transition between the distinct labels 1 and 2 is required for the 120, 112, 122 and 102 alignments.
    /// </summary>
    [Fact]
    public void CalculateLoss_DistinctLabels_MatchesTorchReference()
    {
        var logProbs = LogTensor(new double[,]
        {
            { 0.5, 0.3, 0.2 },
            { 0.2, 0.5, 0.3 },
            { 0.3, 0.1, 0.6 },
        });
        var loss = new CTCLoss<double>(numClasses: 3, blankIndex: 0, inputsAreLogProbs: true);

        double value = loss.CalculateLoss(logProbs, new[] { new[] { 1, 2 } }, new[] { 3 }, new[] { 2 });

        Assert.Equal(1.030019497202498, value, 12);
    }

    /// <summary>
    /// The gradient with respect to log-probabilities is minus the per-frame label posterior,
    /// -(1/p) * sum of the probabilities of the alignments that emit class k at frame t, using the five
    /// alignments enumerated above.
    /// </summary>
    [Fact]
    public void CalculateGradient_DistinctLabels_MatchesHandComputedPosteriors()
    {
        var logProbs = LogTensor(new double[,]
        {
            { 0.5, 0.3, 0.2 },
            { 0.2, 0.5, 0.3 },
            { 0.3, 0.1, 0.6 },
        });
        var loss = new CTCLoss<double>(numClasses: 3, blankIndex: 0, inputsAreLogProbs: true);

        var gradient = loss.CalculateGradient(logProbs, new[] { new[] { 1, 2 } }, new[] { 3 }, new[] { 2 });

        const double p = 0.357;
        var expected = new double[,]
        {
            { -0.15 / p, -0.207 / p, 0.0 },
            { -0.036 / p, -0.24 / p, -0.081 / p },
            { -0.027 / p, 0.0, -0.33 / p },
        };
        for (int t = 0; t < 3; t++)
        {
            for (int c = 0; c < 3; c++)
            {
                Assert.Equal(expected[t, c], gradient[0, t, c], 10);
            }
        }
    }

    /// <summary>
    /// CalculateGradient must be the derivative of the batch-mean value CalculateLoss returns, including the
    /// 1/batch factor, a shorter second sequence, and zero gradient on its padded frame.
    /// </summary>
    [Fact]
    public void CalculateGradient_MatchesCentralFiniteDifferencesOfCalculateLoss()
    {
        const int batch = 2;
        const int timeSteps = 3;
        const int classes = 3;
        var logProbs = new Tensor<double>(new[] { batch, timeSteps, classes });
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < timeSteps; t++)
            {
                double sum = 0.0;
                var logits = new double[classes];
                for (int c = 0; c < classes; c++)
                {
                    logits[c] = Math.Sin(1.3 * b + 0.7 * t + 1.1 * c);
                    sum += Math.Exp(logits[c]);
                }

                for (int c = 0; c < classes; c++)
                {
                    logProbs[b, t, c] = logits[c] - Math.Log(sum);
                }
            }
        }

        int[][] targets = { new[] { 1, 2 }, new[] { 2 } };
        int[] inputLengths = { 3, 2 };
        int[] targetLengths = { 2, 1 };
        var loss = new CTCLoss<double>(numClasses: classes, blankIndex: 0, inputsAreLogProbs: true);

        var gradient = loss.CalculateGradient(logProbs, targets, inputLengths, targetLengths);

        const double h = 1e-5;
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < timeSteps; t++)
            {
                for (int c = 0; c < classes; c++)
                {
                    double original = logProbs[b, t, c];
                    logProbs[b, t, c] = original + h;
                    double plus = loss.CalculateLoss(logProbs, targets, inputLengths, targetLengths);
                    logProbs[b, t, c] = original - h;
                    double minus = loss.CalculateLoss(logProbs, targets, inputLengths, targetLengths);
                    logProbs[b, t, c] = original;

                    double numeric = (plus - minus) / (2 * h);
                    Assert.True(Math.Abs(numeric - gradient[b, t, c]) < 1e-7,
                        $"Gradient mismatch at [{b},{t},{c}]: analytic {gradient[b, t, c]}, numeric {numeric}.");
                }
            }
        }

        for (int c = 0; c < classes; c++)
        {
            Assert.Equal(0.0, gradient[1, 2, c]);
        }
    }

    /// <summary>
    /// With inputsAreLogProbs: false, one frame of probabilities [0.8, 0.2] and target [1] give
    /// -ln(0.2) = 1.6094379124341003 (torch 2.12.1 ctc_loss on the logged probabilities agrees), and the
    /// tape loss must report the same value as CalculateLoss.
    /// </summary>
    [Fact]
    public void ProbabilityInputs_TapeLossAndCalculateLossAgree()
    {
        var probs = new Tensor<double>(new double[] { 0.8, 0.2 }, new[] { 1, 1, 2 });
        var loss = new CTCLoss<double>(numClasses: 2, blankIndex: 0, inputsAreLogProbs: false);

        double direct = loss.CalculateLoss(probs, new[] { new[] { 1 } }, new[] { 1 }, new[] { 1 });
        var tape = loss.ComputeTapeLoss(probs, new Tensor<double>(new double[] { 1, 1, 1 }, new[] { 3 }));

        Assert.Equal(1.6094379124341003, direct, 12);
        Assert.Equal(1.6094379124341003, tape[0], 12);
    }

    /// <summary>
    /// For probability inputs the gradient chains through the log: d(-ln p1)/dp1 = -1/0.2 = -5, and the
    /// blank, which no alignment emits, gets 0.
    /// </summary>
    [Fact]
    public void ProbabilityInputs_GradientChainsThroughLog()
    {
        var probs = new Tensor<double>(new double[] { 0.8, 0.2 }, new[] { 1, 1, 2 });
        var loss = new CTCLoss<double>(numClasses: 2, blankIndex: 0, inputsAreLogProbs: false);

        var gradient = loss.CalculateGradient(probs, new[] { new[] { 1 } }, new[] { 1 }, new[] { 1 });

        Assert.Equal(0.0, gradient[0, 0, 0], 12);
        Assert.Equal(-5.0, gradient[0, 0, 1], 10);
    }

    private static Tensor<double> LogTensor(double[,] probabilities)
    {
        int timeSteps = probabilities.GetLength(0);
        int classes = probabilities.GetLength(1);
        var tensor = new Tensor<double>(new[] { 1, timeSteps, classes });
        for (int t = 0; t < timeSteps; t++)
        {
            for (int c = 0; c < classes; c++)
            {
                tensor[0, t, c] = Math.Log(probabilities[t, c]);
            }
        }

        return tensor;
    }

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}
