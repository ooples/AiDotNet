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
                AiDotNet.Tensors.Engines.AiDotNetEngine.ResetToCpu();
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

    private static bool IsFinite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
}
