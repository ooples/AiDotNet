using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.PhysicsInformed.NeuralOperators;
using AiDotNet.PhysicsInformed.Options;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Training;
using AiDotNet.Video.ActionRecognition;
using AiDotNet.Video.Options;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Training;

/// <summary>
/// Models with their own objective train through <c>NeuralNetworkBase.TrainCustomTapeStep</c> on the shared fused
/// session (#1804). With compilation on every step must run fused, and the weights must land where the eager steps
/// put them. Each scenario feeds different data every step, so a plan that baked a step's data into its trace (a
/// mask, a branch input) would diverge from eager here.
/// </summary>
[Collection("FusedOptimizerGlobalState")]
public class CustomObjectiveFusedTrainingTests
{
    // The fused plan takes Adam's beta1, beta2 and epsilon as float, so double training drifts from the eager step
    // by a few 1e-8 of the parameter scale per step; a step on baked data moves weights by far more.
    private const double RelativeTolerance = 1e-5;

    public static TheoryData<string> Scenarios => new() { "VideoMAE", "DeepONet", "FNO", "MQCNN", "CodeSwitchingASR", "CCDM", "TSDiff", "TimeDiff", "TFC", "NeRF" };

    private sealed record RunResult(Vector<double> Parameters, int FusedSteps, int EagerSteps);

    // Builds the scenario's model and resolves its lazily shaped layers with one forward, so a clone carries
    // every weight: the two arms must start from identical weights, which separately constructed models with
    // unseeded layer initializers do not.
    private static NeuralNetworkBase<double> Build(string scenario) => scenario switch
    {
        "VideoMAE" => WarmVideoMae(BuildVideoMae()),
        "DeepONet" => BuildDeepOnet(),
        "FNO" => Warm(BuildFno(), new Tensor<double>([1, 2, 4])),
        "MQCNN" => Warm(BuildMqcnn(), new Tensor<double>([1, 8, 1])),
        "CodeSwitchingASR" => Warm(BuildAsr(), new Tensor<double>([4, 8])),
        "CCDM" => Warm(BuildCcdm(), new Tensor<double>([1, 8, 1])),
        "TSDiff" => Warm(BuildTsDiff(), new Tensor<double>([1, 16, 1])),
        "TimeDiff" => Warm(BuildTimeDiff(), new Tensor<double>([1, 16, 1])),
        "TFC" => Warm(BuildTfc(), new Tensor<double>([1, 16, 1])),
        "NeRF" => WarmNerf(BuildNerf()),
        _ => throw new ArgumentOutOfRangeException(nameof(scenario)),
    };

    // The reconstruction decoder is shaped lazily on the pretraining path, which Predict never reaches.
    private static VideoMAE<double> WarmVideoMae(VideoMAE<double> model)
    {
        var clip = new Tensor<double>([1, 4, 3, 32, 32]);
        model.Predict(clip);
        model.DecodeForReconstruction(model.EncodeVisiblePatches(clip, model.CreateTubeMask(1, 2, 2)));
        return model;
    }

    private static NeuralNetworkBase<double> Warm(NeuralNetworkBase<double> model, Tensor<double> input)
    {
        model.Predict(input);
        return model;
    }

    private static void Train(string scenario, NeuralNetworkBase<double> model)
    {
        switch (scenario)
        {
            case "VideoMAE": TrainVideoMae((VideoMAE<double>)model); break;
            case "DeepONet": TrainDeepOnet((DeepOperatorNetwork<double>)model); break;
            case "FNO": TrainFno((FourierNeuralOperator<double>)model); break;
            case "MQCNN": TrainOnChangingBatches(model, [1, 8, 1], [1, 2 * MqcnnQuantiles]); break;
            case "CodeSwitchingASR": TrainOnChangingBatches(model, [4, 8], [4, AsrOptions().VocabSize]); break;
            // Every step draws its own diffusion steps, noise and contrastive negatives.
            case "CCDM": TrainOnChangingBatches(model, [1, 8, 1], [1, 4]); break;
            case "TSDiff": TrainOnChangingBatches(model, [1, 16, 1], [1, 4]); break;
            case "TimeDiff": TrainOnChangingBatches(model, [1, 16, 1], [1, 4]); break;
            case "TFC": TrainOnChangingBatches(model, [1, 16, 1], [1, 4]); break;
            case "NeRF": TrainNerf((AiDotNet.NeuralRadianceFields.Models.NeRF<double>)model); break;
        }
    }

    private static RunResult Run(NeuralNetworkBase<double> model, string scenario, bool compile)
    {
        var original = TensorCodecOptions.Current;
        try
        {
            TensorCodecOptions.SetCurrent(new TensorCodecOptions { EnableCompilation = compile });
            CompiledTapeTrainingStep<double>.Invalidate();
            Train(scenario, model);
            return new RunResult(model.GetParameters(), model.CustomFusedStepCount, model.CustomEagerStepCount);
        }
        finally
        {
            TensorCodecOptions.SetCurrent(original);
            CompiledTapeTrainingStep<double>.Invalidate();
        }
    }

    private static VideoMAE<double> BuildVideoMae()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.FourDimensional, taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputFrames: 4, inputDepth: 3, inputHeight: 32, inputWidth: 32, outputSize: 4);
        return new VideoMAE<double>(architecture,
            options: new VideoMAEOptions { NumClasses = 4, NumFrames = 4, NumFeatures = 8, MaskRatio = 0.5, Seed = 3 });
    }

    private static void TrainVideoMae(VideoMAE<double> model)
    {
        for (int step = 0; step < 3; step++)
        {
            var clip = new Tensor<double>([1, 4, 3, 32, 32]);
            for (int i = 0; i < clip.Length; i++) clip[i] = Math.Sin(i * 0.013 + step);
            model.PretrainMAE(clip);
        }
    }

    private static NeuralNetworkArchitecture<double> Linear(int inputSize, int outputSize) => new(
        inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
        complexity: NetworkComplexity.Simple, inputSize: inputSize, outputSize: outputSize,
        layers: new List<ILayer<double>>
        {
            new DenseLayer<double>(4, (IActivationFunction<double>)new TanhActivation<double>()),
            new DenseLayer<double>(outputSize, (IActivationFunction<double>)new IdentityActivation<double>()),
        });

    private static DeepOperatorNetwork<double> BuildDeepOnet()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple, inputSize: 3, outputSize: 1);
        return new DeepOperatorNetwork<double>(architecture, Linear(2, 3), Linear(1, 3),
            options: new DeepOperatorNetworkOptions { LatentDimension = 3, NumSensors = 2 });
    }

    private static void TrainDeepOnet(DeepOperatorNetwork<double> model)
    {
        const int samples = 4, queries = 3;
        var functions = new double[samples, 2];
        var locations = new double[samples, queries, 1];
        var targets = new double[samples, queries];
        for (int s = 0; s < samples; s++)
        {
            functions[s, 0] = 0.1 * (s + 1);
            functions[s, 1] = -0.2 * s;
            for (int q = 0; q < queries; q++)
            {
                locations[s, q, 0] = 0.25 * q - 0.1 * s;
                targets[s, q] = Math.Sin(functions[s, 0] + locations[s, q, 0]);
            }
        }
        model.Train(functions, locations, targets, epochs: 2, learningRate: 0.01, verbose: false);
    }

    private const int MqcnnQuantiles = 3;

    private static AiDotNet.Finance.Forecasting.Neural.MQCNN<double> BuildMqcnn() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 1, outputSize: 2 * MqcnnQuantiles),
        new AiDotNet.Models.Options.MQCNNOptions<double>
        {
            LookbackWindow = 8, ForecastHorizon = 2, EncoderChannels = 4, DecoderChannels = 4,
            NumEncoderLayers = 1, NumDecoderLayers = 1, DropoutRate = 0.0,
        });

    // The contrastive term is on, so the packed batch carries both negative families. Seed gives the two arms the
    // same training draws.
    private static AiDotNet.Finance.Forecasting.Foundation.CCDM<double> BuildCcdm() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 4),
        new AiDotNet.Models.Options.CCDMOptions<double>
        {
            ContextLength = 8, ForecastHorizon = 4, HiddenDimension = 8, NumLayers = 1, NumHeads = 2, EmbeddingLayers = 2,
            DiffusionSteps = 10, NumSamples = 2, TrainingBatchSize = 2, DropoutRate = 0.0, AttentionDropoutRate = 0.0,
            ContrastiveWeight = 0.5, NumNegatives = 2, Seed = 11,
        });

    // Half the rows drop their conditioning, so both classifier-free branches train every step.
    private static AiDotNet.Finance.Forecasting.Foundation.TSDiff<double> BuildTsDiff() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 16, outputSize: 4),
        new AiDotNet.Models.Options.TSDiffOptions<double>
        {
            SequenceLength = 16, ForecastHorizon = 4, HiddenDimension = 8, NumResidualBlocks = 1, NumDiffusionSteps = 10,
            NumSamples = 2, TrainingBatchSize = 4, UnconditionalProbability = 0.5, DropoutRate = 0.0, Seed = 13,
        });

    // Future mixup is on, so every row also draws its own mixing mask.
    private static AiDotNet.Finance.Forecasting.Foundation.TimeDiff<double> BuildTimeDiff() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 16, outputSize: 4),
        new AiDotNet.Models.Options.TimeDiffOptions<double>
        {
            ContextLength = 16, ForecastHorizon = 4, HiddenDimension = 8, NumLayers = 1, NumHeads = 2, DiffusionSteps = 10,
            TrainingBatchSize = 4, DropoutRate = 0.0, Seed = 17,
        });

    // The contrastive term reads the step's input from inside the forward.
    private static AiDotNet.Finance.Forecasting.Foundation.TFC<double> BuildTfc() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 16, outputSize: 4),
        new AiDotNet.Models.Options.TFCOptions<double>
        {
            ContextLength = 16, ForecastHorizon = 4, HiddenDimension = 8, ProjectionDimension = 4, NumTimeLayers = 1,
            NumFreqLayers = 1, DropoutRate = 0.0,
        });

    // Hierarchical sampling is on, so every batch also runs the coarse pass that places the fine samples. Seed gives
    // the two arms the same stratified jitter and importance draws.
    private static AiDotNet.NeuralRadianceFields.Models.NeRF<double> BuildNerf() => new(
        options: new AiDotNet.Models.Options.NeRFOptions
        {
            PositionEncodingLevels = 2, DirectionEncodingLevels = 2, HiddenDim = 8, NumLayers = 1, ColorHiddenDim = 4,
            ColorNumLayers = 1, UseHierarchicalSampling = true, HierarchicalSamples = 2, RenderSamples = 3,
            RenderNearBound = 1.0, RenderFarBound = 3.0, LearningRate = 1e-3, Seed = 7,
        });

    private static AiDotNet.NeuralRadianceFields.Models.NeRF<double> WarmNerf(AiDotNet.NeuralRadianceFields.Models.NeRF<double> model)
    {
        var origins = new Tensor<double>([1, 3]);
        var directions = new Tensor<double>([1, 3]);
        directions[2] = 1.0;
        model.RenderRays(origins, directions, 3, 1.0, 3.0);
        return model;
    }

    private static AiDotNet.NeuralRadianceFields.Data.ImageView<double>[] NerfViews()
    {
        var views = new AiDotNet.NeuralRadianceFields.Data.ImageView<double>[2];
        for (int v = 0; v < views.Length; v++)
        {
            var photo = new Tensor<double>([4, 4, 3]);
            for (int i = 0; i < photo.Length; i++) photo[i] = 0.5 + 0.4 * Math.Sin(0.7 * i + v);
            var rotation = new Matrix<double>(3, 3);
            rotation[0, 0] = 1.0; rotation[1, 1] = 1.0; rotation[2, 2] = 1.0;
            views[v] = new AiDotNet.NeuralRadianceFields.Data.ImageView<double>(
                photo, new Vector<double>(new[] { 0.0, 0.0, -1.0 * v }), rotation, focalLength: 0.0);
        }
        return views;
    }

    // Three batches, each its own rays, targets and sample draws.
    private static void TrainNerf(AiDotNet.NeuralRadianceFields.Models.NeRF<double> model)
    {
        var loader = AiDotNet.NeuralRadianceFields.Data.ImageTrainingDataLoaders.FromViews(NerfViews(), seed: 42);
        for (int step = 0; step < 3; step++)
            model.TrainOnImageBatch(loader, raysPerBatch: 4, optimizerOptions: null);
    }
    private static AiDotNet.SpeechRecognition.Specialized.CodeSwitchingASROptions AsrOptions() => new()
    {
        EncoderDim = 16, NumEncoderLayers = 1, NumAttentionHeads = 2, NumMels = 8, DecoderDim = 16,
        NumDecoderLayers = 1, MandarinCharVocabSize = 6, EnglishBpeVocabSize = 5, MaxTextLength = 16, DropoutRate = 0.0,
    };

    private static AiDotNet.SpeechRecognition.Specialized.CodeSwitchingASR<double> BuildAsr() => new(
        new NeuralNetworkArchitecture<double>(inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification, inputSize: 64, outputSize: 32),
        AsrOptions());

    // Three Train calls, each on its own input and target.
    private static void TrainOnChangingBatches(NeuralNetworkBase<double> model, int[] inputShape, int[] targetShape)
    {
        for (int step = 0; step < 3; step++)
        {
            var input = new Tensor<double>(inputShape);
            var target = new Tensor<double>(targetShape);
            for (int i = 0; i < input.Length; i++) input[i] = Math.Sin(0.37 * i + step);
            for (int i = 0; i < target.Length; i++) target[i] = 0.5 + 0.4 * Math.Cos(0.23 * i - step);
            model.Train(input, target);
        }
    }

    private static FourierNeuralOperator<double> BuildFno() =>
        new(Linear(2, 2), spatialDimensions: new[] { 4 },
            options: new FourierNeuralOperatorOptions { Modes = 2, Width = 3, NumLayers = 1 });

    private static void TrainFno(FourierNeuralOperator<double> model)
    {
        var inputs = new Tensor<double>[3];
        var targets = new Tensor<double>[3];
        for (int s = 0; s < 3; s++)
        {
            inputs[s] = new Tensor<double>([1, 2, 4]);
            targets[s] = new Tensor<double>([1, 2, 4]);
            for (int i = 0; i < 8; i++)
            {
                inputs[s][i] = Math.Cos(0.3 * i + s);
                targets[s][i] = 0.5 * Math.Sin(0.2 * i - s);
            }
        }
        model.Train(inputs, targets, epochs: 2, learningRate: 0.01);
    }
    [Theory(Timeout = 600000)]
    [MemberData(nameof(Scenarios))]
    public async Task CompiledTraining_RunsEveryStepFused_AndTakesTheEagerStep(string scenario)
    {
        await Task.Yield();
        var model = Build(scenario);
        var twin = Build(scenario);
        twin.SetParameters(model.GetParameters());
        Assert.Equal(model.GetParameters().ToArray(), twin.GetParameters().ToArray());
        var eager = Run(model, scenario, compile: false);
        var fused = Run(twin, scenario, compile: true);

        Assert.Equal(0, eager.FusedSteps);
        Assert.True(eager.EagerSteps > 0, $"{scenario} ran no custom training step.");
        Assert.Equal(0, fused.EagerSteps);
        Assert.Equal(eager.EagerSteps, fused.FusedSteps);

        Assert.Equal(eager.Parameters.Length, fused.Parameters.Length);
        double scale = 0.0, worst = 0.0;
        for (int i = 0; i < eager.Parameters.Length; i++)
        {
            scale = Math.Max(scale, Math.Abs(eager.Parameters[i]));
            worst = Math.Max(worst, Math.Abs(eager.Parameters[i] - fused.Parameters[i]));
        }

        Assert.True(worst <= RelativeTolerance * Math.Max(scale, 1.0),
            $"{scenario}: fused training diverged from eager by {worst:G6} (parameter scale {scale:G6}).");
    }
}
