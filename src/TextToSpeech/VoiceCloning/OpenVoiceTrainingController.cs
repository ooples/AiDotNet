using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LearningRateSchedulers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Training;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Separate discriminator/generator optimizer steps over the actual paper graph.</summary>
internal sealed class OpenVoiceTrainingController<T>
{
    internal delegate T TrainingStep(Tensor<T> input, Tensor<T> target, Func<Tensor<T>, Tensor<T>, Tensor<T>> objective,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> optimizer);
    private readonly NeuralNetworkBase<T> _owner;
    private readonly OpenVoiceTrainingOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _supplied;
    private readonly Dictionary<string, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>> _optimizers = new();
    private IReadOnlyList<LayerBase<T>>? _selected;
    internal OpenVoiceTrainingGraph<T> Graph { get; }
    internal Action<string>? AfterStepForAudit { get; set; }

    internal OpenVoiceTrainingController(NeuralNetworkBase<T> owner, OpenVoiceTrainingGraph<T> graph,
        OpenVoiceTrainingOptions options, IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? supplied)
    { _owner = owner; Graph = graph; _options = options; _supplied = supplied; }

    internal IReadOnlyList<Tensor<T>> SelectParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        if (_selected is null) return parameters;
        var selected = new HashSet<Tensor<T>>(TapeTrainingStep<T>.CollectParameters(_selected.Cast<ILayer<T>>().ToList()),
            TensorReferenceComparer<Tensor<T>>.Instance);
        return parameters.Where(selected.Contains).ToArray();
    }

    internal T Train(TtsTrainingSample<T> sample, TrainingStep step)
    {
        // The sample preparation and the no-grad generator forward run outside both step arenas. Under a caller's
        // TensorArena (the model builder, the test harness) nothing rewinds that arena between calls, so each call's
        // vocoder forward stayed allocated: about 650 MB per call on the generated fixture, out of memory within 80 s.
        // The steps still run in their own arenas (TrainWithCustomObjective).
        using var noCallerArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        var data = Graph.Prepare(sample); var draw = Graph.Sample(data);
        _owner.SetTrainingMode(true);
        Tensor<T> generated;
        using (new NoGradScope<T>())
        {
            var value = Graph.Generate(data, draw);
            generated = new Tensor<T>(value.Shape.ToArray(), value.ToVector());
        }
        T Run(IReadOnlyList<LayerBase<T>> layers, string group, Func<Tensor<T>> objective)
        {
            _selected = layers;
            try { return step(data.Tokens, data.Audio, (_, _) => objective(), Optimizer(group)); }
            finally { _selected = null; }
        }
        Run(Graph.DiscriminatorLayers, "discriminator", () => Graph.DiscriminatorLoss(Graph.RealSegment(data, draw), generated));
        AfterStepForAudit?.Invoke("discriminator");
        var loss = Run(Graph.GeneratorLayers, "generator", () => Graph.GeneratorLoss(data, draw, adversarial: true, training: true));
        AfterStepForAudit?.Invoke("generator");
        _options.TrainingSteps++;
        return loss;
    }

    internal T Evaluate(TtsTrainingSample<T> sample)
    {
        using var noCallerArena = AiDotNet.Tensors.Helpers.TensorArena.Suspend();
        var data = Graph.Prepare(sample); var draw = Graph.Sample(data);
        using var noGrad = new NoGradScope<T>();
        return Graph.GeneratorLoss(data, draw, adversarial: false, training: false)[0];
    }

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> Optimizer(string group)
    {
        if (_supplied is not null) return _supplied;
        if (_optimizers.TryGetValue(group, out var existing)) return existing;
        var schedule = new LambdaLRScheduler(_options.LearningRate,
            _ => Math.Pow(_options.LearningRateDecay, Math.Truncate(_options.TrainingSteps / (double)Math.Max(1, _options.UpdatesPerEpoch))));
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(_owner, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = _options.LearningRate, Beta1 = _options.Beta1, Beta2 = _options.Beta2,
            Epsilon = _options.Epsilon, WeightDecay = _options.WeightDecay,
            EnableGradientClipping = false, LearningRateScheduler = schedule,
            SchedulerStepMode = SchedulerStepMode.StepPerBatch
        });
        _optimizers.Add(group, optimizer); return optimizer;
    }
}
