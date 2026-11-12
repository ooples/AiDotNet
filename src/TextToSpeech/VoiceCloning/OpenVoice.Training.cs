using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.VoiceCloning;

public partial class OpenVoice<T>
{
    private OpenVoiceTrainingController<T>? _training;
    private readonly bool _usesDefaultOptimizer;
    internal OpenVoiceTrainingController<T> TrainingForAudit => EnsurePaperTraining();
    protected override TtsSupervision RequiredSupervision => _native is null ? TtsSupervision.None : TtsSupervision.Recording;
    public override bool TrainsOnSampleObjective => _native is not null;

    private OpenVoiceTrainingController<T> EnsurePaperTraining()
    {
        ThrowIfDisposed();
        if (_native is null) throw new NotSupportedException("OpenVoice training requires the native converter.");
        if (_training is not null) return _training;
        var graph = new OpenVoiceTrainingGraph<T>(Engine, _native, _options.Training, _options.HiddenDim, SampleRate);
        if (ComponentLayers.Count != 0) graph.RestoreAuxiliaryParameters(ComponentLayers);
        ComponentLayers.Clear();
        ComponentLayers.AddRange(graph.AdditionalLayers); _options.Training.IncludeTrainingComponents = true;
        InvalidateParameterCountCache();
        return _training = new OpenVoiceTrainingController<T>(this, graph, _options.Training, _usesDefaultOptimizer ? null : _optimizer);
    }

    protected override void OnMutableConstructorConfigurationRestored()
    {
        base.OnMutableConstructorConfigurationRestored();
        _training = null;
        if (_native is not null && _options.Training.IncludeTrainingComponents) EnsurePaperTraining();
    }

    protected override IReadOnlyList<Tensor<T>> SelectTrainableParametersForTraining(IReadOnlyList<Tensor<T>> parameters)
        => _training?.SelectParameters(parameters) ?? parameters;

    /// <summary>Trains on IPA-token IDs and their recording through the complete paper objective.</summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training is not supported in ONNX mode.");
        Train(new TtsTrainingSample<T> { Tokens = input, Audio = expectedOutput });
        return true;
    }

    protected override T TrainOnSample(TtsTrainingSample<T> sample)
    {
        var controller = EnsurePaperTraining(); MaxGradNorm = NumOps.Zero;
        return controller.Train(sample, TrainWithCustomObjective);
    }

    public override T EvaluateTrainingObjective(TtsTrainingSample<T> sample)
    {
        var controller = EnsurePaperTraining(); bool previous = IsTrainingMode; SetTrainingMode(false);
        try { using var noGrad = new NoGradScope<T>(); return controller.Evaluate(sample); }
        finally { SetTrainingMode(previous); }
    }
}
