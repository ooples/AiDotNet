using AiDotNet.Interfaces;
using AiDotNet.Models.Parameters;

namespace AiDotNet.Training;

/// <summary>Reusable model-state snapshot honoring the model's restore contract.</summary>
internal sealed class ModelParameterCheckpoint<T, TInput, TOutput>
{
    private readonly IParameterizable<T, TInput, TOutput> _model;
    private Tensor<T>[]? _snapshot;
#if !AIDN_LEGACY_TFM
    private bool _chunked;
#endif

    internal ModelParameterCheckpoint(IParameterizable<T, TInput, TOutput> model)
        => _model = model ?? throw new ArgumentNullException(nameof(model));

    internal void Capture()
    {
#if !AIDN_LEGACY_TFM
        _chunked = ParameterTransfer.UsesChunkContract(_model);
        if (_chunked)
        {
            var chunks = _model.GetParameterChunks().ToArray();
            _snapshot = TensorCheckpoint<T>.Capture(chunks, _snapshot);
            return;
        }
#endif
        var flat = _model.GetParameters();
        _snapshot = TensorCheckpoint<T>.Capture(new[] { new Tensor<T>(new[] { flat.Length }, flat) }, _snapshot);
    }

    internal void Restore()
    {
        if (_snapshot is null) throw new InvalidOperationException("Capture model state before restoring it.");
#if !AIDN_LEGACY_TFM
        if (_chunked) { _model.SetParameterChunks(_snapshot); return; }
#endif
        _model.SetParameters(ParameterChunkBuffer.ToVector(_snapshot));
    }
}
