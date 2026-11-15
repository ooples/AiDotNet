using AiDotNet.ModelLoading;

namespace AiDotNet.Diffusion.NoisePredictors;

public partial class DiffusionResBlock<T>
{
    /// <summary>
    /// The official tensor names of this block in an LDM checkpoint (<c>openaimodel.py</c> <c>ResBlock</c>):
    /// <c>in_layers.0</c> (GroupNorm), <c>in_layers.2</c> (conv3x3), <c>emb_layers.1</c> (Linear),
    /// <c>out_layers.0</c> (GroupNorm), <c>out_layers.3</c> (conv3x3) and, when the channel count changes,
    /// <c>skip_connection</c> (conv1x1).
    /// </summary>
    internal IReadOnlyList<string> LdmTensorNames(string prefix)
    {
        var names = new List<string>();
        names.AddRange(LdmCheckpoint.Pair(prefix + "in_layers.0"));
        names.AddRange(LdmCheckpoint.Pair(prefix + "in_layers.2"));
        if (_timeMlp is not null) names.AddRange(LdmCheckpoint.Pair(prefix + "emb_layers.1"));
        names.AddRange(LdmCheckpoint.Pair(prefix + "out_layers.0"));
        names.AddRange(LdmCheckpoint.Pair(prefix + "out_layers.3"));
        if (_skipConv is not null) names.AddRange(LdmCheckpoint.Pair(prefix + "skip_connection"));
        return names;
    }

    /// <summary>Loads this block from an LDM checkpoint; see <see cref="LdmTensorNames"/>.</summary>
    /// <returns>The names read.</returns>
    internal IReadOnlyList<string> LoadTorchWeights(TorchTensorSource source, string prefix)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        var names = new List<string>();
        names.AddRange(LdmCheckpoint.GroupNorm(source, _norm1, prefix + "in_layers.0"));
        names.AddRange(LdmCheckpoint.Conv(source, _conv1, prefix + "in_layers.2", _inChannels, _outChannels, 3));
        if (_timeMlp is not null)
            names.AddRange(LdmCheckpoint.Linear(source, _timeMlp, prefix + "emb_layers.1", _timeEmbedDim, _outChannels));
        names.AddRange(LdmCheckpoint.GroupNorm(source, _norm2, prefix + "out_layers.0"));
        names.AddRange(LdmCheckpoint.Conv(source, _conv2, prefix + "out_layers.3", _outChannels, _outChannels, 3));
        if (_skipConv is not null)
            names.AddRange(LdmCheckpoint.Conv(source, _skipConv, prefix + "skip_connection", _inChannels, _outChannels, 1));
        return names;
    }
}
