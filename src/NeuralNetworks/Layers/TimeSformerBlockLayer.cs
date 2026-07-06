using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// TimeSformer encoder block with divided space-time attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// Implements the Bertasius et al. TimeSformer block pattern: temporal attention is
/// applied across frames for each spatial patch, spatial attention is then applied
/// within each frame, and a position-wise FFN follows. Residual connections and
/// pre-layer-normalization match modern transformer training practice.
/// </remarks>
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, HasTrainingMode = true, TestInputShape = "1, 17, 8", TestConstructorArgs = "8, 2, 32, 4")]
public sealed class TimeSformerBlockLayer<T> : LayerBase<T>
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _ffnDim;
    private readonly int _configuredFrames;

    private readonly LayerNormalizationLayer<T> _temporalNorm;
    private readonly MultiHeadAttentionLayer<T> _temporalAttention;
    private readonly LayerNormalizationLayer<T> _spatialNorm;
    private readonly MultiHeadAttentionLayer<T> _spatialAttention;
    private readonly LayerNormalizationLayer<T> _ffnNorm;
    private readonly IActivationFunction<T> _ffnActivation;
    private readonly DenseLayer<T> _ffnUp;
    private readonly DenseLayer<T> _ffnDown;

    public override bool SupportsTraining => true;

    public TimeSformerBlockLayer(
        int hiddenSize,
        int numHeads,
        int ffnDim,
        int numFrames,
        IActivationFunction<T>? ffnActivation = null)
        : base(new[] { hiddenSize }, new[] { hiddenSize })
    {
        if (hiddenSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (hiddenSize % numHeads != 0)
            throw new ArgumentException("hiddenSize must be divisible by numHeads.", nameof(numHeads));
        if (ffnDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(ffnDim));
        if (numFrames <= 0)
            throw new ArgumentOutOfRangeException(nameof(numFrames));

        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _ffnDim = ffnDim;
        _configuredFrames = numFrames;
        _ffnActivation = ffnActivation ?? new GELUActivation<T>();

        var identity = new IdentityActivation<T>() as IActivationFunction<T>;
        _temporalNorm = new LayerNormalizationLayer<T>(hiddenSize);
        _temporalAttention = new MultiHeadAttentionLayer<T>(numHeads, hiddenSize / numHeads, activationFunction: identity);
        _spatialNorm = new LayerNormalizationLayer<T>(hiddenSize);
        _spatialAttention = new MultiHeadAttentionLayer<T>(numHeads, hiddenSize / numHeads, activationFunction: identity);
        _ffnNorm = new LayerNormalizationLayer<T>(hiddenSize);
        _ffnUp = new DenseLayer<T>(ffnDim, _ffnActivation);
        _ffnDown = new DenseLayer<T>(hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);

        RegisterSubLayer(_temporalNorm);
        RegisterSubLayer(_temporalAttention);
        RegisterSubLayer(_spatialNorm);
        RegisterSubLayer(_spatialAttention);
        RegisterSubLayer(_ffnNorm);
        RegisterSubLayer(_ffnUp);
        RegisterSubLayer(_ffnDown);
    }

    /// <summary>
    /// Runs divided attention using the actual frame count from the video tokenizer.
    /// </summary>
    public Tensor<T> Forward(Tensor<T> input, int frameCount)
    {
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (input.Rank != 3)
            throw new ArgumentException("TimeSformerBlockLayer expects [batch, sequence, hidden] input.", nameof(input));
        if (input.Shape[2] != _hiddenSize)
            throw new ArgumentException(
                $"Hidden dimension mismatch: expected {_hiddenSize}, got {input.Shape[2]}.",
                nameof(input));
        if (input.Shape[1] < 2)
            throw new ArgumentException("TimeSformer input must include a CLS token and at least one patch token.", nameof(input));

        int batch = input.Shape[0];
        int sequence = input.Shape[1];
        int patchTokens = sequence - 1;
        int frames = ResolveFrameCount(patchTokens, frameCount);
        int spatialPatches = patchTokens / frames;

        var cls = Engine.TensorSlice(input, [0, 0, 0], [batch, 1, _hiddenSize]);
        var patches = Engine.TensorSlice(input, [0, 1, 0], [batch, patchTokens, _hiddenSize]);
        var patches4D = Engine.Reshape(patches, [batch, frames, spatialPatches, _hiddenSize]);

        var temporalInput = Engine.TensorPermute(patches4D, [0, 2, 1, 3]);
        temporalInput = Engine.Reshape(temporalInput, [batch * spatialPatches, frames, _hiddenSize]);
        var temporalAttn = _temporalAttention.Forward(_temporalNorm.Forward(temporalInput));
        var temporalOut = Engine.TensorAdd(temporalInput, temporalAttn);
        var temporal4D = Engine.Reshape(temporalOut, [batch, spatialPatches, frames, _hiddenSize]);
        var patchesAfterTemporal = Engine.TensorPermute(temporal4D, [0, 2, 1, 3]);

        var clsPerFrame = Engine.Reshape(cls, [batch, 1, 1, _hiddenSize]);
        clsPerFrame = Engine.TensorTile(clsPerFrame, [1, frames, 1, 1]);
        clsPerFrame = Engine.Reshape(clsPerFrame, [batch * frames, 1, _hiddenSize]);
        var spatialTokens = Engine.Reshape(patchesAfterTemporal, [batch * frames, spatialPatches, _hiddenSize]);
        var spatialInput = Engine.TensorConcatenate(new[] { clsPerFrame, spatialTokens }, axis: 1);
        var spatialAttn = _spatialAttention.Forward(_spatialNorm.Forward(spatialInput));
        var spatialOut = Engine.TensorAdd(spatialInput, spatialAttn);

        var clsFrames = Engine.TensorSlice(spatialOut, [0, 0, 0], [batch * frames, 1, _hiddenSize]);
        clsFrames = Engine.Reshape(clsFrames, [batch, frames, _hiddenSize]);
        var clsUpdated = Engine.ReduceMean(clsFrames, [1], keepDims: false);
        clsUpdated = Engine.Reshape(clsUpdated, [batch, 1, _hiddenSize]);

        var patchOut = Engine.TensorSlice(spatialOut, [0, 1, 0], [batch * frames, spatialPatches, _hiddenSize]);
        var patchSequence = Engine.Reshape(patchOut, [batch, patchTokens, _hiddenSize]);
        var sequenceOut = Engine.TensorConcatenate(new[] { clsUpdated, patchSequence }, axis: 1);

        var normed = _ffnNorm.Forward(sequenceOut);
        var flat = Engine.Reshape(normed, [batch * sequence, _hiddenSize]);
        var ffnUp = _ffnUp.Forward(flat);
        var ffnDown = _ffnDown.Forward(ffnUp);
        var ffn = Engine.Reshape(ffnDown, [batch, sequence, _hiddenSize]);

        return Engine.TensorAdd(sequenceOut, ffn);
    }

    public override Tensor<T> Forward(Tensor<T> input)
    {
        return Forward(input, _configuredFrames);
    }

    public override long ParameterCount =>
        _temporalNorm.ParameterCount + _temporalAttention.ParameterCount +
        _spatialNorm.ParameterCount + _spatialAttention.ParameterCount +
        _ffnNorm.ParameterCount + _ffnUp.ParameterCount + _ffnDown.ParameterCount;

    public override Vector<T> GetParameters() => ConcatenateLayerVectors(layer => layer.GetParameters());

    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            MaterializeLazySublayers();

        long expected = ParameterCount;
        if (parameters.Length != expected)
            throw new ArgumentException($"Expected {expected} parameters, got {parameters.Length}.");

        int offset = 0;
        SetSubParams(_temporalNorm, parameters, ref offset);
        SetSubParams(_temporalAttention, parameters, ref offset);
        SetSubParams(_spatialNorm, parameters, ref offset);
        SetSubParams(_spatialAttention, parameters, ref offset);
        SetSubParams(_ffnNorm, parameters, ref offset);
        SetSubParams(_ffnUp, parameters, ref offset);
        SetSubParams(_ffnDown, parameters, ref offset);
    }

    private void MaterializeLazySublayers()
    {
        bool wasTraining = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            int frames = Math.Max(1, _configuredFrames);
            var dummy = new Tensor<T>([1, frames + 1, _hiddenSize]);
            _ = Forward(dummy, frames);
            ResetState();
        }
        finally
        {
            SetTrainingMode(wasTraining);
        }
    }

    public override Vector<T> GetParameterGradients() =>
        ConcatenateLayerVectors(layer => layer.GetParameterGradients());

    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var layer in ParameterLayers)
            layer.ClearGradients();
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in ParameterLayers)
            layer.UpdateParameters(learningRate);
    }

    public override void SetTrainingMode(bool isTraining)
    {
        base.SetTrainingMode(isTraining);
        foreach (var layer in ParameterLayers)
            layer.SetTrainingMode(isTraining);
    }

    public override void ResetState()
    {
        foreach (var layer in ParameterLayers)
            layer.ResetState();
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["HiddenSize"] = _hiddenSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfnDim"] = _ffnDim.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumFrames"] = _configuredFrames.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["FfnActivationType"] = _ffnActivation.GetType().AssemblyQualifiedName
            ?? _ffnActivation.GetType().FullName ?? string.Empty;
        return metadata;
    }

    private LayerBase<T>[] ParameterLayers =>
    [
        _temporalNorm,
        _temporalAttention,
        _spatialNorm,
        _spatialAttention,
        _ffnNorm,
        _ffnUp,
        _ffnDown
    ];

    private Vector<T> ConcatenateLayerVectors(Func<LayerBase<T>, Vector<T>> selector)
    {
        var layers = ParameterLayers;
        if (layers.Length == 0)
            return new Vector<T>(0);

        var result = selector(layers[0]);
        for (int i = 1; i < layers.Length; i++)
            result = Vector<T>.Concatenate(result, selector(layers[i]));

        return result;
    }

    private static void SetSubParams(LayerBase<T> layer, Vector<T> source, ref int offset)
    {
        int count = (int)layer.ParameterCount;
        if (count == 0)
            return;

        var slice = source.Slice(offset, count);
        layer.SetParameters(slice);
        offset += count;
    }

    private int ResolveFrameCount(int patchTokens, int requestedFrameCount)
    {
        int preferred = requestedFrameCount > 0 ? requestedFrameCount : _configuredFrames;
        if (preferred > 0 && patchTokens % preferred == 0)
            return preferred;

        int max = Math.Min(Math.Max(1, preferred), patchTokens);
        for (int frames = max; frames >= 1; frames--)
        {
            if (patchTokens % frames == 0)
                return frames;
        }

        return 1;
    }
}
