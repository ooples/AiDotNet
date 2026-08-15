using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>SVTR local/global component-mixing block with the released model's pre-normalization and MLP.</summary>
[LayerCategory(LayerCategory.Transformer)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.High, TestInputShape = "1, 8, 8", TestConstructorArgs = "8, 2, 2, 4, 4, 2")]
[ElementWiseShape(Note = "Attention and MLP residuals preserve the token grid and hidden width.")]
public class SVTRMixingBlockLayer<T> : LayerBase<T>
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _height;
    private readonly int _width;
    private readonly int _windowHeight;
    private readonly int _windowWidth;
    private readonly bool _local;
    private readonly double _dropPathRate;
    private long _dropPathForwardCounter;
    private readonly LayerNormalizationLayer<T> _norm1;
    private readonly DenseLayer<T> _query;
    private readonly DenseLayer<T> _key;
    private readonly DenseLayer<T> _value;
    private readonly DenseLayer<T> _output;
    private readonly LayerNormalizationLayer<T> _norm2;
    private readonly DenseLayer<T> _mlpUp;
    private readonly DenseLayer<T> _mlpDown;
    private Tensor<bool>? _localMask;

    private ILayer<T>[] ParameterLayers =>
        [_norm1, _query, _key, _value, _output, _norm2, _mlpUp, _mlpDown];

    public bool UsesLocalMixing => _local;
    public bool UsesPreNormalization => true;
    public int HiddenSize => _hiddenSize;
    public int NumHeads => _numHeads;
    public int GridHeight => _height;
    public int GridWidth => _width;
    public int WindowHeight => _windowHeight;
    public int WindowWidth => _windowWidth;
    public double DropPathRate => _dropPathRate;
    public override bool SupportsTraining => true;
    public SVTRMixingBlockLayer(
        int hiddenSize, int numHeads, int height, int width,
        int windowHeight = 7, int windowWidth = 11, bool local = true,
        double dropPathRate = 0.0)
        : base([height * width, hiddenSize], [height * width, hiddenSize])
    {
        if (hiddenSize <= 0 || numHeads <= 0 || hiddenSize % numHeads != 0)
            throw new ArgumentException("hiddenSize must be positive and divisible by numHeads.");
        if (height <= 0 || width <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _height = height;
        _width = width;
        _windowHeight = windowHeight;
        _windowWidth = windowWidth;
        _local = local;
        if (dropPathRate < 0 || dropPathRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropPathRate));
        _dropPathRate = dropPathRate;

        _norm1 = new LayerNormalizationLayer<T>(hiddenSize);
        _query = new DenseLayer<T>(hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        _key = new DenseLayer<T>(hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        _value = new DenseLayer<T>(hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        _output = new DenseLayer<T>(hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        _norm2 = new LayerNormalizationLayer<T>(hiddenSize);
        _mlpUp = new DenseLayer<T>(hiddenSize * 4, new GELUActivation<T>() as IActivationFunction<T>);
        _mlpDown = new DenseLayer<T>(hiddenSize, new IdentityActivation<T>() as IActivationFunction<T>);
        foreach (var layer in ParameterLayers) RegisterSubLayer(layer);
    }

    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        bool unbatched = input.Rank == 2;
        var x = unbatched ? Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]) : input;
        if (x.Rank != 3 || x.Shape[1] != _height * _width || x.Shape[2] != _hiddenSize)
            throw new ArgumentException(
                $"Expected [B,{_height * _width},{_hiddenSize}], got [{string.Join(",", input.Shape)}].",
                nameof(input));

        int batch = x.Shape[0];
        int sequence = x.Shape[1];
        int headDim = _hiddenSize / _numHeads;
        // PaddleOCR's released SVTR-Tiny configuration sets `prenorm: false`, but the
        // corresponding implementation branch is x + mixer(norm1(x)) followed by
        // x + mlp(norm2(x)). Preserve that observable computation rather than the
        // misleading option name.
        var normalized = _norm1.Forward(x);
        var flat = Engine.Reshape(normalized, [batch * sequence, _hiddenSize]);
        var q = ProjectHeads(_query.Forward(flat), batch, sequence, headDim);
        var k = ProjectHeads(_key.Forward(flat), batch, sequence, headDim);
        var v = ProjectHeads(_value.Forward(flat), batch, sequence, headDim);
        Tensor<bool>? mask = _local ? GetLocalMask(batch) : null;
        var attended = Engine.ScaledDotProductAttention(
            q, k, v, mask, 1.0 / Math.Sqrt(headDim), out _);
        attended = Engine.TensorPermute(attended, [0, 2, 1, 3]);
        attended = Engine.Reshape(attended, [batch * sequence, _hiddenSize]);
        attended = _output.Forward(attended);
        attended = Engine.Reshape(attended, [batch, sequence, _hiddenSize]);
        var residual = Engine.TensorAdd(x, DropPath(attended, batch));

        var mlpInput = _norm2.Forward(residual);
        var mlpFlat = Engine.Reshape(mlpInput, [batch * sequence, _hiddenSize]);
        var mlp = _mlpDown.Forward(_mlpUp.Forward(mlpFlat));
        mlp = Engine.Reshape(mlp, [batch, sequence, _hiddenSize]);
        var result = Engine.TensorAdd(residual, DropPath(mlp, batch));
        return unbatched ? Engine.Reshape(result, [sequence, _hiddenSize]) : result;
    }

    private Tensor<T> ProjectHeads(Tensor<T> projected, int batch, int sequence, int headDim)
    {
        var shaped = Engine.Reshape(projected, [batch, sequence, _numHeads, headDim]);
        return Engine.TensorPermute(shaped, [0, 2, 1, 3]);
    }

    private Tensor<bool> GetLocalMask(int batch)
    {
        if (_localMask is not null && _localMask.Shape[0] == batch) return _localMask;
        int sequence = _height * _width;
        var mask = new Tensor<bool>([batch, _numHeads, sequence, sequence]);
        int radiusY = _windowHeight / 2;
        int radiusX = _windowWidth / 2;
        for (int query = 0; query < sequence; query++)
        {
            int qy = query / _width;
            int qx = query % _width;
            for (int key = 0; key < sequence; key++)
            {
                int ky = key / _width;
                int kx = key % _width;
                bool allowed = Math.Abs(qy - ky) <= radiusY && Math.Abs(qx - kx) <= radiusX;
                for (int b = 0; b < batch; b++)
                for (int head = 0; head < _numHeads; head++)
                    mask[b, head, query, key] = allowed;
            }
        }
        _localMask = mask;
        return mask;
    }

    private Tensor<T> DropPath(Tensor<T> branch, int batch)
    {
        if (_dropPathRate <= 0.0 || !IsTrainingMode) return branch;
        double keepProbability = 1.0 - _dropPathRate;
        long counter = System.Threading.Interlocked.Increment(ref _dropPathForwardCounter);
        var random = RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(unchecked((int)((uint)RandomSeed.Value * 2654435761u ^ (uint)counter)))
            : RandomHelper.CreateSecureRandom();
        var mask = new Tensor<T>(branch.Shape.ToArray());
        for (int b = 0; b < batch; b++)
        {
            T value = random.NextDouble() < _dropPathRate
                ? NumOps.Zero
                : NumOps.FromDouble(1.0 / keepProbability);
            int sampleSize = branch.Length / batch;
            for (int i = 0; i < sampleSize; i++) mask[b * sampleSize + i] = value;
        }
        return Engine.TensorMultiply(branch, mask);
    }

    public override Vector<T> GetParameterGradients() => Concatenate(layer => layer.GetParameterGradients());

    private Vector<T> Concatenate(Func<ILayer<T>, Vector<T>> selector)
    {
        var parts = ParameterLayers.Select(selector).ToArray();
        var result = new T[parts.Sum(part => part.Length)];
        int offset = 0;
        foreach (var part in parts)
        {
            part.AsSpan().CopyTo(result.AsSpan(offset, part.Length));
            offset += part.Length;
        }
        return new Vector<T>(result);
    }

    public override void UpdateParameters(T learningRate)
    {
        foreach (var layer in ParameterLayers) layer.UpdateParameters(learningRate);
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        foreach (var layer in ParameterLayers) layer.ClearGradients();
    }

    public override void ResetState()
    {
        foreach (var layer in ParameterLayers) layer.ResetState();
    }

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["HiddenSize"] = _hiddenSize.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["NumHeads"] = _numHeads.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Height"] = _height.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Width"] = _width.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["WindowHeight"] = _windowHeight.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["WindowWidth"] = _windowWidth.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["Local"] = _local.ToString(System.Globalization.CultureInfo.InvariantCulture);
        metadata["DropPathRate"] = _dropPathRate.ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }
}
