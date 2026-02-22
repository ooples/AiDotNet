using System.Diagnostics.CodeAnalysis;
using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Diffusion;

/// <summary>
/// U-shaped Vision Transformer (U-ViT) noise predictor for diffusion models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// U-ViT combines the best of U-Net and Vision Transformer architectures.
/// It applies a transformer to image patches but adds long skip connections
/// between encoder and decoder blocks, similar to U-Net.
/// </para>
/// <para>
/// <b>For Beginners:</b> U-ViT is like a transformer with U-Net-style shortcuts:
///
/// U-Net: Uses conv layers with skip connections
/// DiT: Uses transformer layers without skip connections
/// U-ViT: Uses transformer layers WITH skip connections (best of both)
///
/// Architecture:
/// 1. Patchify: Split image into patches
/// 2. Encoder transformer blocks (L/2 blocks)
/// 3. Middle transformer block
/// 4. Decoder transformer blocks (L/2 blocks) with skip connections from encoder
/// 5. Unpatchify: Reconstruct output
///
/// Used in: UniDiffuser (multi-modal generation).
/// </para>
/// <para>
/// Reference: Bao et al., "All are Worth Words: A ViT Backbone for Diffusion Models", CVPR 2023
/// </para>
/// </remarks>
public class UViTNoisePredictor<T> : NoisePredictorBase<T>
{
    private readonly int _inputChannels;
    private readonly int _hiddenSize;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _patchSize;
    private readonly int _contextDim;

    // Patch embedding
    private DenseLayer<T> _patchEmbed;

    // Time embedding MLP
    private DenseLayer<T> _timeEmbed1;
    private DenseLayer<T> _timeEmbed2;

    // Encoder, middle, and decoder blocks (DiT-style blocks)
    private readonly List<UViTBlock> _encoderBlocks;
    private UViTBlock _middleBlock;
    private readonly List<UViTBlock> _decoderBlocks;

    // Skip connection projection layers: project [2*hidden] → [hidden]
    private readonly List<DenseLayer<T>> _skipProjections;

    // Final layer norm and output projection
    private LayerNormalizationLayer<T> _finalNorm;
    private DenseLayer<T> _outputProj;

    // Position embeddings
    private Tensor<T>? _posEmbed;
    private Tensor<T>? _lastInput;

    /// <inheritdoc />
    public override int InputChannels => _inputChannels;

    /// <inheritdoc />
    public override int OutputChannels => _inputChannels;

    /// <inheritdoc />
    public override int BaseChannels => _hiddenSize;

    /// <inheritdoc />
    public override int TimeEmbeddingDim => _hiddenSize * 4;

    /// <inheritdoc />
    public override bool SupportsCFG => true;

    /// <inheritdoc />
    public override bool SupportsCrossAttention => _contextDim > 0;

    /// <inheritdoc />
    public override int ContextDimension => _contextDim;

    /// <summary>
    /// Gets the patch size used for tokenizing spatial features.
    /// </summary>
    public int PatchSize => _patchSize;

    /// <summary>
    /// Gets the hidden dimension of the transformer.
    /// </summary>
    public int HiddenSize => _hiddenSize;

    /// <summary>
    /// Initializes a new instance of the U-ViT noise predictor.
    /// </summary>
    /// <param name="inputChannels">Number of input channels (4 for latent diffusion).</param>
    /// <param name="hiddenSize">Hidden dimension of the transformer (default: 512 for U-ViT-S/2).</param>
    /// <param name="numLayers">Total number of transformer layers (default: 12). Must be even.</param>
    /// <param name="numHeads">Number of attention heads (default: 8).</param>
    /// <param name="patchSize">Patch size for spatial tokenization (default: 2).</param>
    /// <param name="contextDim">Cross-attention context dimension (0 = no cross-attention).</param>
    /// <param name="seed">Optional random seed for weight initialization.</param>
    public UViTNoisePredictor(
        int inputChannels = 4,
        int hiddenSize = 512,
        int numLayers = 12,
        int numHeads = 8,
        int patchSize = 2,
        int contextDim = 0,
        int? seed = null)
        : base(seed: seed)
    {
        numLayers = numLayers % 2 == 0 ? numLayers : numLayers + 1;

        _inputChannels = inputChannels;
        _hiddenSize = hiddenSize;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _patchSize = patchSize;
        _contextDim = contextDim;

        _encoderBlocks = [];
        _decoderBlocks = [];
        _skipProjections = [];

        InitializeLayers();
    }

    [MemberNotNull(nameof(_patchEmbed), nameof(_timeEmbed1), nameof(_timeEmbed2),
                   nameof(_middleBlock), nameof(_finalNorm), nameof(_outputProj))]
    private void InitializeLayers()
    {
        int patchDim = _inputChannels * _patchSize * _patchSize;
        int timeEmbedDim = _hiddenSize * 4;
        int halfLayers = _numLayers / 2;

        // Patch embedding
        _patchEmbed = new DenseLayer<T>(patchDim, _hiddenSize, activationFunction: null);

        // Time embedding MLP
        _timeEmbed1 = new DenseLayer<T>(
            _hiddenSize, timeEmbedDim,
            (IActivationFunction<T>)new SiLUActivation<T>());
        _timeEmbed2 = new DenseLayer<T>(timeEmbedDim, _hiddenSize, activationFunction: null);

        // Encoder blocks
        for (int i = 0; i < halfLayers; i++)
        {
            _encoderBlocks.Add(CreateBlock());
        }

        // Middle block
        _middleBlock = CreateBlock();

        // Decoder blocks with skip projections
        for (int i = 0; i < halfLayers; i++)
        {
            _decoderBlocks.Add(CreateBlock());
            // Skip projection: [2*hidden] → [hidden]
            _skipProjections.Add(new DenseLayer<T>(_hiddenSize * 2, _hiddenSize, activationFunction: null));
        }

        // Final norm and output
        _finalNorm = new LayerNormalizationLayer<T>(_hiddenSize);
        int outPatchDim = _inputChannels * _patchSize * _patchSize;
        _outputProj = new DenseLayer<T>(_hiddenSize, outPatchDim, activationFunction: null);

        // Position embeddings
        int maxPatches = 256; // 32x32 latent / patch_size=2 → 16x16 = 256
        var posEmbData = new T[maxPatches * _hiddenSize];
        for (int i = 0; i < posEmbData.Length; i++)
        {
            double u1 = 1.0 - RandomGenerator.NextDouble();
            double u2 = RandomGenerator.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            posEmbData[i] = NumOps.FromDouble(normal * 0.02);
        }
        _posEmbed = new Tensor<T>(new[] { 1, maxPatches, _hiddenSize }, new Vector<T>(posEmbData));
    }

    private UViTBlock CreateBlock()
    {
        return new UViTBlock
        {
            Norm1 = new LayerNormalizationLayer<T>(_hiddenSize),
            Attention = new SelfAttentionLayer<T>(256, _hiddenSize, _numHeads, activationFunction: null),
            Norm2 = new LayerNormalizationLayer<T>(_hiddenSize),
            MLP1 = new DenseLayer<T>(_hiddenSize, _hiddenSize * 4, (IActivationFunction<T>)new GELUActivation<T>()),
            MLP2 = new DenseLayer<T>(_hiddenSize * 4, _hiddenSize, activationFunction: null)
        };
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoise(Tensor<T> noisySample, int timestep, Tensor<T>? conditioning = null)
    {
        _lastInput = noisySample;
        var timeEmbed = GetTimestepEmbedding(timestep);
        timeEmbed = ProjectTimeEmbedding(timeEmbed);
        return Forward(noisySample, timeEmbed, conditioning);
    }

    /// <inheritdoc />
    public override Tensor<T> PredictNoiseWithEmbedding(Tensor<T> noisySample, Tensor<T> timeEmbedding, Tensor<T>? conditioning = null)
    {
        _lastInput = noisySample;
        return Forward(noisySample, timeEmbedding, conditioning);
    }

    private Tensor<T> ProjectTimeEmbedding(Tensor<T> timeEmbed)
    {
        var x = _timeEmbed1.Forward(timeEmbed);
        x = _timeEmbed2.Forward(x);
        return x;
    }

    private Tensor<T> Forward(Tensor<T> x, Tensor<T> timeEmbed, Tensor<T>? conditioning)
    {
        var shape = x.Shape;
        var batch = shape[0];

        // Patchify and embed
        var patches = Patchify(x);
        patches = _patchEmbed.Forward(patches);

        // Add position embeddings
        if (_posEmbed != null)
        {
            var posData = new T[patches.Shape[0] * patches.Shape[1] * patches.Shape[2]];
            for (int i = 0; i < posData.Length; i++)
            {
                int seqIdx = (i / patches.Shape[2]) % patches.Shape[1];
                int dimIdx = i % patches.Shape[2];
                if (seqIdx < _posEmbed.Shape[1] && dimIdx < _posEmbed.Shape[2])
                {
                    posData[i] = NumOps.Add(
                        patches.AsSpan()[i],
                        _posEmbed[0, seqIdx, dimIdx]);
                }
                else
                {
                    posData[i] = patches.AsSpan()[i];
                }
            }
            patches = new Tensor<T>(patches.Shape, new Vector<T>(posData));
        }

        // Add time embedding (broadcast to all tokens)
        patches = AddTimeToPatches(patches, timeEmbed);

        int halfLayers = _numLayers / 2;

        // Encoder: store activations for skip connections
        var skipActivations = new Tensor<T>[halfLayers];
        for (int i = 0; i < halfLayers; i++)
        {
            skipActivations[i] = CloneTensor(patches);
            patches = ApplyBlock(_encoderBlocks[i], patches);
        }

        // Middle block
        patches = ApplyBlock(_middleBlock, patches);

        // Decoder with skip connections
        for (int i = 0; i < halfLayers; i++)
        {
            int skipIdx = halfLayers - 1 - i;
            // Concatenate along feature dimension and project
            patches = ConcatenateTensors(patches, skipActivations[skipIdx]);
            patches = _skipProjections[i].Forward(patches);
            patches = ApplyBlock(_decoderBlocks[i], patches);
        }

        // Final norm and unpatchify
        patches = _finalNorm.Forward(patches);
        patches = _outputProj.Forward(patches);

        return Unpatchify(patches, batch, shape.Length > 2 ? shape[2] : 32, shape.Length > 3 ? shape[3] : 32);
    }

    private Tensor<T> ApplyBlock(UViTBlock block, Tensor<T> x)
    {
        // Self-attention with residual
        var normed = block.Norm1!.Forward(x);
        var attn = block.Attention!.Forward(normed);
        x = AddTensors(x, attn);

        // MLP with residual
        normed = block.Norm2!.Forward(x);
        var mlp = block.MLP1!.Forward(normed);
        mlp = block.MLP2!.Forward(mlp);
        x = AddTensors(x, mlp);

        return x;
    }

    #region Tensor Utilities

    private Tensor<T> Patchify(Tensor<T> x)
    {
        var shape = x.Shape;
        var batch = shape[0];
        var height = shape.Length > 2 ? shape[2] : 1;
        var width = shape.Length > 3 ? shape[3] : 1;

        int patchH = height / _patchSize;
        int patchW = width / _patchSize;
        int numPatches = patchH * patchW;
        int patchDim = _inputChannels * _patchSize * _patchSize;

        var result = new T[batch * numPatches * patchDim];
        var xSpan = x.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int patchIdx = ph * patchW + pw;
                    int outBase = (b * numPatches + patchIdx) * patchDim;
                    int d = 0;

                    for (int c = 0; c < _inputChannels; c++)
                    {
                        for (int dy = 0; dy < _patchSize; dy++)
                        {
                            for (int dx = 0; dx < _patchSize; dx++)
                            {
                                int h = ph * _patchSize + dy;
                                int w = pw * _patchSize + dx;
                                int inIdx = ((b * _inputChannels + c) * height + h) * width + w;
                                result[outBase + d] = inIdx < xSpan.Length ? xSpan[inIdx] : NumOps.Zero;
                                d++;
                            }
                        }
                    }
                }
            }
        }

        return new Tensor<T>(new[] { batch, numPatches, patchDim }, new Vector<T>(result));
    }

    private Tensor<T> Unpatchify(Tensor<T> patches, int batch, int height, int width)
    {
        int patchH = height / _patchSize;
        int patchW = width / _patchSize;
        int outPatchDim = _inputChannels * _patchSize * _patchSize;

        var result = new T[batch * _inputChannels * height * width];
        var pSpan = patches.AsSpan();

        for (int b = 0; b < batch; b++)
        {
            for (int ph = 0; ph < patchH; ph++)
            {
                for (int pw = 0; pw < patchW; pw++)
                {
                    int patchIdx = ph * patchW + pw;
                    int inBase = (b * patchH * patchW + patchIdx) * outPatchDim;
                    int d = 0;

                    for (int c = 0; c < _inputChannels; c++)
                    {
                        for (int dy = 0; dy < _patchSize; dy++)
                        {
                            for (int dx = 0; dx < _patchSize; dx++)
                            {
                                int h = ph * _patchSize + dy;
                                int w = pw * _patchSize + dx;
                                int outIdx = ((b * _inputChannels + c) * height + h) * width + w;
                                if (outIdx < result.Length && inBase + d < pSpan.Length)
                                    result[outIdx] = pSpan[inBase + d];
                                d++;
                            }
                        }
                    }
                }
            }
        }

        return new Tensor<T>(new[] { batch, _inputChannels, height, width }, new Vector<T>(result));
    }

    private static Tensor<T> AddTimeToPatches(Tensor<T> patches, Tensor<T> timeEmbed)
    {
        var data = new T[patches.AsSpan().Length];
        var pSpan = patches.AsSpan();
        var tSpan = timeEmbed.AsSpan();

        for (int i = 0; i < data.Length; i++)
        {
            int dimIdx = i % (patches.Shape.Length > 2 ? patches.Shape[2] : patches.Shape[^1]);
            data[i] = NumOps.Add(pSpan[i], dimIdx < tSpan.Length ? tSpan[dimIdx] : NumOps.Zero);
        }

        return new Tensor<T>(patches.Shape, new Vector<T>(data));
    }

    private static Tensor<T> CloneTensor(Tensor<T> t)
    {
        var span = t.AsSpan();
        var data = new T[span.Length];
        for (int i = 0; i < span.Length; i++) data[i] = span[i];
        return new Tensor<T>(t.Shape, new Vector<T>(data));
    }

    private static Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        int len = Math.Min(aSpan.Length, bSpan.Length);
        var data = new T[aSpan.Length];
        for (int i = 0; i < len; i++) data[i] = NumOps.Add(aSpan[i], bSpan[i]);
        for (int i = len; i < aSpan.Length; i++) data[i] = aSpan[i];
        return new Tensor<T>(a.Shape, new Vector<T>(data));
    }

    private static Tensor<T> ConcatenateTensors(Tensor<T> a, Tensor<T> b)
    {
        // Concatenate along feature dimension (last dim)
        var aShape = a.Shape;
        var bShape = b.Shape;
        int batch = aShape[0];
        int seqLen = aShape.Length > 1 ? aShape[1] : 1;
        int dimA = aShape.Length > 2 ? aShape[2] : aShape[^1];
        int dimB = bShape.Length > 2 ? bShape[2] : bShape[^1];
        int dimOut = dimA + dimB;

        var data = new T[batch * seqLen * dimOut];
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();

        for (int n = 0; n < batch * seqLen; n++)
        {
            for (int d = 0; d < dimA; d++)
            {
                int aIdx = n * dimA + d;
                data[n * dimOut + d] = aIdx < aSpan.Length ? aSpan[aIdx] : NumOps.Zero;
            }
            for (int d = 0; d < dimB; d++)
            {
                int bIdx = n * dimB + d;
                data[n * dimOut + dimA + d] = bIdx < bSpan.Length ? bSpan[bIdx] : NumOps.Zero;
            }
        }

        return new Tensor<T>(new[] { batch, seqLen, dimOut }, new Vector<T>(data));
    }

    #endregion

    #region IParameterizable

    /// <inheritdoc />
    public override int ParameterCount
    {
        get
        {
            int count = _patchEmbed.ParameterCount + _timeEmbed1.ParameterCount + _timeEmbed2.ParameterCount;

            foreach (var block in _encoderBlocks)
                count += GetBlockParamCount(block);

            count += GetBlockParamCount(_middleBlock);

            for (int i = 0; i < _decoderBlocks.Count; i++)
            {
                count += GetBlockParamCount(_decoderBlocks[i]);
                count += _skipProjections[i].ParameterCount;
            }

            count += _finalNorm.ParameterCount + _outputProj.ParameterCount;
            return count;
        }
    }

    private static int GetBlockParamCount(UViTBlock block)
    {
        int c = 0;
        if (block.Norm1 != null) c += block.Norm1.ParameterCount;
        if (block.Attention != null) c += block.Attention.ParameterCount;
        if (block.Norm2 != null) c += block.Norm2.ParameterCount;
        if (block.MLP1 != null) c += block.MLP1.ParameterCount;
        if (block.MLP2 != null) c += block.MLP2.ParameterCount;
        return c;
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();
        AddLayerParams(allParams, _patchEmbed);
        AddLayerParams(allParams, _timeEmbed1);
        AddLayerParams(allParams, _timeEmbed2);

        foreach (var block in _encoderBlocks) AddBlockParams(allParams, block);
        AddBlockParams(allParams, _middleBlock);

        for (int i = 0; i < _decoderBlocks.Count; i++)
        {
            AddBlockParams(allParams, _decoderBlocks[i]);
            AddLayerParams(allParams, _skipProjections[i]);
        }

        AddLayerParams(allParams, _finalNorm);
        AddLayerParams(allParams, _outputProj);

        var result = new Vector<T>(allParams.Count);
        for (int i = 0; i < allParams.Count; i++) result[i] = allParams[i];
        return result;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        int offset = 0;
        offset = SetLayerParams(_patchEmbed, parameters, offset);
        offset = SetLayerParams(_timeEmbed1, parameters, offset);
        offset = SetLayerParams(_timeEmbed2, parameters, offset);
    }

    private static void AddLayerParams(List<T> list, ILayer<T> layer)
    {
        var p = layer.GetParameters();
        for (int i = 0; i < p.Length; i++) list.Add(p[i]);
    }

    private static void AddBlockParams(List<T> list, UViTBlock block)
    {
        if (block.Norm1 != null) AddLayerParams(list, block.Norm1);
        if (block.Attention != null) AddLayerParams(list, block.Attention);
        if (block.Norm2 != null) AddLayerParams(list, block.Norm2);
        if (block.MLP1 != null) AddLayerParams(list, block.MLP1);
        if (block.MLP2 != null) AddLayerParams(list, block.MLP2);
    }

    private static int SetLayerParams(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = layer.ParameterCount;
        var p = new T[count];
        for (int i = 0; i < count && offset + i < parameters.Length; i++)
            p[i] = parameters[offset + i];
        layer.SetParameters(new Vector<T>(p));
        return offset + count;
    }

    #endregion

    /// <inheritdoc />
    public override INoisePredictor<T> Clone()
    {
        var clone = new UViTNoisePredictor<T>(
            inputChannels: _inputChannels,
            hiddenSize: _hiddenSize,
            numLayers: _numLayers,
            numHeads: _numHeads,
            patchSize: _patchSize,
            contextDim: _contextDim);
        clone.SetParameters(GetParameters());
        return clone;
    }

    /// <inheritdoc />
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy() => Clone();

    /// <summary>
    /// Block structure for U-ViT transformer layers.
    /// </summary>
    public class UViTBlock
    {
        public LayerNormalizationLayer<T>? Norm1 { get; set; }
        public SelfAttentionLayer<T>? Attention { get; set; }
        public LayerNormalizationLayer<T>? Norm2 { get; set; }
        public DenseLayer<T>? MLP1 { get; set; }
        public DenseLayer<T>? MLP2 { get; set; }
    }
}
