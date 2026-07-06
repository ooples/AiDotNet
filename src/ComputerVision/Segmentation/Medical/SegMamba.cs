using System.IO;
using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Layers.SSM;
using AiDotNet.Optimizers;
using Microsoft.ML.OnnxRuntime;
using OnnxTensors = Microsoft.ML.OnnxRuntime.Tensors;

namespace AiDotNet.ComputerVision.Segmentation.Medical;

/// <summary>
/// SegMamba: long-range sequential modeling Mamba for 3D medical image segmentation
/// (Xing et al., 2024, arXiv:2401.13560).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>Architecture (paper-faithful).</b> SegMamba is a 3D U-Net whose encoder replaces the usual
/// self-attention / convolution stack with a Mamba state-space backbone:
/// </para>
/// <list type="number">
///   <item><b>Stem</b>: a single 7×7×7 stride-2 3D convolution that embeds the input volume into
///         the first feature scale.</item>
///   <item><b>Encoder</b>: four hierarchical stages. Stage <c>i</c> applies (for <c>i&gt;0</c>) an
///         InstanceNorm + 2×2×2 stride-2 downsampling convolution, then a <b>Gated Spatial
///         Convolution (GSC)</b> module for local feature enhancement, then <c>depths[i]</c>
///         <b>TSMamba</b> blocks. Each TSMamba block normalizes the feature volume and runs a
///         <b>Tri-orientated Mamba (ToM)</b>: the 3-D feature is flattened to a token sequence and
///         scanned by a Mamba SSM in three orientations — forward, reverse, and inter-slice — whose
///         outputs are summed (§3.2 of the paper).</item>
///   <item><b>Decoder</b>: a CNN decoder that trilinearly upsamples and fuses the four encoder
///         feature scales through skip connections, ending in a 1×1×1 convolution to the class
///         logits at full input resolution.</item>
/// </list>
/// <para>
/// The Mamba scan gives linear complexity in the number of voxels, which is what makes whole-volume
/// 3D segmentation tractable where attention would be quadratic.
/// </para>
/// <para>
/// <b>For Beginners:</b> This model labels every voxel of a 3-D medical scan (e.g. a CT volume) with
/// the organ/structure it belongs to. It "reads" the whole volume as a long sequence in several
/// directions so it can relate far-apart regions cheaply.
/// </para>
/// <para><b>Reference:</b> Xing et al., "SegMamba: Long-range Sequential Modeling Mamba For 3D
/// Medical Image Segmentation", arXiv:2401.13560, 2024.</para>
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Segmentation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("SegMamba: Long-range Sequential Modeling Mamba For 3D Medical Image Segmentation", "https://arxiv.org/abs/2401.13560", Year = 2024, Authors = "Xing et al.")]
public class SegMamba<T> : NeuralNetworkBase<T>, IMedicalSegmentation<T>
{
    private readonly SegMambaOptions _options;
    public override ModelOptions GetOptions() => _options;

    #region Fields
    private readonly int _inChannels, _numClasses;
    private readonly int[] _channelDims;
    private readonly int[] _depths;
    private readonly int _stateDim;
    private readonly double _dropRate;
    private readonly bool _useNativeMode;
    private readonly string? _onnxModelPath;
    private InferenceSession? _onnxSession;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _disposed;

    // --- Typed layer references for the custom (skip-connected) forward pass.
    // All of these are ALSO held in the base Layers list (parameter management);
    // they are re-derived from Layers after deserialization via ExtractLayerReferences.
    private Conv3DLayer<T>? _stem;
    private readonly List<InstanceNormalizationLayer<T>> _downNorms = new();
    private readonly List<Conv3DLayer<T>> _downConvs = new();
    private readonly List<GscModule> _gsc = new();
    private readonly List<List<TomModule>> _tom = new();
    private readonly List<InstanceNormalizationLayer<T>> _encNorms = new();
    private readonly List<Upsample3DLayer<T>> _decUps = new();
    private readonly List<Conv3DLayer<T>> _decConvs = new();
    private readonly List<InstanceNormalizationLayer<T>> _decNorms = new();
    private Conv3DLayer<T>? _outConv;
    #endregion

    private sealed class GscModule
    {
        public readonly Conv3DLayer<T> Proj;
        public readonly InstanceNormalizationLayer<T> NormA;
        public readonly Conv3DLayer<T> Proj2;
        public readonly InstanceNormalizationLayer<T> NormB;
        public readonly Conv3DLayer<T> Proj3;
        public readonly InstanceNormalizationLayer<T> NormC;

        public GscModule(Conv3DLayer<T> proj, InstanceNormalizationLayer<T> normA,
            Conv3DLayer<T> proj2, InstanceNormalizationLayer<T> normB,
            Conv3DLayer<T> proj3, InstanceNormalizationLayer<T> normC)
        {
            Proj = proj; NormA = normA; Proj2 = proj2; NormB = normB; Proj3 = proj3; NormC = normC;
        }
    }

    private sealed class TomModule
    {
        public readonly InstanceNormalizationLayer<T> Norm;
        public readonly MambaBlock<T> Forward;
        public readonly MambaBlock<T> Reverse;
        public readonly MambaBlock<T> InterSlice;

        public TomModule(InstanceNormalizationLayer<T> norm, MambaBlock<T> forward,
            MambaBlock<T> reverse, MambaBlock<T> interSlice)
        {
            Norm = norm; Forward = forward; Reverse = reverse; InterSlice = interSlice;
        }
    }

    #region Properties
    public override bool SupportsTraining => _useNativeMode;
    internal bool UseNativeMode => _useNativeMode;
    internal int NumClasses => _numClasses;
    #endregion

    #region Constructors
    /// <summary>Initializes SegMamba in native (trainable) mode.</summary>
    public SegMamba(NeuralNetworkArchitecture<T> architecture,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null, int numClasses = 14,
        double dropRate = 0,
        SegMambaOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new SegMambaOptions(); Options = _options;
        _inChannels = architecture.InputDepth > 0 ? architecture.InputDepth : 1;
        _numClasses = numClasses; _dropRate = dropRate;
        _useNativeMode = true; _onnxModelPath = null;
        // Paper-faithful encoder widths/depths (Xing et al. 2024, §4): feature dims
        // [48, 96, 192, 384] with two TSMamba blocks per stage.
        _channelDims = [48, 96, 192, 384];
        _depths = [2, 2, 2, 2];
        _stateDim = 16;
        // SegMamba trains with AdamW at a small LR (paper §4.2 uses 1e-4 with
        // warmup/poly decay); the framework default 1e-3 is too aggressive for the
        // hybrid conv-Mamba encoder.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this, new Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>> { InitialLearningRate = 1e-4 });
        InitializeLayers();
    }

    /// <summary>Initializes SegMamba in ONNX (inference-only) mode.</summary>
    public SegMamba(NeuralNetworkArchitecture<T> architecture, string onnxModelPath,
        int numClasses = 14,
        SegMambaOptions? options = null)
        : base(architecture, new CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new SegMambaOptions(); Options = _options;
        if (string.IsNullOrWhiteSpace(onnxModelPath))
            throw new ArgumentException("ONNX model path cannot be null or empty.", nameof(onnxModelPath));
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"SegMamba ONNX model not found: {onnxModelPath}");
        _inChannels = architecture.InputDepth > 0 ? architecture.InputDepth : 1;
        _numClasses = numClasses; _dropRate = 0;
        _useNativeMode = false; _onnxModelPath = onnxModelPath; _optimizer = null;
        _channelDims = [48, 96, 192, 384];
        _depths = [2, 2, 2, 2];
        _stateDim = 16;
        try { _onnxSession = new InferenceSession(onnxModelPath); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to load SegMamba ONNX model: {ex.Message}", ex); }
        InitializeLayers();
    }
    #endregion

    #region Public Methods
    /// <summary>Runs a forward pass to produce segmentation logits.</summary>
    /// <param name="input">Input volume [C, D, H, W] or [B, C, D, H, W].</param>
    protected override Tensor<T> PredictCore(Tensor<T> input) => _useNativeMode ? Forward(input) : PredictOnnx(input);

    /// <inheritdoc/>
    public override Tensor<T> ForwardForTraining(Tensor<T> input) => Forward(input);

    /// <summary>Performs one training step.</summary>
    /// <exception cref="InvalidOperationException">Thrown when called on an ONNX-mode model.</exception>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode) throw new InvalidOperationException("Training is not supported in ONNX mode. Use the native mode constructor for training.");
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }
    #endregion

    #region Forward
    private Tensor<T> Forward(Tensor<T> input)
    {
        bool hasBatch = input.Rank == 5;
        if (!hasBatch)
        {
            if (input.Rank != 4)
                throw new ArgumentException(
                    $"SegMamba is a 3D model: input must be rank-4 [C, D, H, W] or rank-5 [B, C, D, H, W], got rank {input.Rank}.",
                    nameof(input));
            input = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]]);
        }

        // ---- Encoder: stem -> 4 stages, collecting one skip per stage. ----
        var skips = new Tensor<T>[_channelDims.Length];
        var cur = _stem!.Forward(input);
        for (int stage = 0; stage < _channelDims.Length; stage++)
        {
            if (stage > 0)
            {
                cur = _downNorms[stage - 1].Forward(cur);
                cur = _downConvs[stage - 1].Forward(cur);
            }

            cur = ApplyGsc(_gsc[stage], cur);

            for (int block = 0; block < _depths[stage]; block++)
                cur = ApplyTsMamba(_tom[stage][block], cur);

            skips[stage] = _encNorms[stage].Forward(cur);
            // The next downsample consumes the pre-norm stage output `cur`.
        }

        // ---- Decoder: upsample + skip-concat + conv, from the coarsest scale up. ----
        var d = skips[^1];
        int convIdx = 0;
        for (int stage = _channelDims.Length - 2; stage >= 0; stage--)
        {
            d = _decUps[convIdx].Forward(d);
            // Align the upsampled tensor to the skip's spatial dims before the
            // channel concat. Each encoder downsample computes ceil(n/2), so a
            // x2 upsample yields 2*ceil(n/2) >= n — one voxel larger on odd
            // sizes (e.g. D: 3 -> 2 -> 1, then 1 -> 2 vs skip 1). The official
            // SegMamba (Xing et al. 2024) trains on 2^depth-divisible crops
            // where the sizes match exactly; for general inputs we follow the
            // U-Net convention (Ronneberger et al. 2015) of cropping the larger
            // tensor to the skip so the concat is well-formed.
            d = CropToSpatial(d, skips[stage]);
            d = Engine.TensorConcatenate([d, skips[stage]], axis: 1);
            d = ApplyConvBlock(_decConvs[convIdx], _decNorms[convIdx], d);
            convIdx++;
        }

        // Final upsample back to full input resolution + conv block. Crop to the
        // exact input spatial dims so the logits match the input volume on odd
        // sizes (the x2 upsample of ceil(n/2) overshoots by one voxel).
        d = _decUps[convIdx].Forward(d);
        d = CropToSpatial(d, input);
        d = ApplyConvBlock(_decConvs[convIdx], _decNorms[convIdx], d);

        // 1x1x1 projection to class logits.
        var logits = _outConv!.Forward(d);

        if (!hasBatch)
        {
            var s = logits._shape;
            logits = Engine.Reshape(logits, [s[1], s[2], s[3], s[4]]);
        }
        return logits;
    }

    /// <summary>
    /// Center-crops the spatial dims (D, H, W = axes 2..4) of <paramref name="t"/>
    /// down to the reference tensor's spatial dims when larger. Cropping (rather
    /// than padding) is sufficient because every encoder downsample produces
    /// ceil(n/2), so a x2 upsample is always >= the matching skip / input size.
    /// Uses Engine.TensorNarrow, which stays on the autodiff tape.
    /// </summary>
    private Tensor<T> CropToSpatial(Tensor<T> t, Tensor<T> reference)
    {
        bool cropped = false;
        for (int axis = 2; axis <= 4; axis++)
        {
            int cur = t.Shape[axis];
            int target = reference.Shape[axis];
            if (cur > target)
            {
                int start = (cur - target) / 2;
                t = Engine.TensorNarrow(t, dim: axis, start: start, length: target);
                cropped = true;
            }
        }

        // TensorNarrow returns a strided VIEW; the downstream concat/conv reads
        // raw Data and requires contiguous storage. Materialize once after all
        // axis crops (same pattern as VideoUNetPredictor's Permute().Contiguous()).
        return cropped ? t.Contiguous() : t;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// The base implementation runs the flat <c>Layers</c> list sequentially on the raw input,
    /// which does not match SegMamba's skip-connected encoder/decoder graph (it would feed the
    /// wrong channel counts between stages). Capture activations along the real encoder path.
    /// </remarks>
    public override Dictionary<string, Tensor<T>> GetNamedLayerActivations(Tensor<T> input)
    {
        var activations = new Dictionary<string, Tensor<T>>();
        if (!_useNativeMode) return activations;

        var x = input.Rank == 5
            ? input
            : Engine.Reshape(input, [1, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3]]);

        var cur = _stem!.Forward(x);
        activations["stem"] = cur.Clone();
        for (int stage = 0; stage < _channelDims.Length; stage++)
        {
            if (stage > 0)
            {
                cur = _downNorms[stage - 1].Forward(cur);
                cur = _downConvs[stage - 1].Forward(cur);
            }
            cur = ApplyGsc(_gsc[stage], cur);
            for (int block = 0; block < _depths[stage]; block++)
                cur = ApplyTsMamba(_tom[stage][block], cur);
            activations[$"stage{stage}"] = _encNorms[stage].Forward(cur).Clone();
        }
        return activations;
    }

    /// <summary>Gated Spatial Convolution (paper §3.3): two stacked 3×3×3 conv-norm-ReLU
    /// branches summed with a 1×1×1 conv-norm-ReLU branch, plus a residual connection.</summary>
    private Tensor<T> ApplyGsc(GscModule g, Tensor<T> x)
    {
        var residual = x;
        var x1 = Engine.ReLU(g.NormA.Forward(g.Proj.Forward(x)));
        x1 = Engine.ReLU(g.NormB.Forward(g.Proj2.Forward(x1)));
        var x2 = Engine.ReLU(g.NormC.Forward(g.Proj3.Forward(x)));
        return Engine.TensorAdd(Engine.TensorAdd(x1, x2), residual);
    }

    /// <summary>One TSMamba block: residual + Tri-orientated Mamba over the normalized volume.</summary>
    private Tensor<T> ApplyTsMamba(TomModule m, Tensor<T> x)
    {
        var normed = m.Norm.Forward(x);
        var tom = ApplyTriOrientatedMamba(m, normed);
        return Engine.TensorAdd(x, tom);
    }

    /// <summary>Conv → InstanceNorm → ReLU block (decoder).</summary>
    private Tensor<T> ApplyConvBlock(Conv3DLayer<T> conv, InstanceNormalizationLayer<T> norm, Tensor<T> x)
        => Engine.ReLU(norm.Forward(conv.Forward(x)));

    /// <summary>
    /// Tri-orientated Mamba (ToM, paper §3.2): flatten the 3-D feature volume into a token
    /// sequence and scan it with a Mamba SSM in three orientations — forward, reverse, and
    /// inter-slice — summing the three results. Every op is tape-aware so gradients reach all
    /// three SSM scans.
    /// </summary>
    private Tensor<T> ApplyTriOrientatedMamba(TomModule m, Tensor<T> x)
    {
        int b = x.Shape[0], c = x.Shape[1], dD = x.Shape[2], dH = x.Shape[3], dW = x.Shape[4];
        int len = dD * dH * dW;

        // Forward scan: [B, C, D, H, W] -> [B, L, C] in (D,H,W) row-major order.
        var seqF = Engine.TensorPermute(Engine.Reshape(x, [b, c, len]), [0, 2, 1]); // [B, L, C]
        var outF = m.Forward.Forward(seqF);

        // Reverse scan: gather the sequence backwards, scan, gather back to forward order.
        var revIdx = BuildReverseIndices(len);
        var seqR = Engine.TensorGather(seqF, revIdx, axis: 1);
        var outR = Engine.TensorGather(m.Reverse.Forward(seqR), revIdx, axis: 1);

        var frSeq = Engine.TensorAdd(outF, outR);                                  // [B, L, C]
        var fr = Engine.Reshape(Engine.TensorPermute(frSeq, [0, 2, 1]), [b, c, dD, dH, dW]);

        // Inter-slice scan: permute the volume so the scan crosses slices first
        // ([B,C,D,H,W] -> [B,C,W,H,D]), flatten, scan, then map back to [B,C,D,H,W].
        var xp = Engine.TensorPermute(x, [0, 1, 4, 3, 2]);                          // [B, C, W, H, D]
        var seqI = Engine.TensorPermute(Engine.Reshape(xp, [b, c, len]), [0, 2, 1]);
        var outI = m.InterSlice.Forward(seqI);
        var iVol = Engine.Reshape(Engine.TensorPermute(outI, [0, 2, 1]), [b, c, dW, dH, dD]);
        iVol = Engine.TensorPermute(iVol, [0, 1, 4, 3, 2]);                         // back to [B, C, D, H, W]

        return Engine.TensorAdd(fr, iVol);
    }

    private static Tensor<int> BuildReverseIndices(int len)
    {
        var idx = new int[len];
        for (int i = 0; i < len; i++) idx[i] = len - 1 - i;
        return new Tensor<int>(idx, [len]);
    }

    private Tensor<T> PredictOnnx(Tensor<T> input)
    {
        if (_onnxSession is null) throw new InvalidOperationException("ONNX session is not initialized.");
        bool hasBatch = input.Rank == 5; if (!hasBatch) input = AddBatchDimension(input);
        var inputData = new float[input.Length];
        for (int i = 0; i < input.Length; i++) inputData[i] = Convert.ToSingle(input.Data.Span[i]);
        var onnxInput = new OnnxTensors.DenseTensor<float>(inputData, input._shape);
        string inputName = _onnxSession.InputMetadata.Keys.FirstOrDefault() ?? "images";
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, onnxInput) };
        using var results = _onnxSession.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();
        var outputData = new T[outputTensor.Length];
        for (int i = 0; i < outputTensor.Length; i++) outputData[i] = NumOps.FromDouble(outputTensor.GetValue(i));
        var result = new Tensor<T>(outputTensor.Dimensions.ToArray(), new Vector<T>(outputData));
        if (!hasBatch) result = RemoveBatchDimension(result); return result;
    }

    private Tensor<T> AddBatchDimension(Tensor<T> tensor)
    { var s = new int[tensor.Shape.Length + 1]; s[0] = 1; for (int i = 0; i < tensor.Shape.Length; i++) s[i + 1] = tensor.Shape[i]; var result = new Tensor<T>(s); tensor.Data.Span.CopyTo(result.Data.Span); return result; }

    private Tensor<T> RemoveBatchDimension(Tensor<T> tensor)
    { int[] s = new int[tensor.Shape.Length - 1]; for (int i = 0; i < s.Length; i++) s[i] = tensor.Shape[i + 1]; var r = new Tensor<T>(s); tensor.Data.Span.CopyTo(r.Data.Span); return r; }
    #endregion

    #region Layer construction
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) { ClearLayers(); return; }
        ClearLayers();
        _downNorms.Clear(); _downConvs.Clear(); _gsc.Clear(); _tom.Clear();
        _encNorms.Clear(); _decUps.Clear(); _decConvs.Clear(); _decNorms.Clear();

        IActivationFunction<T> identity = new IdentityActivation<T>();

        // Stem: 7x7x7 stride-2 conv (channel count inferred from input on first forward).
        _stem = new Conv3DLayer<T>(_channelDims[0], kernelSize: 7, stride: 2, padding: 3, identity);
        Layers.Add(_stem);

        for (int stage = 0; stage < _channelDims.Length; stage++)
        {
            int dim = _channelDims[stage];

            if (stage > 0)
            {
                var dn = new InstanceNormalizationLayer<T>(_channelDims[stage - 1]);
                var dc = new Conv3DLayer<T>(dim, kernelSize: 2, stride: 2, padding: 0, identity);
                _downNorms.Add(dn); _downConvs.Add(dc);
                Layers.Add(dn); Layers.Add(dc);
            }

            var gsc = new GscModule(
                new Conv3DLayer<T>(dim, 3, 1, 1, identity),
                new InstanceNormalizationLayer<T>(dim),
                new Conv3DLayer<T>(dim, 3, 1, 1, identity),
                new InstanceNormalizationLayer<T>(dim),
                new Conv3DLayer<T>(dim, 1, 1, 0, identity),
                new InstanceNormalizationLayer<T>(dim));
            _gsc.Add(gsc);
            Layers.Add(gsc.Proj); Layers.Add(gsc.NormA); Layers.Add(gsc.Proj2);
            Layers.Add(gsc.NormB); Layers.Add(gsc.Proj3); Layers.Add(gsc.NormC);

            var stageToms = new List<TomModule>();
            for (int block = 0; block < _depths[stage]; block++)
            {
                var tom = new TomModule(
                    new InstanceNormalizationLayer<T>(dim),
                    new MambaBlock<T>(sequenceLength: 1, modelDimension: dim, stateDimension: _stateDim),
                    new MambaBlock<T>(sequenceLength: 1, modelDimension: dim, stateDimension: _stateDim),
                    new MambaBlock<T>(sequenceLength: 1, modelDimension: dim, stateDimension: _stateDim));
                stageToms.Add(tom);
                Layers.Add(tom.Norm); Layers.Add(tom.Forward); Layers.Add(tom.Reverse); Layers.Add(tom.InterSlice);
            }
            _tom.Add(stageToms);

            var en = new InstanceNormalizationLayer<T>(dim);
            _encNorms.Add(en); Layers.Add(en);
        }

        // Decoder: one (upsample, conv-block) per coarse->fine transition + a final full-res block.
        int decBlocks = _channelDims.Length; // 3 skip-fusions + 1 final full-res
        for (int i = 0; i < decBlocks; i++)
        {
            int outDim = i < _channelDims.Length - 1 ? _channelDims[_channelDims.Length - 2 - i] : _channelDims[0];
            var up = new Upsample3DLayer<T>(2);
            var conv = new Conv3DLayer<T>(outDim, 3, 1, 1, identity);
            var norm = new InstanceNormalizationLayer<T>(outDim);
            _decUps.Add(up); _decConvs.Add(conv); _decNorms.Add(norm);
            Layers.Add(up); Layers.Add(conv); Layers.Add(norm);
        }

        _outConv = new Conv3DLayer<T>(_numClasses, 1, 1, 0, identity);
        Layers.Add(_outConv);
    }

    /// <summary>
    /// Re-derives the typed sub-layer references from the canonical <see cref="NeuralNetworkBase{T}.Layers"/>
    /// list after deserialization rebuilds it. Without this a cloned/loaded model would run the
    /// constructor's randomly-initialized layers in Forward while the loaded weights sit unused.
    /// Walks <c>Layers</c> in exactly the order <see cref="InitializeLayers"/> appended them.
    /// </summary>
    private void ExtractLayerReferences()
    {
        _downNorms.Clear(); _downConvs.Clear(); _gsc.Clear(); _tom.Clear();
        _encNorms.Clear(); _decUps.Clear(); _decConvs.Clear(); _decNorms.Clear();

        int idx = 0;
        _stem = (Conv3DLayer<T>)Layers[idx++];

        for (int stage = 0; stage < _channelDims.Length; stage++)
        {
            if (stage > 0)
            {
                _downNorms.Add((InstanceNormalizationLayer<T>)Layers[idx++]);
                _downConvs.Add((Conv3DLayer<T>)Layers[idx++]);
            }

            var gsc = new GscModule(
                (Conv3DLayer<T>)Layers[idx++],
                (InstanceNormalizationLayer<T>)Layers[idx++],
                (Conv3DLayer<T>)Layers[idx++],
                (InstanceNormalizationLayer<T>)Layers[idx++],
                (Conv3DLayer<T>)Layers[idx++],
                (InstanceNormalizationLayer<T>)Layers[idx++]);
            _gsc.Add(gsc);

            var stageToms = new List<TomModule>();
            for (int block = 0; block < _depths[stage]; block++)
            {
                stageToms.Add(new TomModule(
                    (InstanceNormalizationLayer<T>)Layers[idx++],
                    (MambaBlock<T>)Layers[idx++],
                    (MambaBlock<T>)Layers[idx++],
                    (MambaBlock<T>)Layers[idx++]));
            }
            _tom.Add(stageToms);

            _encNorms.Add((InstanceNormalizationLayer<T>)Layers[idx++]);
        }

        int decBlocks = _channelDims.Length;
        for (int i = 0; i < decBlocks; i++)
        {
            _decUps.Add((Upsample3DLayer<T>)Layers[idx++]);
            _decConvs.Add((Conv3DLayer<T>)Layers[idx++]);
            _decNorms.Add((InstanceNormalizationLayer<T>)Layers[idx++]);
        }

        _outConv = (Conv3DLayer<T>)Layers[idx++];
    }
    #endregion

    #region Abstract Implementation
    public override void UpdateParameters(Vector<T> parameters)
    { int o = 0; foreach (var l in Layers) { var p = l.GetParameters(); int c = p.Length; if (c == 0) continue; if (o + c <= parameters.Length) { var n = new Vector<T>(c); for (int i = 0; i < c; i++) n[i] = parameters[o + i]; l.UpdateParameters(n); o += c; } } }

    public override ModelMetadata<T> GetModelMetadata() => new()
    {
        AdditionalInfo = new Dictionary<string, object> { { "ModelName", "SegMamba" }, { "InChannels", _inChannels }, { "NumClasses", _numClasses }, { "UseNativeMode", _useNativeMode }, { "NumLayers", Layers.Count } },
        ModelData = SerializeForMetadata()
    };

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_inChannels); writer.Write(_numClasses); writer.Write(_stateDim);
        writer.Write(_dropRate); writer.Write(_useNativeMode); writer.Write(_onnxModelPath ?? string.Empty);
        writer.Write(_channelDims.Length); foreach (int d in _channelDims) writer.Write(d);
        writer.Write(_depths.Length); foreach (int d in _depths) writer.Write(d);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32(); _ = reader.ReadInt32(); _ = reader.ReadInt32();
        _ = reader.ReadDouble(); _ = reader.ReadBoolean(); _ = reader.ReadString();
        int dc = reader.ReadInt32(); for (int i = 0; i < dc; i++) _ = reader.ReadInt32();
        int dd = reader.ReadInt32(); for (int i = 0; i < dd; i++) _ = reader.ReadInt32();

        // Layers has already been rebuilt with the loaded weights; re-point the typed
        // references at them so Forward uses the loaded layers, not the ctor's fresh ones.
        if (_useNativeMode && Layers.Count > 0)
            ExtractLayerReferences();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => _useNativeMode
        ? new SegMamba<T>(Architecture, _optimizer, LossFunction, _numClasses, _dropRate, _options)
        : new SegMamba<T>(Architecture, _onnxModelPath ?? throw new InvalidOperationException("ONNX model path not initialized."), _numClasses, _options);

    protected override void Dispose(bool disposing)
    { if (!_disposed) { if (disposing) { _onnxSession?.Dispose(); _onnxSession = null; } _disposed = true; } base.Dispose(disposing); }
    #endregion

    #region IMedicalSegmentation Implementation
    int ISegmentationModel<T>.NumClasses => _numClasses;
    int ISegmentationModel<T>.InputHeight => Architecture.InputHeight;
    int ISegmentationModel<T>.InputWidth => Architecture.InputWidth;
    bool ISegmentationModel<T>.IsOnnxMode => !_useNativeMode;
    Tensor<T> ISegmentationModel<T>.Segment(Tensor<T> image) => Predict(image);
    IReadOnlyList<string> IMedicalSegmentation<T>.SupportedModalities => ["CT", "MRI"];
    bool IMedicalSegmentation<T>.Supports3D => true;
    bool IMedicalSegmentation<T>.Supports2D => false;
    bool IMedicalSegmentation<T>.SupportsFewShot => false;

    MedicalSegmentationResult<T> IMedicalSegmentation<T>.SegmentSlice(Tensor<T> slice)
        => throw new NotSupportedException("SegMamba is a 3D model. Use SegmentVolume with a [C, D, H, W] volume.");

    MedicalSegmentationResult<T> IMedicalSegmentation<T>.SegmentVolume(Tensor<T> volume)
    {
        var output = Predict(volume); // [numClasses, D, H, W] (batch stripped) or [B, numClasses, D, H, W]
        // SegmentVolume is single-volume: the IMedicalSegmentation contract returns
        // ONE MedicalSegmentationResult, so a true batch [B, C, D, H, W] with B > 1
        // can't be represented. RemoveBatchDimension only handles B == 1; for B > 1
        // it would either throw or silently collapse multiple volumes into one mask.
        // Fail fast with a clear error so callers either feed B==1 here or pre-split
        // the batch upstream.
        if (output.Rank == 5 && output.Shape[0] != 1)
            throw new InvalidOperationException(
                $"SegmentVolume returns a single MedicalSegmentationResult; the model produced a batch " +
                $"of size {output.Shape[0]}. Pre-split the batch and call SegmentVolume per-volume, " +
                $"or use the lower-level Predict API directly to handle a batched output.");
        var logits = output.Rank == 5 ? RemoveBatchDimension(output) : output;
        int numC = logits.Shape[0], depth = logits.Shape[1], h = logits.Shape[2], w = logits.Shape[3];

        var labels = new Tensor<T>([depth, h, w]);
        var probs = Common.SegmentationTensorOps.SoftmaxAlongClassDim(logits);
        var structAccum = new Dictionary<int, (double area, double confSum)>();
        for (int z = 0; z < depth; z++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int best = 0; double bestVal = double.NegativeInfinity;
                    for (int c = 0; c < numC; c++)
                    {
                        double v = NumOps.ToDouble(logits[c, z, y, x]);
                        if (v > bestVal) { bestVal = v; best = c; }
                    }
                    labels[z, y, x] = NumOps.FromDouble(best);
                    double conf = NumOps.ToDouble(probs[best, z, y, x]);
                    if (structAccum.TryGetValue(best, out var ex))
                        structAccum[best] = (ex.area + 1, ex.confSum + conf);
                    else
                        structAccum[best] = (1, conf);
                }

        var structures = new List<SegmentedStructure>();
        foreach (var kvp in structAccum)
            if (kvp.Key != 0)
                structures.Add(new SegmentedStructure { ClassId = kvp.Key, Name = $"Class_{kvp.Key}", VolumeOrArea = kvp.Value.area, MeanConfidence = kvp.Value.confSum / kvp.Value.area });

        return new MedicalSegmentationResult<T> { Labels = labels, Probabilities = probs, Structures = structures };
    }

    MedicalSegmentationResult<T> IMedicalSegmentation<T>.SegmentFewShot(Tensor<T> queryImage, Tensor<T> supportImages, Tensor<T> supportMasks)
        => throw new NotSupportedException("SegMamba does not support few-shot segmentation. Use SegmentVolume for 3D volumes.");
    #endregion
}
