using AiDotNet.NeuralRadianceFields.NeuS;

namespace AiDotNet.Diffusion.ThreeD;

/// <summary>
/// LoFTR (Sun et al. 2021, "LoFTR: Detector-Free Local Feature Matching with Transformers", zju3dv/LoFTR): the
/// detector-free matcher One-2-3-45 uses to estimate the input's elevation from its nearby Zero123 views.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The full architecture of the reference (the copy vendored in One-2-3-45 <c>elevation_estimate/loftr/</c>):
/// </para>
/// <list type="number">
/// <item>a ResNet-18-style backbone with an FPN, giving a 1/8 coarse map and a 1/2 fine map
/// (<c>ResNetFPN_8_2</c>, grayscale input, batch normalization in evaluation mode);</item>
/// <item>the sine positional encoding added to the coarse map (<c>PositionEncodingSine</c>);</item>
/// <item>the coarse transformer: 4 self/cross pairs of linear-attention encoder layers, width 256, 8 heads;</item>
/// <item>dual-softmax coarse matching with the confidence threshold, border removal and mutual-nearest check;</item>
/// <item>5 × 5 fine windows around each coarse match merged with the projected coarse features, one self/cross
/// fine transformer pair (width 128), and expectation over the fine heatmap for sub-pixel refinement.</item>
/// </list>
/// <para>
/// A new matcher is randomly initialized as the reference initializes it and does not match meaningfully until the
/// released weights are loaded: call <see cref="LoadParameters"/> with the tensors of One-2-3-45's
/// <c>elevation_estimate/weights/indoor_ds_new.ckpt</c> (its <c>state_dict</c>, under the reference names such as
/// <c>backbone.layer1.0.conv1.weight</c>; a <c>matcher.</c> prefix is stripped and <c>num_batches_tracked</c> is
/// ignored, as the reference's <c>load_state_dict</c> does).
/// </para>
/// <para>
/// As in One-2-3-45's <c>elev_est_api.py</c>, images are converted to grayscale (<c>0.299 R + 0.587 G + 0.114 B</c>),
/// resized to <see cref="LoFTROptions.InputSize"/> (480) and the keypoints scaled back to the input size.
/// </para>
/// <para><b>For Beginners:</b> LoFTR looks at two pictures at once, lets every coarse patch of one "talk" to every
/// patch of the other through a transformer, and keeps the pairs that pick each other. Each pair is then
/// refined to sub-pixel accuracy in a small window.</para>
/// </remarks>
/// <example>
/// <code>
/// var matcher = new LoFTRMatcher&lt;float&gt;();
/// matcher.LoadParameters(stateDict); // indoor_ds_new.ckpt tensors by name
/// var one2345 = new One2345Model&lt;float&gt;(featureMatcher: matcher);
/// </code>
/// </example>
public sealed class LoFTRMatcher<T> : IViewFeatureMatcher<T>
{
    private const string CheckpointPrefix = "matcher.";
    private const string BatchCounterSuffix = ".num_batches_tracked";

    private readonly IEngine _engine;
    private readonly object _buildLock = new object();
    private LoFTRNetwork<T>? _network;

    /// <summary>Creates a matcher; the network is built on first use.</summary>
    /// <param name="options">The configuration; the One-2-3-45 values by default.</param>
    public LoFTRMatcher(LoFTROptions? options = null)
    {
        Options = options ?? new LoFTROptions();
        Options.Validate();
        _engine = AiDotNetEngine.Current;
    }

    /// <summary>The configuration.</summary>
    public LoFTROptions Options { get; }

    /// <summary>The trainable parameter count (the running statistics excluded), as PyTorch counts it.</summary>
    public int ParameterCount => Network.Weights.ParameterCount;

    private LoFTRNetwork<T> Network
    {
        get
        {
            lock (_buildLock)
            {
                if (_network is null)
                {
                    var random = Options.Seed.HasValue
                        ? RandomHelper.CreateSeededRandom(Options.Seed.Value)
                        : RandomHelper.CreateSecureRandom();
                    _network = new LoFTRNetwork<T>(Options, random, _engine);
                }

                return _network;
            }
        }
    }

    /// <summary>The state dict: every parameter and running statistic under its reference name, in module order.</summary>
    public IReadOnlyDictionary<string, Tensor<T>> GetParameters()
    {
        var weights = Network.Weights;
        var result = new Dictionary<string, Tensor<T>>(StringComparer.Ordinal);
        foreach (var name in weights.Names) result[name] = weights[name];
        return result;
    }

    /// <summary>
    /// Loads reference tensors by name. A leading <c>matcher.</c> (the training checkpoint's prefix) is stripped and
    /// <c>num_batches_tracked</c> counters are ignored; names not in <see cref="GetParameters"/> are rejected. Tensors
    /// not given keep their values.
    /// </summary>
    /// <param name="parameters">The tensors; each must have its reference element count.</param>
    /// <exception cref="ArgumentException">A name is unknown or a size is wrong.</exception>
    public void LoadParameters(IReadOnlyDictionary<string, Tensor<T>> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var weights = Network.Weights;
        var resolved = new List<KeyValuePair<string, Tensor<T>>>();
        foreach (var entry in parameters)
        {
            string name = entry.Key.StartsWith(CheckpointPrefix, StringComparison.Ordinal)
                ? entry.Key.Substring(CheckpointPrefix.Length)
                : entry.Key;
            if (name.EndsWith(BatchCounterSuffix, StringComparison.Ordinal)) continue;
            if (!weights.Contains(name)) throw new ArgumentException($"'{entry.Key}' is not a LoFTR parameter.", nameof(parameters));
            if (entry.Value is null) throw new ArgumentException($"'{entry.Key}' has no tensor.", nameof(parameters));
            if (entry.Value.Length != weights[name].Length)
                throw new ArgumentException($"'{entry.Key}' expects {weights[name].Length} values, got {entry.Value.Length}.", nameof(parameters));
            resolved.Add(new KeyValuePair<string, Tensor<T>>(name, entry.Value));
        }

        foreach (var entry in resolved) weights.Set(entry.Key, entry.Value);
    }

    /// <inheritdoc />
    public IReadOnlyList<FeatureCorrespondence> Match(Tensor<T> image0, Tensor<T> image1)
    {
        if (image0 is null) throw new ArgumentNullException(nameof(image0));
        if (image1 is null) throw new ArgumentNullException(nameof(image1));
        if (image0.Shape.Length != 3 || image0.Shape[2] != 3) throw new ArgumentException("Expected [H, W, 3].", nameof(image0));
        if (!image1.Shape.ToArray().SequenceEqual(image0.Shape.ToArray()))
            throw new ArgumentException("Both images must have the same size.", nameof(image1));

        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();
        int height = image0.Shape[0], width = image0.Shape[1];
        int inputHeight = Options.InputSize > 0 ? Options.InputSize : height;
        int inputWidth = Options.InputSize > 0 ? Options.InputSize : width;
        if (inputHeight % 8 != 0 || inputWidth % 8 != 0)
            throw new ArgumentException("Without resizing, the image sides must be multiples of 8.", nameof(image0));

        var pair = _engine.TensorConcatenate(new[]
        {
            Grayscale(image0, inputHeight, inputWidth),
            Grayscale(image1, inputHeight, inputWidth),
        }, 0);
        var matches = Run(pair, inputHeight);
        double sx = (double)width / inputWidth, sy = (double)height / inputHeight;
        return matches.Select(m => new FeatureCorrespondence(m.X0 * sx, m.Y0 * sy, m.X1 * sx, m.Y1 * sy, m.Confidence)).ToList();
    }

    /// <summary><c>LoFTR.forward</c> on a <c>[2, 1, H, W]</c> pair; keypoints in input pixels.</summary>
    internal IReadOnlyList<FeatureCorrespondence> Run(Tensor<T> pair, int inputHeight)
    {
        var e = _engine;
        var network = Network;
        var o = Options;
        var (coarse, fine) = network.Backbone(pair);
        int cc = coarse.Shape[1], hc = coarse.Shape[2], wc = coarse.Shape[3];
        int cf = fine.Shape[1], hf = fine.Shape[2], wf = fine.Shape[3];
        var encoding = NeuSHost<T>.FromDoubles(LoFTRNetwork<T>.PositionEncoding(cc, hc, wc, o.PositionEncodingBugFix), 1, cc, hc, wc);
        coarse = e.TensorAdd(coarse, encoding);

        var c0 = Tokens(coarse, 0);
        var c1 = Tokens(coarse, 1);
        (c0, c1) = network.Transformer(c0, c1, "loftr_coarse", o.CoarseLayerPairs, o.CoarseHeads);
        c0 = e.Reshape(c0, new[] { hc * wc, cc });
        c1 = e.Reshape(c1, new[] { hc * wc, cc });

        var confidence = NeuSHost<T>.ToDoubles(LoFTRNetwork<T>.DualSoftmax(e, c0, c1, o.DualSoftmaxTemperature));
        var coarseMatches = LoFTRNetwork<T>.CoarseMatches(confidence, hc, wc, hc, wc, o.MatchThreshold, o.BorderRemoval);
        if (coarseMatches.Count == 0) return Array.Empty<FeatureCorrespondence>();

        int m = coarseMatches.Count, window = o.FineWindowSize, ww = window * window;
        var iIds = coarseMatches.Select(x => x.I).ToArray();
        var jIds = coarseMatches.Select(x => x.J).ToArray();
        int stride = hf / hc;
        var f0 = Windows(FineRows(fine, 0), iIds, wc, stride, window, hf, wf, cf);
        var f1 = Windows(FineRows(fine, 1), jIds, wc, stride, window, hf, wf, cf);
        if (o.FineConcatCoarseFeature)
        {
            // FinePreprocess: project the coarse features of both sides, repeat them over the window, merge.
            var picked = e.TensorConcatenate(new[]
            {
                e.TensorIndexSelect(c0, NeuSHost<T>.Indices(iIds), 0),
                e.TensorIndexSelect(c1, NeuSHost<T>.Indices(jIds), 0),
            }, 0);
            var projected = network.Linear(picked, "fine_preprocess.down_proj");                    // [2M, Cf]
            var repeat = new int[2 * m * ww];
            for (int r = 0; r < repeat.Length; r++) repeat[r] = r / ww;
            var repeated = e.TensorIndexSelect(projected, NeuSHost<T>.Indices(repeat), 0);           // [2M·WW, Cf]
            var windows = e.Reshape(e.TensorConcatenate(new[] { f0, f1 }, 0), new[] { 2 * m * ww, cf });
            var merged = network.Linear(e.TensorConcatenate(new[] { windows, repeated }, 1), "fine_preprocess.merge_feat");
            merged = e.Reshape(merged, new[] { 2 * m, ww, cf });
            f0 = e.TensorSlice(merged, new[] { 0, 0, 0 }, new[] { m, ww, cf });
            f1 = e.TensorSlice(merged, new[] { m, 0, 0 }, new[] { m, ww, cf });
        }

        (f0, f1) = network.Transformer(f0, f1, "loftr_fine", o.FineLayerPairs, o.FineHeads);
        var offsets = NeuSHost<T>.ToDoubles(LoFTRNetwork<T>.FineExpectation(e, f0, f1, window));
        double coarseScale = (double)inputHeight / hc, fineScale = (double)inputHeight / hf;
        var result = new List<FeatureCorrespondence>(m);
        for (int k = 0; k < m; k++)
        {
            var (i, j, conf) = coarseMatches[k];
            double x0 = i % wc * coarseScale, y0 = i / wc * coarseScale;
            double x1 = j % wc * coarseScale + offsets[2 * k] * (window / 2) * fineScale;
            double y1 = j / wc * coarseScale + offsets[2 * k + 1] * (window / 2) * fineScale;
            result.Add(new FeatureCorrespondence(x0, y0, x1, y1, conf));
        }

        return result;
    }

    // [2, C, h, w] → image b's [1, h·w, C] tokens.
    private Tensor<T> Tokens(Tensor<T> map, int b)
    {
        int c = map.Shape[1], hw = map.Shape[2] * map.Shape[3];
        var single = _engine.Reshape(_engine.TensorSlice(map, new[] { b, 0, 0, 0 }, new[] { 1, c, map.Shape[2], map.Shape[3] }), new[] { c, hw });
        return _engine.Reshape(_engine.TensorTranspose(single), new[] { 1, hw, c });
    }

    // Image b's fine map as [h·w + 1, C] rows; the last row is the zero padding.
    private Tensor<T> FineRows(Tensor<T> fine, int b)
    {
        int c = fine.Shape[1], hw = fine.Shape[2] * fine.Shape[3];
        var rows = _engine.Reshape(Tokens(fine, b), new[] { hw, c });
        return _engine.TensorConcatenate(new[] { rows, new Tensor<T>(new[] { 1, c }) }, 0);
    }

    private Tensor<T> Windows(Tensor<T> rows, int[] ids, int coarseWidth, int stride, int window, int hf, int wf, int c)
    {
        var index = LoFTRNetwork<T>.WindowRows(ids, coarseWidth, stride, window, hf, wf, hf * wf);
        return _engine.Reshape(_engine.TensorIndexSelect(rows, NeuSHost<T>.Indices(index), 0), new[] { ids.Length, window * window, c });
    }

    // cv2 grayscale weights, then cv2.resize's bilinear (half-pixel centers) to the network size: [1, 1, H, W].
    private Tensor<T> Grayscale(Tensor<T> image, int height, int width)
    {
        int h = image.Shape[0], w = image.Shape[1];
        var weights = NeuSHost<T>.FromDoubles(new[] { 0.299, 0.587, 0.114 }, 3, 1);
        var gray = _engine.Reshape(_engine.TensorMatMul(_engine.Reshape(image, new[] { h * w, 3 }), weights), new[] { 1, 1, h, w });
        return h == height && w == width ? gray : _engine.Interpolate(gray, new[] { height, width }, InterpolateMode.Bilinear, false);
    }
}
