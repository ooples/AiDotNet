using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.ComputerVision.Segmentation.Common;

/// <summary>
/// One prediction of kMaX-DeepLab's decoder: class logits <c>[B, N, K + 1]</c> (void last), cluster logits
/// <c>[B, N, h * w]</c> and, in native mode, the L2-normalized pixel features <c>[B, D, h * w]</c> the cluster logits
/// were read from (the instance-discrimination loss uses them).
/// </summary>
internal sealed class KMaXPrediction<T>
{
    internal KMaXPrediction(Tensor<T> classLogits, Tensor<T> maskLogits, Tensor<T>? pixelFeatures, int height, int width)
    {
        ClassLogits = classLogits;
        MaskLogits = maskLogits;
        PixelFeatures = pixelFeatures;
        Height = height;
        Width = width;
    }

    internal Tensor<T> ClassLogits { get; }
    internal Tensor<T> MaskLogits { get; }
    internal Tensor<T>? PixelFeatures { get; }
    internal int Height { get; }
    internal int Width { get; }
}

/// <summary>
/// One prediction of the Mask2Former transformer decoder ODISE uses: mask logits <c>[B, N, h * w]</c>, the queries'
/// class embeddings in the text space <c>[B, N, E]</c> and their cosine similarity to the void (null) embedding
/// <c>[B, N, 1]</c>.
/// </summary>
internal sealed class Mask2FormerPrediction<T>
{
    internal Mask2FormerPrediction(Tensor<T> maskLogits, Tensor<T> classEmbeddings, Tensor<T> voidSimilarity, int height, int width)
    {
        MaskLogits = maskLogits;
        ClassEmbeddings = classEmbeddings;
        VoidSimilarity = voidSimilarity;
        Height = height;
        Width = width;
    }

    internal Tensor<T> MaskLogits { get; }
    internal Tensor<T> ClassEmbeddings { get; }
    internal Tensor<T> VoidSimilarity { get; }
    internal int Height { get; }
    internal int Width { get; }
}

/// <summary>The widths of kMaX-DeepLab's transformer decoder.</summary>
internal readonly struct KMaXDecoderShape
{
    /// <param name="pixelDim">Width of the pixel features and of the cluster mask kernels
    /// (<c>pixel_space_head.output_channels</c>, 128 in the reference).</param>
    /// <param name="queryDim">Width of the cluster centers (in_channel_query, 256).</param>
    /// <param name="heads">Self-attention heads (8).</param>
    /// <param name="feedForwardDim">Hidden width of the centers' FFN (2048).</param>
    /// <param name="classes">Class-head outputs, the classes plus void.</param>
    /// <param name="pixelHeadDim">Width of the pixel-space head's 1x1 convolution after its 5x5 depthwise convolution
    /// (<c>pixel_space_head.head_channels</c>, 256).</param>
    /// <param name="voidPriorProbability">The void probability the fixed class-logit bias gives an uninformative class
    /// head (deeplab2's <c>add_bias_towards_void</c>, <c>void_prior_prob=0.9</c>); must lie in (0, 1).</param>
    internal KMaXDecoderShape(int pixelDim, int queryDim, int heads, int feedForwardDim, int classes, int pixelHeadDim,
        double voidPriorProbability)
    {
        if (!(voidPriorProbability > 0 && voidPriorProbability < 1))
            throw new ArgumentOutOfRangeException(nameof(voidPriorProbability), "The void prior must lie in (0, 1).");
        if (classes < 2) throw new ArgumentOutOfRangeException(nameof(classes), "The class head needs a class and void.");
        VoidPriorProbability = voidPriorProbability;
        PixelDim = pixelDim;
        QueryDim = queryDim;
        Heads = heads;
        FeedForwardDim = feedForwardDim;
        Classes = classes;
        PixelHeadDim = pixelHeadDim;
    }

    internal int PixelDim { get; }
    internal int QueryDim { get; }
    internal int Heads { get; }
    internal int FeedForwardDim { get; }
    internal int Classes { get; }
    internal int PixelHeadDim { get; }
    internal double VoidPriorProbability { get; }

    /// <summary>
    /// The fixed bias added to the void (last) class logit, <c>log((C - 1) * p / (1 - p))</c> with <c>C</c> the
    /// class-head outputs (void included) and <c>p</c> the void prior: deeplab2 model/decoder/max_deeplab.py L91-L96
    /// at 64bc1c1a (<c>add_bias_towards_void</c>, applied at L343-L344) and the PyTorch port's
    /// kmax_transformer_decoder.py L49-L54 (applied at L108). With equal raw logits the softmax gives void exactly p.
    /// </summary>
    internal double VoidLogitBias => Math.Log((Classes - 1) * VoidPriorProbability / (1 - VoidPriorProbability));

    /// <summary>Key width: base_filters * key_expansion = 128 * 1, half the 256-wide bottleneck.</summary>
    internal int KeyDim => QueryDim / 2;

    /// <summary>Value width: base_filters * value_expansion = 128 * 2, the bottleneck width.</summary>
    internal int ValueDim => QueryDim;
}

/// <summary>
/// kMaX-DeepLab's transformer decoder (Yu et al. 2022, Sec. 3.2-3.3; the reference is deeplab2's kmax_deeplab and its
/// PyTorch port's <c>kMaXTransformerLayer</c> and <c>kMaXPredictor</c>), built from standard layers held in the model's
/// layer list so the model's parameter count, parameters and serialization cover them.
/// </summary>
/// <remarks>
/// <para>
/// Each stage runs, on the cluster centers <c>C</c> <c>[B, N, Dq]</c> and one pixel-feature scale <c>X</c>:
/// </para>
/// <list type="number">
/// <item>k-means cross-attention. The pixel features are projected to the bottleneck (<c>GELU(BN(W X))</c>); the
/// query projection <c>GELU(BN(W C))</c> gives the predictor's mask kernels and class logits; the predictor's key
/// projection <c>BN(W x)</c> is L2-normalized; the cluster logits are <c>BN_1(kernel . key)</c>. Every pixel is
/// assigned to its argmax cluster (a cluster-wise argmax, no gradient), the value projection <c>BN(W x)</c> is
/// summed into each cluster, and the output projection <c>BN(W BN(sum))</c> (zero-initialized scale) is added to
/// <c>C</c>.</item>
/// <item>Self-attention among the centers: a fused Q/K/V projection <c>BN(W q)</c> of the same query projection,
/// similarity <c>BN(q . k)</c> per head (no 1/sqrt(d) scale, as the reference), softmax, retrieved values
/// <c>GELU(BN(.))</c>, output projection <c>BN(W .)</c> (zero-initialized scale); <c>C = GELU(C + .)</c>.</item>
/// <item>FFN: <c>C = GELU(C + BN(W2 GELU(BN(W1 C))))</c> (the second scale zero-initialized).</item>
/// </list>
/// <para>
/// After the last stage the class and mask embedding projections (<c>GELU(BN(W C))</c>) feed a final predictor on
/// the finest pixel features. Every stage's prediction is returned so the loss can supervise each one (deep
/// supervision).
/// </para>
/// <para>
/// Each predictor's pixel-space head is deeplab2's <c>PanopticDeepLabSingleHead</c> with a depthwise-separable
/// convolution (the PyTorch port's <c>_pixel_space_head_conv0bnact</c>, <c>_pixel_space_head_conv1bnact</c> and
/// <c>_pixel_space_head_last_convbn</c>): a bias-free 5x5 depthwise convolution, BN and GELU on the predictor's input,
/// a bias-free 1x1 convolution to <see cref="KMaXDecoderShape.PixelHeadDim"/> (256), BN and GELU, then the biased 1x1
/// key projection to <see cref="KMaXDecoderShape.PixelDim"/> (128) and BN (<c>pixel_space_feature_batch_norm</c>),
/// L2-normalized.
/// </para>
/// </remarks>
internal static class KMaXTransformerDecoder
{
    /// <summary>Layers of one predictor: the pixel-space head (5x5 depthwise + BN, 1x1 + BN, key projection + BN),
    /// mask kernel + BN, cluster-logit BN, class head.</summary>
    internal const int PredictorLayerCount = 10;

    /// <summary>Kernel of the pixel-space head's depthwise convolution (5, deeplab2's PanopticDeepLabSingleHead).</summary>
    internal const int PixelHeadKernel = 5;

    /// <summary>Layers of one decoder stage.</summary>
    internal const int StageLayerCount = 19 + PredictorLayerCount;

    /// <summary>Layers after the last stage: class and mask embedding projections (with BN) and the final predictor.</summary>
    internal const int FinalLayerCount = 4 + PredictorLayerCount;

    /// <summary>The cluster-logit BatchNorm's initial scale (<c>_pixel_space_mask_batch_norm</c>, 0.1).</summary>
    internal const double MaskLogitNormInitialScale = 0.1;

    /// <summary>All layers of the decoder: the initial centers, the stages and the final heads.</summary>
    internal static int LayerCount(int stages) => 1 + stages * StageLayerCount + FinalLayerCount;

    /// <summary>Creates the decoder's layers, in the order <see cref="Run"/> reads them.</summary>
    internal static IEnumerable<ILayer<T>> CreateLayers<T>(KMaXDecoderShape shape, int stages)
    {
        // The learnable initial cluster centers, read on the identity (an embedding table).
        yield return new DenseLayer<T>(shape.QueryDim, biasMode: BiasMode.Never);
        for (int s = 0; s < stages; s++)
        {
            yield return Linear<T>(shape.QueryDim); yield return Norm<T>();           // pixel bottleneck projection
            yield return Linear<T>(shape.QueryDim); yield return Norm<T>();           // query projection
            yield return Linear<T>(shape.ValueDim); yield return Norm<T>();           // value projection
            foreach (var layer in CreatePredictorLayers<T>(shape, shape.QueryDim)) yield return layer; // pixel head, mask kernel, class head
            yield return Norm<T>();                                                   // retrieved k-means value
            yield return Linear<T>(shape.QueryDim); yield return ZeroNorm<T>();       // k-means output projection
            yield return Linear<T>(2 * shape.KeyDim + shape.ValueDim); yield return Norm<T>(); // fused Q/K/V
            yield return Norm<T>();                                                   // similarity, per head
            yield return Norm<T>();                                                   // retrieved attention value
            yield return Linear<T>(shape.QueryDim); yield return ZeroNorm<T>();       // attention output projection
            yield return Linear<T>(shape.FeedForwardDim); yield return Norm<T>();     // FFN up
            yield return Linear<T>(shape.QueryDim); yield return ZeroNorm<T>();       // FFN down
        }

        yield return Linear<T>(shape.QueryDim); yield return Norm<T>();               // class embedding projection
        yield return Linear<T>(shape.QueryDim); yield return Norm<T>();               // mask embedding projection
        foreach (var layer in CreatePredictorLayers<T>(shape, shape.PixelDim)) yield return layer;
    }

    /// <param name="shape">The decoder widths.</param>
    /// <param name="inputChannels">Width of the pixel features the predictor reads: a stage's bottleneck
    /// (<see cref="KMaXDecoderShape.QueryDim"/>) or the finest pixel-decoder features.</param>
    internal static IEnumerable<ILayer<T>> CreatePredictorLayers<T>(KMaXDecoderShape shape, int inputChannels)
    {
        yield return ConvolutionalLayer<T>.WithInputDepth(inputChannels, inputChannels, PixelHeadKernel, 1, PixelHeadKernel / 2,
            groups: inputChannels, biasMode: BiasMode.Never);                       // 5x5 depthwise convolution
        yield return Norm<T>();
        yield return ConvolutionalLayer<T>.WithInputDepth(inputChannels, shape.PixelHeadDim, 1, 1, 0,
            biasMode: BiasMode.Never);                                              // 1x1 convolution to the head width
        yield return Norm<T>();
        yield return new DenseLayer<T>(shape.PixelDim); yield return Norm<T>();      // pixel key projection (bias, as the reference)
        yield return Linear<T>(shape.PixelDim); yield return Norm<T>();              // mask kernel (query) projection
        var maskNorm = new BatchNormalizationLayer<T>(1);
        maskNorm.GetGamma().Fill(MathHelper.GetNumericOperations<T>().FromDouble(MaskLogitNormInitialScale));
        yield return maskNorm;
        yield return new DenseLayer<T>(shape.Classes);                               // class head
    }

    private static DenseLayer<T> Linear<T>(int width) => new DenseLayer<T>(width, biasMode: BiasMode.Never);

    private static BatchNormalizationLayer<T> Norm<T>() => new BatchNormalizationLayer<T>();

    private static BatchNormalizationLayer<T> ZeroNorm<T>()
    {
        var norm = new BatchNormalizationLayer<T>();
        norm.ZeroInitGamma();
        return norm;
    }

    /// <summary>
    /// Runs the decoder.
    /// </summary>
    /// <param name="layers">The model's layers.</param>
    /// <param name="start">Index of the decoder's first layer (the initial centers).</param>
    /// <param name="shape">The decoder widths.</param>
    /// <param name="scales">Pixel features <c>[B, Dp, h * w]</c> with their sizes, in reading order (coarse to fine);
    /// stage <c>t</c> of <paramref name="stages"/> reads scale <c>floor(t * S / stages)</c>.</param>
    /// <param name="stages">Number of decoder stages (6 in the paper: two per scale).</param>
    /// <param name="finest">The finest pixel features <c>[B, Dp, h * w]</c> the final predictor reads.</param>
    /// <param name="finestHeight">Height of <paramref name="finest"/>.</param>
    /// <param name="finestWidth">Width of <paramref name="finest"/>.</param>
    /// <param name="numClusters">Number of cluster centers <c>N</c>.</param>
    /// <returns>Every stage's prediction, then the final prediction.</returns>
    internal static IReadOnlyList<KMaXPrediction<T>> Run<T>(
        IReadOnlyList<ILayer<T>> layers, int start, KMaXDecoderShape shape,
        IReadOnlyList<(Tensor<T> Features, int Height, int Width)> scales, int stages,
        Tensor<T> finest, int finestHeight, int finestWidth, int numClusters)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        if (scales is null) throw new ArgumentNullException(nameof(scales));
        if (finest is null) throw new ArgumentNullException(nameof(finest));
        if (scales.Count == 0) throw new ArgumentException("At least one pixel-feature scale is required.", nameof(scales));
        if (stages < 1) throw new ArgumentOutOfRangeException(nameof(stages), "At least one decoder stage is required.");
        if (start < 0 || start + LayerCount(stages) > layers.Count)
            throw new ArgumentException("The layer list does not hold the decoder's layers at the given start.", nameof(layers));

        var engine = AiDotNetEngine.Current;
        int b = finest.Shape[0];
        var identity = engine.TensorOneHot<T>(engine.TensorArange<int>(0, numClusters, 1), numClusters);
        var centers = engine.TensorBroadcastTo(
            engine.Reshape(layers[start].Forward(identity), new[] { 1, numClusters, shape.QueryDim }),
            new[] { b, numClusters, shape.QueryDim });

        var predictions = new List<KMaXPrediction<T>>(stages + 1);
        for (int t = 0; t < stages; t++)
        {
            var scale = scales[(int)((long)t * scales.Count / stages)];
            (centers, var prediction) = Stage(layers, start + 1 + t * StageLayerCount, shape, centers,
                scale.Features, scale.Height, scale.Width);
            predictions.Add(prediction);
        }

        int final = start + 1 + stages * StageLayerCount;
        var classEmbeddings = engine.TensorGELU(ProjectNorm(layers[final], layers[final + 1], centers));
        var maskEmbeddings = engine.TensorGELU(ProjectNorm(layers[final + 2], layers[final + 3], centers));
        predictions.Add(Predict(layers, final + 4, shape, classEmbeddings, maskEmbeddings,
            engine.TensorPermute(finest, new[] { 0, 2, 1 }), finestHeight, finestWidth));
        return predictions;
    }

    /// <summary>One kMaX transformer layer: k-means cross-attention, self-attention and FFN on the centers.</summary>
    internal static (Tensor<T> Centers, KMaXPrediction<T> Prediction) Stage<T>(
        IReadOnlyList<ILayer<T>> layers, int start, KMaXDecoderShape shape, Tensor<T> centers,
        Tensor<T> pixelFeatures, int height, int width)
    {
        var engine = AiDotNetEngine.Current;
        var pixels = engine.TensorGELU(engine.TensorPermute(pixelFeatures, new[] { 0, 2, 1 }));            // [B, hw, Dp]
        var pixelSpace = engine.TensorGELU(ProjectNorm(layers[start], layers[start + 1], pixels));          // [B, hw, Dq]
        var querySpace = engine.TensorGELU(ProjectNorm(layers[start + 2], layers[start + 3], centers));     // [B, N, Dq]
        var pixelValue = ProjectNorm(layers[start + 4], layers[start + 5], pixelSpace);                     // [B, hw, Dv]
        var prediction = Predict(layers, start + 6, shape, querySpace, querySpace, pixelSpace, height, width);

        // k-means cross-attention: cluster-wise argmax of the cluster logits, then the assigned values summed into
        // each cluster center.
        int p = start + 6 + PredictorLayerCount;
        var assignment = HardAssignment(prediction.MaskLogits);                                             // [B, N, hw]
        var update = Rows(layers[p], engine.TensorMatMul(assignment, pixelValue));                          // [B, N, Dv]
        update = ProjectNorm(layers[p + 1], layers[p + 2], update);
        centers = engine.TensorAdd(centers, update);

        // Self-attention among the cluster centers, on the query projection taken before the k-means update.
        var qkv = ProjectNorm(layers[p + 3], layers[p + 4], querySpace);                                    // [B, N, 2Dk + Dv]
        var query = engine.TensorNarrow(qkv, 2, 0, shape.KeyDim);
        var key = engine.TensorNarrow(qkv, 2, shape.KeyDim, shape.KeyDim);
        var value = engine.TensorNarrow(qkv, 2, 2 * shape.KeyDim, shape.ValueDim);
        var attended = SelfAttention(query, key, value, shape.Heads, layers[p + 5], layers[p + 6]);
        centers = engine.TensorGELU(engine.TensorAdd(centers, ProjectNorm(layers[p + 7], layers[p + 8], attended)));

        // FFN.
        var hidden = engine.TensorGELU(ProjectNorm(layers[p + 9], layers[p + 10], centers));
        centers = engine.TensorGELU(engine.TensorAdd(centers, ProjectNorm(layers[p + 11], layers[p + 12], hidden)));
        return (centers, prediction);
    }

    /// <summary>
    /// The one-hot cluster-wise argmax <c>[B, N, hw]</c> of <paramref name="maskLogits"/> (each pixel goes to the
    /// cluster with the largest logit), computed without a gradient as in the reference.
    /// </summary>
    internal static Tensor<T> HardAssignment<T>(Tensor<T> maskLogits)
    {
        var engine = AiDotNetEngine.Current;
        int b = maskLogits.Shape[0], n = maskLogits.Shape[1], hw = maskLogits.Shape[2];
        var detached = engine.StopGradient(maskLogits);
        // The winners are flattened before the one-hot and the batch axis restored after it, so the result is
        // [B, hw, N] whatever rank TensorOneHot gives a [B, hw] index tensor.
        var winners = engine.TensorArgMax(detached, 1).Reshape(new[] { b * hw });
        var oneHot = engine.Reshape(engine.TensorOneHot<T>(winners, n), new[] { b, hw, n });
        return engine.TensorPermute(oneHot, new[] { 0, 2, 1 });
    }

    /// <summary>
    /// kMaXPredictor: class logits from the class embeddings, cluster logits on the normalized pixel keys of the
    /// pixel-space head (5x5 depthwise conv, BN, GELU; 1x1 conv, BN, GELU; biased 1x1 key projection, BN; L2 norm).
    /// </summary>
    internal static KMaXPrediction<T> Predict<T>(
        IReadOnlyList<ILayer<T>> layers, int start, KMaXDecoderShape shape, Tensor<T> classEmbeddings,
        Tensor<T> maskEmbeddings, Tensor<T> pixels, int height, int width)
    {
        var engine = AiDotNetEngine.Current;
        int b = pixels.Shape[0], hw = pixels.Shape[1], c = pixels.Shape[2], n = maskEmbeddings.Shape[1];
        if (hw != height * width) throw new ArgumentException("The pixel features do not match the given size.", nameof(pixels));
        var spatial = engine.Reshape(engine.TensorPermute(pixels, new[] { 0, 2, 1 }), new[] { b, c, height, width });
        spatial = engine.TensorGELU(layers[start + 1].Forward(layers[start].Forward(spatial)));               // 5x5 depthwise
        spatial = engine.TensorGELU(layers[start + 3].Forward(layers[start + 2].Forward(spatial)));           // 1x1 to the head width
        var head = engine.TensorPermute(engine.Reshape(spatial, new[] { b, spatial.Shape[1], hw }), new[] { 0, 2, 1 }); // [B, hw, Dh]
        var key = MaskClassificationOps.NormalizeAxis(ProjectNorm(layers[start + 4], layers[start + 5], head), 2); // [B, hw, Dp]
        var keyT = engine.TensorPermute(key, new[] { 0, 2, 1 });                                                // [B, Dp, hw]
        var kernel = ProjectNorm(layers[start + 6], layers[start + 7], maskEmbeddings);                          // [B, N, Dp]
        var logits = engine.TensorMatMul(kernel, keyT);                                                          // [B, N, hw]
        logits = engine.Reshape(layers[start + 8].Forward(engine.Reshape(logits, new[] { b * n * hw, 1 })), new[] { b, n, hw });
        var classLogits = AddVoidBias(Rows(layers[start + 9], classEmbeddings), shape);
        return new KMaXPrediction<T>(classLogits, logits, keyT, height, width);
    }

    /// <summary>
    /// Adds <see cref="KMaXDecoderShape.VoidLogitBias"/> to the void (last) logit of <c>[B, N, C]</c> class logits, a
    /// constant (not a parameter) as in deeplab2's <c>add_bias_towards_void</c>.
    /// </summary>
    internal static Tensor<T> AddVoidBias<T>(Tensor<T> classLogits, KMaXDecoderShape shape)
    {
        var engine = AiDotNetEngine.Current;
        int c = classLogits.Shape[2];
        var bias = new double[c];
        bias[c - 1] = shape.VoidLogitBias;
        var row = engine.Reshape(new Tensor<double>(new[] { c }, new Vector<double>(bias)).Cast<T>(), new[] { 1, 1, c });
        return engine.TensorAdd(classLogits, engine.TensorBroadcastTo(row, classLogits.Shape.ToArray()));
    }

    /// <summary>
    /// kMaX's AttentionOperation: per-head similarity <c>q . k</c> through a per-head BatchNorm, softmax over the
    /// keys, retrieved values through BatchNorm and GELU.
    /// </summary>
    private static Tensor<T> SelfAttention<T>(Tensor<T> q, Tensor<T> k, Tensor<T> v, int heads, ILayer<T> similarityNorm, ILayer<T> valueNorm)
    {
        var engine = AiDotNetEngine.Current;
        int b = q.Shape[0], n = q.Shape[1], dk = q.Shape[2] / heads, dv = v.Shape[2] / heads;
        var qh = SplitHeads(q, heads, dk);
        var kh = SplitHeads(k, heads, dk);
        var vh = SplitHeads(v, heads, dv);
        var similarity = engine.Reshape(engine.TensorMatMul(qh, engine.TensorPermute(kh, new[] { 0, 2, 1 })), new[] { b, heads, n, n });
        var weights = engine.Reshape(engine.TensorSoftmax(similarityNorm.Forward(similarity), 3), new[] { b * heads, n, n });
        var retrieved = MergeHeads(engine.TensorMatMul(weights, vh), b, heads, n, dv);
        return engine.TensorGELU(Rows(valueNorm, retrieved));
    }

    /// <summary><c>[B, L, H * d]</c> to <c>[B * H, L, d]</c>.</summary>
    internal static Tensor<T> SplitHeads<T>(Tensor<T> x, int heads, int headDim)
    {
        var engine = AiDotNetEngine.Current;
        int b = x.Shape[0], l = x.Shape[1];
        return engine.Reshape(
            engine.TensorPermute(engine.Reshape(x, new[] { b, l, heads, headDim }), new[] { 0, 2, 1, 3 }),
            new[] { b * heads, l, headDim });
    }

    /// <summary><c>[B * H, L, d]</c> to <c>[B, L, H * d]</c>.</summary>
    internal static Tensor<T> MergeHeads<T>(Tensor<T> x, int batch, int heads, int length, int headDim)
    {
        var engine = AiDotNetEngine.Current;
        return engine.Reshape(
            engine.TensorPermute(engine.Reshape(x, new[] { batch, heads, length, headDim }), new[] { 0, 2, 1, 3 }),
            new[] { batch, length, heads * headDim });
    }

    /// <summary>Applies a per-row layer to <c>[B, L, C]</c> as <c>[B * L, C]</c> (a 1x1 convolution or 1-D BN).</summary>
    internal static Tensor<T> Rows<T>(ILayer<T> layer, Tensor<T> x)
    {
        var engine = AiDotNetEngine.Current;
        int b = x.Shape[0], l = x.Shape[1];
        var y = layer.Forward(engine.Reshape(x, new[] { b * l, x.Shape[2] }));
        return engine.Reshape(y, new[] { b, l, y.Shape[1] });
    }

    private static Tensor<T> ProjectNorm<T>(ILayer<T> projection, ILayer<T> norm, Tensor<T> x) => Rows(norm, Rows(projection, x));
}

/// <summary>
/// Mask2Former's transformer decoder (Cheng et al. 2022, Sec. 3.2; the reference is
/// <c>MultiScaleMaskedTransformerDecoder</c>), the mask decoder ODISE uses (Xu et al. 2023), built from standard
/// layers held in the model's layer list.
/// </summary>
/// <remarks>
/// <para>
/// Learnable query features and query positional embeddings (<c>N x D</c>) are refined by <c>L</c> layers, each
/// masked cross-attention, then self-attention, then an FFN, each followed by a residual connection and LayerNorm
/// (post-norm, dropout 0, as Mask2Former's defaults):
/// </para>
/// <list type="number">
/// <item><c>X = LN(X + W_o MHA(W_q (X + P_q), W_k (F + P_f), W_v F; M))</c>, where the attention mask <c>M</c> blocks
/// the pixels the previous prediction gives <c>sigmoid &lt; 0.5</c> (no pixel is blocked for a query whose mask is
/// empty).</item>
/// <item><c>X = LN(X + W_o MHA(W_q (X + P_q), W_k (X + P_q), W_v X))</c>.</item>
/// <item><c>X = LN(X + W_2 ReLU(W_1 X))</c>.</item>
/// </list>
/// <para>
/// The prediction heads (on <c>LN(X)</c>) run before the first layer and after every layer: the mask is
/// <c>MLP(LN(X)) . F</c> (a 3-layer MLP) and the class embedding a linear map to the text space, as in ODISE, whose
/// void logit compares the embedding with a learnable null embedding. <c>P_f</c> is the fixed 2-D sine positional
/// encoding of the pixels (normalized, temperature 10000).
/// </para>
/// </remarks>
internal static class Mask2FormerTransformerDecoder
{
    /// <summary>Layers before the decoder layers: query features, query positions, decoder norm, class embedding,
    /// the 3-layer mask MLP and the null embedding.</summary>
    internal const int HeadLayerCount = 8;

    /// <summary>Layers of one decoder layer: cross-attention Q/K/V/O + LN, self-attention Q/K/V/O + LN, FFN + LN.</summary>
    internal const int BlockLayerCount = 13;

    /// <summary>The blocked-pixel score added before the softmax.</summary>
    private const double BlockedScore = -1e9;

    /// <summary>All decoder layers for <paramref name="numLayers"/> decoder layers.</summary>
    internal static int LayerCount(int numLayers) => HeadLayerCount + numLayers * BlockLayerCount;

    /// <summary>Creates the decoder's layers, in the order <see cref="Run"/> reads them.</summary>
    internal static IEnumerable<ILayer<T>> CreateLayers<T>(int hiddenDim, int feedForwardDim, int embeddingDim, int numLayers)
    {
        IActivationFunction<T> relu = new ReLUActivation<T>();
        yield return new DenseLayer<T>(hiddenDim, biasMode: BiasMode.Never);          // query_feat (an embedding table)
        yield return new DenseLayer<T>(hiddenDim, biasMode: BiasMode.Never);          // query_embed
        yield return new LayerNormalizationLayer<T>(hiddenDim);                       // decoder_norm
        yield return new DenseLayer<T>(embeddingDim);                                 // class embedding (text space)
        yield return new DenseLayer<T>(hiddenDim, relu);                              // mask_embed MLP
        yield return new DenseLayer<T>(hiddenDim, relu);
        yield return new DenseLayer<T>(hiddenDim);
        yield return new DenseLayer<T>(embeddingDim, biasMode: BiasMode.Never);       // null (void) embedding
        for (int l = 0; l < numLayers; l++)
        {
            for (int p = 0; p < 4; p++) yield return new DenseLayer<T>(hiddenDim);     // cross-attention Q, K, V, O
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            for (int p = 0; p < 4; p++) yield return new DenseLayer<T>(hiddenDim);     // self-attention Q, K, V, O
            yield return new LayerNormalizationLayer<T>(hiddenDim);
            yield return new DenseLayer<T>(feedForwardDim, relu);                      // FFN
            yield return new DenseLayer<T>(hiddenDim);
            yield return new LayerNormalizationLayer<T>(hiddenDim);
        }
    }

    /// <summary>
    /// Runs the decoder on the mask features <paramref name="pixel"/> <c>[B, D, h * w]</c>.
    /// </summary>
    /// <returns>The prediction of the initial queries, then one per decoder layer (the last is the output).</returns>
    internal static IReadOnlyList<Mask2FormerPrediction<T>> Run<T>(
        IReadOnlyList<ILayer<T>> layers, int start, int numQueries, int heads, int numLayers,
        Tensor<T> pixel, int height, int width)
    {
        if (layers is null) throw new ArgumentNullException(nameof(layers));
        if (pixel is null) throw new ArgumentNullException(nameof(pixel));
        if (numLayers < 1) throw new ArgumentOutOfRangeException(nameof(numLayers), "At least one decoder layer is required.");
        if (start < 0 || start + LayerCount(numLayers) > layers.Count)
            throw new ArgumentException("The layer list does not hold the decoder's layers at the given start.", nameof(layers));
        var engine = AiDotNetEngine.Current;
        int b = pixel.Shape[0], d = pixel.Shape[1], hw = pixel.Shape[2];
        if (hw != height * width) throw new ArgumentException("The mask features do not match the given size.", nameof(pixel));
        if (heads < 1 || d % heads != 0)
            throw new ArgumentException($"The number of heads ({heads}) must divide the hidden width ({d}).", nameof(heads));

        var identity = engine.TensorOneHot<T>(engine.TensorArange<int>(0, numQueries, 1), numQueries);
        var queryShape = new[] { b, numQueries, d };
        var query = engine.TensorBroadcastTo(engine.Reshape(layers[start].Forward(identity), new[] { 1, numQueries, d }), queryShape);
        var queryPosition = engine.TensorBroadcastTo(
            engine.Reshape(layers[start + 1].Forward(identity), new[] { 1, numQueries, d }), queryShape);
        var memory = engine.TensorPermute(pixel, new[] { 0, 2, 1 });                                     // [B, HW, D]
        var memoryKey = engine.TensorAdd(memory, engine.TensorBroadcastTo(
            engine.Reshape(SinePositionEmbedding<T>(d, height, width), new[] { 1, hw, d }), new[] { b, hw, d }));

        var predictions = new List<Mask2FormerPrediction<T>>(numLayers + 1);
        var prediction = Heads(layers, start, query, pixel, height, width);
        predictions.Add(prediction);
        for (int l = 0; l < numLayers; l++)
        {
            int s = start + HeadLayerCount + l * BlockLayerCount;
            var blocked = AttentionMask(prediction.MaskLogits);

            var positioned = engine.TensorAdd(query, queryPosition);
            var cross = Attend(Rows(layers[s], positioned), Rows(layers[s + 1], memoryKey), Rows(layers[s + 2], memory), heads, blocked);
            query = Rows(layers[s + 4], engine.TensorAdd(query, Rows(layers[s + 3], cross)));

            positioned = engine.TensorAdd(query, queryPosition);
            var self = Attend(Rows(layers[s + 5], positioned), Rows(layers[s + 6], positioned), Rows(layers[s + 7], query), heads, null);
            query = Rows(layers[s + 9], engine.TensorAdd(query, Rows(layers[s + 8], self)));

            var ffn = Rows(layers[s + 11], Rows(layers[s + 10], query));
            query = Rows(layers[s + 12], engine.TensorAdd(query, ffn));

            prediction = Heads(layers, start, query, pixel, height, width);
            predictions.Add(prediction);
        }

        return predictions;
    }

    /// <summary>Mask2Former's prediction heads on <c>LN(X)</c>, with ODISE's text-space class embedding.</summary>
    private static Mask2FormerPrediction<T> Heads<T>(
        IReadOnlyList<ILayer<T>> layers, int start, Tensor<T> query, Tensor<T> pixel, int height, int width)
    {
        var engine = AiDotNetEngine.Current;
        var normalized = Rows(layers[start + 2], query);
        var classEmbeddings = Rows(layers[start + 3], normalized);                                      // [B, N, E]
        var maskEmbeddings = Rows(layers[start + 6], Rows(layers[start + 5], Rows(layers[start + 4], normalized)));
        var maskLogits = engine.TensorMatMul(maskEmbeddings, pixel);                                    // [B, N, HW]
        var nullEmbedding = layers[start + 7].Forward(engine.TensorOneHot<T>(engine.TensorArange<int>(0, 1, 1), 1)); // [1, E]
        var voidSimilarity = engine.TensorMatMul(
            MaskClassificationOps.NormalizeAxis(classEmbeddings, 2),
            engine.TensorTranspose(MaskClassificationOps.NormalizeAxis(nullEmbedding, 1)));              // [B, N, 1]
        return new Mask2FormerPrediction<T>(maskLogits, classEmbeddings, voidSimilarity, height, width);
    }

    /// <summary>
    /// Mask2Former's attention mask <c>[B, N, HW]</c>: 1 where a query may not attend (its previous mask has
    /// <c>sigmoid(logit) &lt; 0.5</c>), and all 0 for a query whose previous mask is empty.
    /// </summary>
    internal static Tensor<T> AttentionMask<T>(Tensor<T> maskLogits)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        var detached = engine.StopGradient(maskLogits);
        var foreground = engine.TensorGreaterEqual(detached, Tensor<T>.CreateDefault(detached.Shape.ToArray(), ops.Zero));
        var anyForeground = engine.ReduceMax(foreground, new[] { 2 }, true);
        return engine.TensorMultiply(
            engine.ScalarMinusTensor(ops.One, foreground),
            engine.TensorBroadcastTo(anyForeground, foreground.Shape.ToArray()));
    }

    /// <summary>
    /// Multi-head scaled dot-product attention of <c>[B, N, D]</c> queries over <c>[B, M, D]</c> keys and values;
    /// <paramref name="blocked"/> <c>[B, N, M]</c> (1 = blocked) applies to every head.
    /// </summary>
    internal static Tensor<T> Attend<T>(Tensor<T> q, Tensor<T> k, Tensor<T> v, int heads, Tensor<T>? blocked)
    {
        var engine = AiDotNetEngine.Current;
        var ops = MathHelper.GetNumericOperations<T>();
        int b = q.Shape[0], n = q.Shape[1], m = k.Shape[1], headDim = q.Shape[2] / heads;
        var qh = KMaXTransformerDecoder.SplitHeads(q, heads, headDim);
        var kh = KMaXTransformerDecoder.SplitHeads(k, heads, headDim);
        var vh = KMaXTransformerDecoder.SplitHeads(v, heads, headDim);
        var scores = engine.TensorMultiplyScalar(
            engine.TensorMatMul(qh, engine.TensorPermute(kh, new[] { 0, 2, 1 })), ops.FromDouble(1.0 / Math.Sqrt(headDim)));
        if (blocked is not null)
        {
            var perHead = engine.Reshape(
                engine.TensorBroadcastTo(engine.Reshape(blocked, new[] { b, 1, n, m }), new[] { b, heads, n, m }),
                new[] { b * heads, n, m });
            scores = engine.TensorWhere(perHead, Tensor<T>.CreateDefault(new[] { b * heads, n, m }, ops.FromDouble(BlockedScore)), scores);
        }

        var weights = engine.TensorSoftmax(scores, 2);
        return KMaXTransformerDecoder.MergeHeads(engine.TensorMatMul(weights, vh), b, heads, n, headDim);
    }

    /// <summary>
    /// DETR/Mask2Former's 2-D sine positional encoding <c>[h * w, d]</c> (<c>PositionEmbeddingSine</c>, normalize=True,
    /// scale 2 pi, temperature 10000): the first <c>d / 2</c> channels encode the row, the rest the column, alternating
    /// sin and cos. An odd <paramref name="d"/> keeps the first <paramref name="d"/> of <c>2 ceil(d / 2)</c> channels.
    /// </summary>
    internal static Tensor<T> SinePositionEmbedding<T>(int d, int height, int width)
    {
        int features = (d + 1) / 2;
        const double eps = 1e-6, scale = 2 * Math.PI, temperature = 10000.0;
        var table = new double[height * width * d];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                double yEmbed = (y + 1) / (height + eps) * scale;
                double xEmbed = (x + 1) / (width + eps) * scale;
                int row = (y * width + x) * d;
                for (int c = 0; c < d; c++)
                {
                    int axisChannel = c < features ? c : c - features;
                    double embed = c < features ? yEmbed : xEmbed;
                    double phase = embed / Math.Pow(temperature, 2.0 * (axisChannel / 2) / features);
                    table[row + c] = axisChannel % 2 == 0 ? Math.Sin(phase) : Math.Cos(phase);
                }
            }
        }

        return new Tensor<double>(new[] { height * width, d }, new Vector<double>(table)).Cast<T>();
    }

    private static Tensor<T> Rows<T>(ILayer<T> layer, Tensor<T> x) => KMaXTransformerDecoder.Rows(layer, x);
}
