using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Helpers;

public static partial class LayerHelper<T>
{
    /// <summary>
    /// Creates the native EfficientConformer CTC encoder from Burchi and Vielzeuf (2021):
    /// three Conformer stages with 120/168/240-width scaling, progressive temporal
    /// downsampling, grouped attention in the first stage, and a linear CTC head.
    /// </summary>
    /// <remarks>
    /// The authors' small CTC recipe uses fifteen blocks split evenly across the three stages,
    /// group size three before the first expansion, a 15-sample convolution kernel, and total
    /// temporal reduction of eight. In the library's [batch,time,features] representation, each
    /// stride-two transition is frame splicing followed immediately by a learned projection.
    /// </remarks>
    public static IEnumerable<ILayer<T>> CreateDefaultEfficientConformerLayers(
        int encoderDim = 120,
        int numLayers = 15,
        int numAttentionHeads = 4,
        int feedForwardExpansionFactor = 4,
        int convKernelSize = 15,
        int downsamplingFactor = 8,
        int attentionGroupSize = 3,
        int numMels = 80,
        int vocabSize = 256,
        double dropoutRate = 0.1,
        int maxSequenceLength = 750,
        bool useLayerNormalization = false)
    {
        if (encoderDim <= 0) throw new ArgumentOutOfRangeException(nameof(encoderDim));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (numAttentionHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numAttentionHeads));
        if (feedForwardExpansionFactor <= 0)
            throw new ArgumentOutOfRangeException(nameof(feedForwardExpansionFactor));
        if (convKernelSize <= 0) throw new ArgumentOutOfRangeException(nameof(convKernelSize));
        if (attentionGroupSize <= 0) throw new ArgumentOutOfRangeException(nameof(attentionGroupSize));
        if (numMels <= 0) throw new ArgumentOutOfRangeException(nameof(numMels));
        if (vocabSize <= 1) throw new ArgumentOutOfRangeException(nameof(vocabSize));
        if (maxSequenceLength <= 0) throw new ArgumentOutOfRangeException(nameof(maxSequenceLength));
        if (downsamplingFactor is not (1 or 2 or 4 or 8))
            throw new ArgumentOutOfRangeException(
                nameof(downsamplingFactor), "Downsampling factor must be one of 1, 2, 4, or 8.");
        if (dropoutRate < 0 || dropoutRate >= 1)
            throw new ArgumentOutOfRangeException(nameof(dropoutRate), "dropoutRate must be in [0, 1).");

        int RoundToHeadMultiple(double width)
        {
            int multiple = (int)Math.Round(width / numAttentionHeads) * numAttentionHeads;
            return Math.Max(numAttentionHeads, multiple);
        }

        int[] stageDimensions =
        [
            RoundToHeadMultiple(encoderDim),
            RoundToHeadMultiple(encoderDim * 7.0 / 5.0),
            RoundToHeadMultiple(encoderDim * 2.0),
        ];
        int baseDepth = numLayers / 3;
        int remainder = numLayers % 3;
        int[] stageDepths =
        [
            baseDepth + (remainder > 0 ? 1 : 0),
            baseDepth + (remainder > 1 ? 1 : 0),
            baseDepth,
        ];

        var identity = (IActivationFunction<T>)new IdentityActivation<T>();
        int appliedDownsampling = 1;

        // The official CTC-small front-end begins with one stride-two convolutional subsampler.
        if (downsamplingFactor >= 2)
        {
            yield return new TemporalFrameSplicingLayer<T>(2);
            appliedDownsampling = 2;
        }

        yield return new DenseLayer<T>(stageDimensions[0], identity);
        yield return useLayerNormalization
            ? new LayerNormalizationLayer<T>()
            : new BatchNormalizationLayer<T>(stageDimensions[0]);
        if (dropoutRate > 0) yield return new DropoutLayer<T>(dropoutRate);

        for (int stage = 0; stage < stageDimensions.Length; stage++)
        {
            if (stage > 0)
            {
                if (appliedDownsampling < downsamplingFactor)
                {
                    yield return new TemporalFrameSplicingLayer<T>(2);
                    appliedDownsampling *= 2;
                }

                yield return new DenseLayer<T>(stageDimensions[stage], identity);
                yield return useLayerNormalization
                    ? new LayerNormalizationLayer<T>()
                    : new BatchNormalizationLayer<T>(stageDimensions[stage]);
            }

            int heads = ChooseDivisibleHeadConfig(
                stageDimensions[stage], numAttentionHeads).heads;
            int groupSize = stage == 0 ? attentionGroupSize : 1;
            for (int block = 0; block < stageDepths[stage]; block++)
            {
                yield return new ConformerBlockLayer<T>(
                    stageDimensions[stage],
                    heads,
                    feedForwardExpansionFactor,
                    convKernelSize,
                    maxSequenceLength: maxSequenceLength,
                    attentionGroupSize: groupSize);
                if (dropoutRate > 0) yield return new DropoutLayer<T>(dropoutRate);
            }
        }

        yield return new LayerNormalizationLayer<T>();
        yield return new DenseLayer<T>(vocabSize, identity);
    }
}
