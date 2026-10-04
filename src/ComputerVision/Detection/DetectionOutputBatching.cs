namespace AiDotNet.ComputerVision.Detection;

/// <summary>
/// Splits a batched detector forward into one output list per image, for post-processing that reads a
/// single image at a time.
/// </summary>
/// <remarks>
/// Object and text detectors both run one forward pass over the whole batch and then decode each image
/// separately; their decoders index batch position 0. Before the text-detector base used this, its
/// models decoded only <c>[0, ...]</c> of a batched forward and silently dropped every other image.
/// </remarks>
internal static class DetectionOutputBatching<T>
{
    /// <summary>Returns the outputs of batch item <paramref name="batchIndex"/>, each with a leading batch of 1.</summary>
    internal static List<Tensor<T>> SliceItem(List<Tensor<T>> batchOutputs, int batchIndex)
    {
        var itemOutputs = new List<Tensor<T>>(batchOutputs.Count);
        foreach (var output in batchOutputs)
        {
            int batchSize = output.Shape[0];
            if (batchSize == 1 && batchIndex == 0)
            {
                itemOutputs.Add(output);
                continue;
            }

            var itemShape = new int[output.Shape.Length];
            itemShape[0] = 1;
            for (int d = 1; d < output.Shape.Length; d++)
                itemShape[d] = output.Shape[d];

            var itemTensor = new Tensor<T>(itemShape);
            int elementsPerItem = output.Length / batchSize;
            int sourceOffset = batchIndex * elementsPerItem;
            for (int j = 0; j < elementsPerItem; j++)
                itemTensor[j] = output[sourceOffset + j];

            itemOutputs.Add(itemTensor);
        }

        return itemOutputs;
    }
}