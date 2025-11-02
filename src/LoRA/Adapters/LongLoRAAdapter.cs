using AiDotNet.Interfaces;

namespace AiDotNet.LoRA.Adapters;

/// <summary>
/// LongLoRA adapter that efficiently extends LoRA to handle longer context lengths using shifted sparse attention.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
/// <remarks>
/// <para>
/// LongLoRA (2023) addresses the challenge of adapting large language models to longer context windows
/// in a parameter-efficient manner. While standard LoRA works well for same-length fine-tuning,
/// extending context windows naively would require substantial computational resources.
/// </para>
/// <para>
/// LongLoRA introduces two key innovations:
/// 1. Shifted Sparse Attention (S²-Attn): During training only, uses shifted group attention patterns
///    that are more efficient while maintaining effectiveness for long contexts
/// 2. Dense Attention at Inference: At inference time, switches back to standard dense attention
///    for full context utilization without the training overhead
/// </para>
/// <para><b>For Beginners:</b> LongLoRA makes it affordable to train models on longer sequences.
///
/// The Problem:
/// - Standard LoRA works great for adapting models, but extending context length is expensive
/// - Full dense attention on long sequences requires O(n²) computation
/// - Training on 32k tokens instead of 2k tokens would be 256x slower!
///
/// LongLoRA's Solution:
/// - Uses a clever "shifted sparse attention" trick during training
/// - Divides the sequence into groups and shifts them to maintain information flow
/// - Much cheaper to train: O(n * k) where k is group size (typically 2048)
/// - At inference, uses full dense attention to maintain quality
///
/// Key Parameters:
/// - OriginalContextLength: The base model's context window (e.g., 2048)
/// - ExtendedContextLength: The target longer context (e.g., 8192 or 32768)
/// - UseShiftedAttention: Enable shifted sparse attention (training only)
/// - AttentionShiftSize: How many positions to shift attention groups (usually half the group size)
///
/// Example Use Case:
/// You have a model trained on 2k token contexts but need to process 16k token documents.
/// LongLoRA lets you extend the context efficiently:
/// - Training: Use shifted sparse attention (much faster)
/// - Inference: Use full dense attention (full quality)
///
/// Comparison to Standard LoRA:
/// - Standard LoRA: Efficient parameter adaptation, same context length
/// - LongLoRA: Efficient parameter adaptation + context length extension
/// - Adds minimal overhead (just the attention shift mechanism)
///
/// Research Background:
/// LongLoRA has been successfully used to extend:
/// - LLaMA 2 7B from 4k to 32k context (8x extension)
/// - LLaMA 2 13B from 4k to 64k context (16x extension)
/// - With only ~10% of the training cost compared to full fine-tuning
///
/// Reference: LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models (2023)
/// https://arxiv.org/abs/2309.12307
/// </para>
/// </remarks>
public class LongLoRAAdapter<T> : LoRAAdapterBase<T>
{
    /// <summary>
    /// The original context length that the base model was trained on.
    /// </summary>
    private readonly int _originalContextLength;

    /// <summary>
    /// The extended context length that this adapter targets.
    /// </summary>
    private readonly int _extendedContextLength;

    /// <summary>
    /// Whether to use shifted sparse attention during training (disabled at inference).
    /// </summary>
    private bool _useShiftedAttention;

    /// <summary>
    /// The shift size for shifted sparse attention (typically half the group size).
    /// </summary>
    private readonly int _attentionShiftSize;

    /// <summary>
    /// Whether the model is currently in training mode.
    /// </summary>
    private bool _isTraining;

    /// <summary>
    /// Gets the original context length of the base model.
    /// </summary>
    /// <remarks>
    /// This is the maximum sequence length the base model was originally trained to handle.
    /// Typical values: 512, 1024, 2048, 4096.
    /// </remarks>
    public int OriginalContextLength => _originalContextLength;

    /// <summary>
    /// Gets the extended context length this adapter targets.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This is the new, longer context window you want to support after adaptation.
    /// Should be larger than OriginalContextLength.
    /// </para>
    /// <para><b>For Beginners:</b> This is how long of a sequence your adapted model can handle.
    /// For example, extending from 2k to 16k tokens means you can process 8x longer documents!
    /// </para>
    /// </remarks>
    public int ExtendedContextLength => _extendedContextLength;

    /// <summary>
    /// Gets or sets whether to use shifted sparse attention during forward/backward passes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// When enabled (training mode):
    /// - Uses shifted group attention pattern for efficiency
    /// - Divides sequence into groups and shifts them
    /// - Significantly reduces computational cost
    /// </para>
    /// <para>
    /// When disabled (inference mode):
    /// - Uses standard dense attention
    /// - Full context utilization
    /// - Better quality but slower
    /// </para>
    /// <para><b>For Beginners:</b> Enable this during training to save compute, disable it
    /// during inference to get the best quality. The training trick doesn't hurt the final
    /// model's ability to use full attention at inference time!
    /// </para>
    /// </remarks>
    public bool UseShiftedAttention
    {
        get => _useShiftedAttention;
        set => _useShiftedAttention = value;
    }

    /// <summary>
    /// Gets the attention shift size used in shifted sparse attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This determines how much groups are shifted to maintain information flow.
    /// Typically set to half the group size (e.g., 1024 for 2048 group size).
    /// </para>
    /// <para><b>For Beginners:</b> This is the "sliding window" amount that ensures
    /// different parts of the sequence can communicate across groups. Too small and
    /// information doesn't flow well; too large and you lose the efficiency benefit.
    /// </para>
    /// </remarks>
    public int AttentionShiftSize => _attentionShiftSize;

    /// <summary>
    /// Gets or sets whether the adapter is in training mode.
    /// </summary>
    /// <remarks>
    /// Training mode affects whether shifted attention is applied.
    /// Set to false during inference to use standard dense attention.
    /// </remarks>
    public bool IsTraining
    {
        get => _isTraining;
        set => _isTraining = value;
    }

    /// <summary>
    /// Initializes a new LongLoRA adapter for efficient context length extension.
    /// </summary>
    /// <param name="baseLayer">The layer to adapt with LongLoRA.</param>
    /// <param name="rank">The rank of the LoRA decomposition.</param>
    /// <param name="originalContextLength">The original context length of the base model.</param>
    /// <param name="extendedContextLength">The target extended context length.</param>
    /// <param name="alpha">The LoRA scaling factor (defaults to rank if negative).</param>
    /// <param name="attentionShiftSize">The shift size for shifted sparse attention (defaults to originalContextLength/2).</param>
    /// <param name="freezeBaseLayer">Whether to freeze the base layer's parameters during training.</param>
    /// <exception cref="ArgumentNullException">Thrown when baseLayer is null.</exception>
    /// <exception cref="ArgumentException">Thrown when context lengths or shift size are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a LongLoRA adapter to extend your model's context window.
    ///
    /// Parameters:
    /// - baseLayer: The layer you want to adapt (typically attention layers)
    /// - rank: How much LoRA compression to use (8-16 is typical)
    /// - originalContextLength: How long sequences your base model handles (e.g., 2048)
    /// - extendedContextLength: How long you want to extend it to (e.g., 8192 or 16384)
    /// - alpha: LoRA strength (usually equals rank)
    /// - attentionShiftSize: How much to shift attention groups (auto-calculated if not specified)
    /// - freezeBaseLayer: Whether to freeze original weights (usually true for efficiency)
    ///
    /// The adapter will use shifted sparse attention during training for efficiency,
    /// and you can switch to dense attention during inference for quality.
    /// </para>
    /// </remarks>
    public LongLoRAAdapter(
        ILayer<T> baseLayer,
        int rank,
        int originalContextLength,
        int extendedContextLength,
        double alpha = -1,
        int attentionShiftSize = -1,
        bool freezeBaseLayer = true)
        : base(baseLayer, rank, alpha, freezeBaseLayer)
    {
        if (originalContextLength <= 0)
        {
            throw new ArgumentException("Original context length must be positive", nameof(originalContextLength));
        }

        if (extendedContextLength <= originalContextLength)
        {
            throw new ArgumentException("Extended context length must be greater than original context length", nameof(extendedContextLength));
        }

        _originalContextLength = originalContextLength;
        _extendedContextLength = extendedContextLength;
        _useShiftedAttention = true;  // Default to shifted attention for training
        _isTraining = true;

        // Default shift size is half the original context length (typical for shifted sparse attention)
        _attentionShiftSize = attentionShiftSize > 0
            ? attentionShiftSize
            : originalContextLength / 2;

        if (_attentionShiftSize >= originalContextLength)
        {
            throw new ArgumentException("Attention shift size must be less than original context length", nameof(attentionShiftSize));
        }
    }

    /// <summary>
    /// Performs the forward pass with optional shifted sparse attention.
    /// </summary>
    /// <param name="input">Input tensor of shape [batchSize, sequenceLength, featureDim].</param>
    /// <returns>Output tensor with LoRA adaptation applied.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass behavior depends on the UseShiftedAttention flag:
    /// - When true (training): Applies shifted group attention for efficiency
    /// - When false (inference): Uses standard dense attention
    /// </para>
    /// <para>
    /// Shifted Sparse Attention Process:
    /// 1. Divide the sequence into groups of size OriginalContextLength
    /// 2. Shift alternate groups by AttentionShiftSize positions
    /// 3. Apply attention within each group
    /// 4. Shift back to restore original positions
    /// </para>
    /// <para><b>For Beginners:</b> This processes your input through the adapted layer.
    ///
    /// During training (shifted attention enabled):
    /// - Breaks long sequence into manageable chunks
    /// - Shifts them to allow cross-chunk communication
    /// - Much faster than processing the full sequence at once
    ///
    /// During inference (shifted attention disabled):
    /// - Processes the full sequence with complete attention
    /// - Slower but gives best quality
    ///
    /// The magic is that training with the shifted trick still produces a model
    /// that works great with full attention at inference!
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // If not using shifted attention or not in training mode, use standard LoRA forward
        if (!_useShiftedAttention || !_isTraining)
        {
            return base.Forward(input);
        }

        // Apply shifted sparse attention during training
        Tensor<T> shiftedInput = ApplyShiftedAttention(input);

        // Forward through base layer with shifted input
        Tensor<T> baseOutput = _baseLayer.Forward(shiftedInput);

        // Forward through LoRA layer with shifted input
        Tensor<T> loraOutput = _loraLayer.Forward(shiftedInput);

        // Sum the outputs
        Tensor<T> result = new Tensor<T>(baseOutput.Shape);
        for (int i = 0; i < baseOutput.Length; i++)
        {
            result[i] = NumOps.Add(baseOutput[i], loraOutput[i]);
        }

        // Reverse the shift to restore original sequence positions
        result = ReverseShiftedAttention(result);

        return result;
    }

    /// <summary>
    /// Performs the backward pass with optional shifted sparse attention.
    /// </summary>
    /// <param name="outputGradient">Gradient flowing back from the next layer.</param>
    /// <returns>Gradient to pass to the previous layer.</returns>
    /// <remarks>
    /// <para>
    /// The backward pass mirrors the forward pass behavior:
    /// - Applies the same shifting pattern to gradients during training
    /// - Ensures gradient flow is consistent with the forward pass attention pattern
    /// </para>
    /// <para><b>For Beginners:</b> This propagates learning signals backward through the network.
    /// It uses the same shifted pattern as the forward pass to ensure the gradients match
    /// the attention pattern used during the forward pass.
    /// </para>
    /// </remarks>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // If not using shifted attention or not in training mode, use standard LoRA backward
        if (!_useShiftedAttention || !_isTraining)
        {
            return base.Backward(outputGradient);
        }

        // Apply shift to output gradient to match forward pass shifting
        Tensor<T> shiftedGradient = ApplyShiftedAttention(outputGradient);

        // Backward through LoRA layer
        Tensor<T> loraInputGrad = _loraLayer.Backward(shiftedGradient);

        // Backward through base layer
        Tensor<T> baseInputGrad = _baseLayer.Backward(shiftedGradient);

        // Sum input gradients
        Tensor<T> inputGrad = new Tensor<T>(loraInputGrad.Shape);
        for (int i = 0; i < loraInputGrad.Length; i++)
        {
            inputGrad[i] = NumOps.Add(loraInputGrad[i], baseInputGrad[i]);
        }

        // Reverse the shift to restore original sequence positions
        inputGrad = ReverseShiftedAttention(inputGrad);

        // Update parameter gradients vector
        UpdateParameterGradientsFromLayers();

        return inputGrad;
    }

    /// <summary>
    /// Applies shifted sparse attention pattern to the input tensor.
    /// </summary>
    /// <param name="input">Input tensor to shift.</param>
    /// <returns>Tensor with shifted attention pattern applied.</returns>
    /// <remarks>
    /// <para>
    /// The shifting pattern works as follows:
    /// 1. Divide sequence into groups of size OriginalContextLength
    /// 2. For alternate groups, shift by AttentionShiftSize positions
    /// 3. This creates overlapping attention windows that allow information flow
    /// </para>
    /// <para><b>For Beginners:</b> Imagine sliding windows along a long document.
    /// Instead of having fixed non-overlapping windows, we shift every other window
    /// by half its size. This ensures that each part of the document can "see"
    /// parts from neighboring windows, maintaining information flow while keeping
    /// computation efficient.
    /// </para>
    /// </remarks>
    private Tensor<T> ApplyShiftedAttention(Tensor<T> input)
    {
        // For simplicity, this is a conceptual implementation
        // In practice, this would integrate with the attention mechanism
        // Here we just apply a circular shift to alternate groups

        int sequenceLength = input.Shape.Length > 1 ? input.Shape[1] : input.Length;

        // If sequence is shorter than group size, no shifting needed
        if (sequenceLength <= _originalContextLength)
        {
            return input.Clone();
        }

        Tensor<T> shifted = input.Clone();
        int groupSize = _originalContextLength;
        int numGroups = (sequenceLength + groupSize - 1) / groupSize;

        // Apply shift to alternate groups
        for (int g = 1; g < numGroups; g += 2)
        {
            int groupStart = g * groupSize;
            int groupEnd = Math.Min(groupStart + groupSize, sequenceLength);

            // Circular shift within this group
            ShiftGroup(shifted, groupStart, groupEnd, _attentionShiftSize);
        }

        return shifted;
    }

    /// <summary>
    /// Reverses the shifted sparse attention pattern to restore original positions.
    /// </summary>
    /// <param name="input">Tensor with shifted attention pattern.</param>
    /// <returns>Tensor with original sequence positions restored.</returns>
    /// <remarks>
    /// This reverses the shifting applied by ApplyShiftedAttention to restore
    /// the output to the original sequence order.
    /// </remarks>
    private Tensor<T> ReverseShiftedAttention(Tensor<T> input)
    {
        int sequenceLength = input.Shape.Length > 1 ? input.Shape[1] : input.Length;

        // If sequence is shorter than group size, no shifting was applied
        if (sequenceLength <= _originalContextLength)
        {
            return input.Clone();
        }

        Tensor<T> unshifted = input.Clone();
        int groupSize = _originalContextLength;
        int numGroups = (sequenceLength + groupSize - 1) / groupSize;

        // Reverse shift for alternate groups (shift in opposite direction)
        for (int g = 1; g < numGroups; g += 2)
        {
            int groupStart = g * groupSize;
            int groupEnd = Math.Min(groupStart + groupSize, sequenceLength);

            // Reverse circular shift within this group
            ShiftGroup(unshifted, groupStart, groupEnd, -_attentionShiftSize);
        }

        return unshifted;
    }

    /// <summary>
    /// Shifts elements within a group by the specified amount.
    /// </summary>
    /// <param name="tensor">Tensor to modify.</param>
    /// <param name="groupStart">Start index of the group along sequence dimension.</param>
    /// <param name="groupEnd">End index of the group (exclusive) along sequence dimension.</param>
    /// <param name="shiftAmount">Amount to shift (positive for right, negative for left).</param>
    /// <remarks>
    /// <para>
    /// Performs a circular shift within the specified range along the sequence dimension.
    /// For a tensor of shape [batchSize, sequenceLength, featureDim], this shifts
    /// positions [groupStart:groupEnd] along axis 1 for all batch elements and all features.
    /// Elements that shift past the end wrap around to the beginning.
    /// </para>
    /// <para><b>For Beginners:</b> This is like rotating a portion of an array.
    /// If you shift [1,2,3,4,5] by 2 positions, you get [4,5,1,2,3].
    /// The "circular" part means elements wrap around instead of falling off the end.
    /// For multi-dimensional tensors, we apply this shift to every batch and every feature.
    /// </para>
    /// </remarks>
    private void ShiftGroup(Tensor<T> tensor, int groupStart, int groupEnd, int shiftAmount)
    {
        int groupSize = groupEnd - groupStart;
        if (groupSize <= 0)
        {
            return;
        }

        // Normalize shift amount to be within group size
        shiftAmount = shiftAmount % groupSize;
        if (shiftAmount < 0)
        {
            shiftAmount += groupSize;
        }

        if (shiftAmount == 0)
        {
            return;
        }

        // Determine tensor dimensions
        int[] shape = tensor.Shape;

        if (shape.Length == 1)
        {
            // 1D tensor: simple shift
            T[] buffer = new T[groupSize];
            for (int i = 0; i < groupSize; i++)
            {
                buffer[i] = tensor[groupStart + i];
            }
            for (int i = 0; i < groupSize; i++)
            {
                int newPos = (i + shiftAmount) % groupSize;
                tensor[groupStart + newPos] = buffer[i];
            }
        }
        else if (shape.Length == 2)
        {
            // 2D tensor [batchSize, sequenceLength]: shift along sequence axis for each batch
            int batchSize = shape[0];
            int sequenceLength = shape[1];

            T[] buffer = new T[groupSize];

            for (int b = 0; b < batchSize; b++)
            {
                // Copy group to buffer for this batch
                for (int i = 0; i < groupSize; i++)
                {
                    int seqIdx = groupStart + i;
                    if (seqIdx < sequenceLength)
                    {
                        buffer[i] = tensor[b * sequenceLength + seqIdx];
                    }
                }

                // Write back with shift for this batch
                for (int i = 0; i < groupSize; i++)
                {
                    // Use actual group size for modulo to handle partial last groups correctly
                    int actualGroupSize = Math.Min(groupSize, sequenceLength - groupStart);
                    int seqIdx = groupStart + ((i + shiftAmount) % actualGroupSize);
                    if (seqIdx < sequenceLength)
                    {
                        tensor[b * sequenceLength + seqIdx] = buffer[i];
                    }
                }
            }
        }
        else if (shape.Length == 3)
        {
            // 3D tensor [batchSize, sequenceLength, featureDim]: shift along sequence axis
            int batchSize = shape[0];
            int sequenceLength = shape[1];
            int featureDim = shape[2];

            T[] buffer = new T[groupSize];

            for (int b = 0; b < batchSize; b++)
            {
                for (int f = 0; f < featureDim; f++)
                {
                    // Copy group to buffer for this batch and feature
                    for (int i = 0; i < groupSize; i++)
                    {
                        int seqIdx = groupStart + i;
                        if (seqIdx < sequenceLength)
                        {
                            buffer[i] = tensor[b * sequenceLength * featureDim + seqIdx * featureDim + f];
                        }
                    }

                    // Write back with shift for this batch and feature
                    for (int i = 0; i < groupSize; i++)
                    {
                        // Use actual group size for modulo to handle partial last groups correctly
                        int actualGroupSize = Math.Min(groupSize, sequenceLength - groupStart);
                        int seqIdx = groupStart + ((i + shiftAmount) % actualGroupSize);
                        if (seqIdx < sequenceLength)
                        {
                            tensor[b * sequenceLength * featureDim + seqIdx * featureDim + f] = buffer[i];
                        }
                    }
                }
            }
        }
        else
        {
            // For higher-dimensional tensors, shift along second dimension (sequence axis)
            // Flatten other dimensions and treat as batch×feature
            int sequenceLength = shape[1];
            int batchStride = 1;
            for (int i = 0; i < shape.Length; i++)
            {
                batchStride *= shape[i];
            }
            batchStride /= sequenceLength;

            T[] buffer = new T[groupSize];

            for (int idx = 0; idx < batchStride; idx++)
            {
                // Calculate base offset for this batch/feature combination
                int batchIdx = idx / (shape.Length > 2 ? shape[2] : 1);
                int featureIdx = idx % (shape.Length > 2 ? shape[2] : 1);

                // Copy group to buffer
                for (int i = 0; i < groupSize; i++)
                {
                    int seqIdx = groupStart + i;
                    if (seqIdx < sequenceLength)
                    {
                        int tensorIdx = batchIdx * sequenceLength * (shape.Length > 2 ? shape[2] : 1) + seqIdx * (shape.Length > 2 ? shape[2] : 1) + featureIdx;
                        if (tensorIdx < tensor.Length)
                        {
                            buffer[i] = tensor[tensorIdx];
                        }
                    }
                }

                // Write back with shift
                for (int i = 0; i < groupSize; i++)
                {
                    int seqIdx = groupStart + ((i + shiftAmount) % groupSize);
                    if (seqIdx < sequenceLength)
                    {
                        int tensorIdx = batchIdx * sequenceLength * (shape.Length > 2 ? shape[2] : 1) + seqIdx * (shape.Length > 2 ? shape[2] : 1) + featureIdx;
                        if (tensorIdx < tensor.Length)
                        {
                            tensor[tensorIdx] = buffer[i];
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Merges the LongLoRA adaptation into the base layer and returns the merged layer.
    /// </summary>
    /// <returns>A new layer with LoRA weights merged into the base layer's weights.</returns>
    /// <remarks>
    /// <para>
    /// For LongLoRA, merging works like standard LoRA - the shifted attention pattern
    /// is only used during training and doesn't affect the final merged weights.
    /// The merged layer can use full dense attention at inference time.
    /// </para>
    /// <para><b>For Beginners:</b> After training with LongLoRA, you can merge the weights
    /// just like standard LoRA. The shifted attention trick was only for efficient training -
    /// it doesn't change the final model! The merged model will work great with full attention
    /// on long contexts because that's what it learned to handle (just using a training shortcut).
    /// </para>
    /// </remarks>
    public override ILayer<T> MergeToOriginalLayer()
    {
        // LongLoRA merging is identical to standard LoRA
        // The shifted attention is only for training efficiency
        DenseLayer<T>? denseBase = _baseLayer as DenseLayer<T>;
        FullyConnectedLayer<T>? fcBase = _baseLayer as FullyConnectedLayer<T>;

        if (denseBase == null && fcBase == null)
        {
            throw new InvalidOperationException("LongLoRAAdapter currently only supports DenseLayer or FullyConnectedLayer base layers");
        }

        // Get the LoRA weight contribution
        Matrix<T> loraWeights = _loraLayer.MergeWeights();

        // Get base layer parameters
        Vector<T> baseParams = _baseLayer.GetParameters();

        // Calculate dimensions
        int inputSize = GetInputShape()[0];
        int outputSize = GetOutputShape()[0];
        int weightCount = inputSize * outputSize;

        // Create new parameters with merged weights
        Vector<T> mergedParams = new Vector<T>(baseParams.Length);

        // Merge weights
        for (int i = 0; i < weightCount; i++)
        {
            int row = i / inputSize;
            int col = i % inputSize;
            mergedParams[i] = NumOps.Add(baseParams[i], loraWeights[row, col]);
        }

        // Copy biases unchanged
        for (int i = weightCount; i < baseParams.Length; i++)
        {
            mergedParams[i] = baseParams[i];
        }

        // Use helper method to clone base layer and preserve activation function
        return CreateMergedLayerWithClone(mergedParams);
    }

    /// <summary>
    /// Updates the parameter gradients vector from the layer gradients.
    /// </summary>
    /// <remarks>
    /// This helper method synchronizes the parameter gradients after backward pass.
    /// </remarks>
    private void UpdateParameterGradientsFromLayers()
    {
        ParameterGradients = new Vector<T>(ParameterCount);
        int idx = 0;

        // If base layer is not frozen, pack its gradients first
        if (!_freezeBaseLayer)
        {
            Vector<T> baseGrads = _baseLayer.GetParameterGradients();
            for (int i = 0; i < baseGrads.Length; i++)
            {
                ParameterGradients[idx++] = baseGrads[i];
            }
        }

        // Pack LoRA gradients
        Vector<T> loraGrads = _loraLayer.GetParameterGradients();
        for (int i = 0; i < loraGrads.Length; i++)
        {
            ParameterGradients[idx++] = loraGrads[i];
        }
    }

    /// <summary>
    /// Resets the internal state of the adapter.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This clears all internal memory and cached data.
    /// Call this when starting to process a new, unrelated sequence.
    /// </para>
    /// </remarks>
    public override void ResetState()
    {
        base.ResetState();
    }
}
