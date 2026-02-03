namespace AiDotNet.Enums;

/// <summary>
/// Specifies the Quantization-Aware Training (QAT) method to use during model training.
/// QAT simulates quantization effects during training so the model learns to be robust to low precision.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Normally, we train a model in full precision (32-bit) and then
/// compress it afterward (PTQ). QAT is smarter - it simulates the compression DURING training,
/// so the model learns to work well even when compressed.</para>
///
/// <para><b>Analogy:</b> It's like training for a marathon at high altitude - when you compete
/// at sea level, you perform better because you trained under harder conditions.</para>
///
/// <para><b>QAT vs PTQ Comparison:</b></para>
/// <list type="table">
/// <listheader>
/// <term>Method</term>
/// <description>Accuracy at 4-bit</description>
/// </listheader>
/// <item><term>PTQ (Post-Training)</term><description>85-95% of original</description></item>
/// <item><term>QAT (Quantization-Aware)</term><description>95-99% of original</description></item>
/// </list>
///
/// <para><b>Research References:</b></para>
/// <list type="bullet">
/// <item><description>Standard QAT: Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference" (2018)</description></item>
/// <item><description>EfficientQAT: "Efficient Quantization-Aware Training for LLMs" (ACL 2025)</description></item>
/// <item><description>ZeroQAT: "End-to-End On-Device Quantization-Aware Training for LLMs at Inference Cost" (2025)</description></item>
/// <item><description>ParetoQ: Liu et al., achieves SOTA across bit widths (2025)</description></item>
/// </list>
/// </remarks>
public enum QATMethod
{
    /// <summary>
    /// Standard QAT using Straight-Through Estimator (STE) for gradient propagation.
    /// Inserts fake quantization nodes after weight layers and passes gradients through unchanged.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> During training, we pretend to quantize (compress) the values,
    /// but when calculating gradients for learning, we pretend the quantization didn't happen.
    /// This trick (STE) lets gradients flow through the non-differentiable rounding operation.</para>
    /// <para><b>How it works:</b></para>
    /// <list type="number">
    /// <item><description>Forward pass: Apply fake quantization (quantize then immediately dequantize)</description></item>
    /// <item><description>Backward pass: Pass gradients through unchanged (Straight-Through Estimator)</description></item>
    /// </list>
    /// <para><b>Memory requirement:</b> Same as normal training (stores full-precision gradients)</para>
    /// <para><b>Best for:</b> Standard use cases, good baseline</para>
    /// </remarks>
    Standard,

    /// <summary>
    /// EfficientQAT - memory-efficient QAT optimized for large language models.
    /// Uses block-wise quantization and efficient gradient computation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Standard QAT uses a lot of memory because it keeps full-precision
    /// copies of weights. EfficientQAT is smarter about memory, letting you train bigger models
    /// on the same hardware.</para>
    /// <para><b>Key Innovations:</b></para>
    /// <list type="bullet">
    /// <item><description>Block-wise quantization with shared scales</description></item>
    /// <item><description>Efficient gradient computation avoiding full weight materialization</description></item>
    /// <item><description>Progressive quantization schedule (start at higher bits, reduce over time)</description></item>
    /// </list>
    /// <para><b>Memory reduction:</b> 2-4x less memory than standard QAT</para>
    /// <para><b>Best for:</b> Large models, memory-constrained environments</para>
    /// <para><b>Reference:</b> ACL 2025 paper on Efficient QAT for LLMs</para>
    /// </remarks>
    EfficientQAT,

    /// <summary>
    /// ZeroQAT - zeroth-order optimization based QAT that doesn't require backpropagation.
    /// Enables QAT on extremely memory-constrained devices (e.g., 8GB GPU or mobile).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Normal training uses "backpropagation" which requires storing
    /// lots of intermediate values (high memory). ZeroQAT uses a different approach that estimates
    /// gradients without storing all those values.</para>
    /// <para><b>Key Innovation:</b> Uses zeroth-order optimization (gradient estimation via finite
    /// differences) instead of backpropagation, dramatically reducing memory requirements.</para>
    /// <para><b>Capabilities:</b></para>
    /// <list type="bullet">
    /// <item><description>Fine-tune 13B model at 2-4 bits on single 8GB GPU</description></item>
    /// <item><description>Fine-tune 6.7B model on mobile device (OnePlus 12)</description></item>
    /// </list>
    /// <para><b>Trade-off:</b> Slower convergence than standard QAT, but enables previously impossible scenarios</para>
    /// <para><b>Best for:</b> Edge devices, mobile deployment, extreme memory constraints</para>
    /// </remarks>
    ZeroQAT,

    /// <summary>
    /// ParetoQ - state-of-the-art QAT achieving optimal accuracy across all bit widths.
    /// Uses different techniques for different bit widths (Elastic Binarization for 1-bit, LSQ for 3+, SEQ for 2-bit).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> ParetoQ is like having a specialist for each compression level.
    /// Instead of using one technique for all bit widths, it uses the best technique for each.</para>
    /// <para><b>Techniques by bit width:</b></para>
    /// <list type="bullet">
    /// <item><description>1-bit: Elastic Binarization (learns scaling factors)</description></item>
    /// <item><description>2-bit: SEQ (Stochastic Equalization Quantization)</description></item>
    /// <item><description>3-bit and higher: LSQ (Learned Step Size Quantization)</description></item>
    /// </list>
    /// <para><b>Performance:</b> Achieves Pareto-optimal accuracy/efficiency across all bit widths</para>
    /// <para><b>Best for:</b> When you need the absolute best accuracy at any bit width</para>
    /// </remarks>
    ParetoQ,

    /// <summary>
    /// QA-BLoRA - Quantization-Aware fine-tuning with Balanced Low-Rank Adaptation.
    /// Combines QAT with LoRA for efficient fine-tuning of quantized models.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Combines two powerful techniques: QAT (training for compression)
    /// and LoRA (efficient fine-tuning using small adapter matrices). The result is a fine-tuned
    /// model that's already compressed and ready for deployment.</para>
    /// <para><b>Key Innovation:</b> Aligns LoRA adapter training with block-wise quantization,
    /// directly producing low-precision inference models.</para>
    /// <para><b>Benefits:</b></para>
    /// <list type="bullet">
    /// <item><description>No separate quantization step after training</description></item>
    /// <item><description>Better accuracy than QLoRA + PTQ</description></item>
    /// <item><description>Memory-efficient fine-tuning</description></item>
    /// </list>
    /// <para><b>Best for:</b> Fine-tuning pre-trained models for deployment on edge devices</para>
    /// </remarks>
    QABLoRA
}
