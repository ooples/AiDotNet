namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of teacher model to use for knowledge distillation.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> The teacher model is the "expert" that guides the student model's learning.
/// Different teacher types are suited for different scenarios and distillation goals.</para>
///
/// <para><b>Choosing a Teacher:</b>
/// - Use **NeuralNetwork** for standard NN-to-NN distillation
/// - Use **Ensemble** to combine knowledge from multiple models
/// - Use **Pretrained** to load from checkpoints or ONNX
/// - Use **Adaptive** for curriculum learning (progressive difficulty)
/// - Use **Online** when teacher should update during training</para>
/// </remarks>
public enum TeacherModelType
{
    /// <summary>
    /// Standard neural network teacher.
    /// Uses a single, pre-trained neural network as the teacher model.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Standard distillation scenarios, single teacher.</para>
    /// <para><b>Requirements:</b> Pre-trained teacher model.</para>
    /// <para><b>Pros:</b> Simple, straightforward, fast.</para>
    /// <para><b>Cons:</b> Limited to single model's knowledge.</para>
    /// </remarks>
    NeuralNetwork = 0,

    /// <summary>
    /// Ensemble of multiple teacher models.
    /// Combines predictions from multiple teachers (averaging, voting, or weighted combination).
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> High-accuracy requirements, combining diverse models.</para>
    /// <para><b>Requirements:</b> Multiple pre-trained teacher models.</para>
    /// <para><b>Pros:</b> More robust, captures diverse knowledge.</para>
    /// <para><b>Cons:</b> Slower (multiple forward passes), requires more memory.</para>
    /// </remarks>
    Ensemble = 1,

    /// <summary>
    /// Pretrained model loaded from checkpoint or ONNX.
    /// Loads a teacher from a saved checkpoint, ONNX model, or other serialized format.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Using external models, cross-framework distillation.</para>
    /// <para><b>Requirements:</b> Model checkpoint or ONNX file.</para>
    /// <para><b>Pros:</b> Reuse existing models, framework-agnostic.</para>
    /// <para><b>Cons:</b> May require format conversions.</para>
    /// </remarks>
    Pretrained = 2,

    /// <summary>
    /// Transformer-based teacher (BERT, GPT, ViT, etc.).
    /// Specialized for transformer architectures with attention mechanisms.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Language models, vision transformers, attention-based models.</para>
    /// <para><b>Requirements:</b> Transformer teacher model.</para>
    /// <para><b>Pros:</b> Supports attention distillation, handles sequences.</para>
    /// <para><b>Cons:</b> Specific to transformer architecture.</para>
    /// </remarks>
    Transformer = 3,

    /// <summary>
    /// Multi-modal teacher (e.g., CLIP, vision-language models).
    /// Handles multiple input modalities (text, images, audio, etc.).
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Cross-modal learning, vision-language tasks.</para>
    /// <para><b>Requirements:</b> Multi-modal pre-trained model.</para>
    /// <para><b>Pros:</b> Handles multiple modalities, rich representations.</para>
    /// <para><b>Cons:</b> Complex, requires multi-modal data.</para>
    /// </remarks>
    MultiModal = 4,

    /// <summary>
    /// Adaptive teacher that adjusts teaching based on student performance.
    /// Modulates difficulty or focus areas based on how well the student is learning.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Curriculum learning, progressive training.</para>
    /// <para><b>Requirements:</b> Teacher model + adaptation logic.</para>
    /// <para><b>Pros:</b> Optimizes teaching strategy, faster convergence.</para>
    /// <para><b>Cons:</b> More complex, requires performance monitoring.</para>
    /// </remarks>
    Adaptive = 5,

    /// <summary>
    /// Online teacher that updates during student training.
    /// Teacher weights are updated simultaneously with student (co-training).
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Continuous learning, evolving data distributions.</para>
    /// <para><b>Requirements:</b> Updateable teacher model.</para>
    /// <para><b>Pros:</b> Adapts to new data, maintains relevance.</para>
    /// <para><b>Cons:</b> Risk of teacher degradation, complex optimization.</para>
    /// </remarks>
    Online = 6,

    /// <summary>
    /// Curriculum teacher that provides progressive difficulty.
    /// Starts with easy samples and gradually increases difficulty.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Complex tasks, improving convergence.</para>
    /// <para><b>Requirements:</b> Teacher model + curriculum strategy.</para>
    /// <para><b>Pros:</b> Better convergence, handles complex tasks.</para>
    /// <para><b>Cons:</b> Requires curriculum design, longer training.</para>
    /// </remarks>
    Curriculum = 7,

    /// <summary>
    /// Self-teacher where model teaches itself (Born-Again Networks).
    /// The model acts as its own teacher to improve calibration and generalization.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Improving calibration, no separate teacher available.</para>
    /// <para><b>Requirements:</b> Initial trained model.</para>
    /// <para><b>Pros:</b> No separate teacher needed, improves calibration.</para>
    /// <para><b>Cons:</b> Requires multiple training generations.</para>
    /// </remarks>
    Self = 8,

    /// <summary>
    /// Quantized teacher with reduced precision (INT8, INT4).
    /// Uses quantized version of teacher for faster inference during distillation.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Fast distillation, resource-constrained environments.</para>
    /// <para><b>Requirements:</b> Quantized teacher model.</para>
    /// <para><b>Pros:</b> Faster, less memory, still effective.</para>
    /// <para><b>Cons:</b> Slight accuracy loss, quantization overhead.</para>
    /// </remarks>
    Quantized = 9,

    /// <summary>
    /// Distributed teacher split across multiple devices/nodes.
    /// Large teacher model is distributed for efficient inference.
    /// </summary>
    /// <remarks>
    /// <para><b>Best for:</b> Very large teachers, distributed training.</para>
    /// <para><b>Requirements:</b> Multi-device setup, large teacher.</para>
    /// <para><b>Pros:</b> Handles very large models, parallel processing.</para>
    /// <para><b>Cons:</b> Complex setup, communication overhead.</para>
    /// </remarks>
    Distributed = 10
}
