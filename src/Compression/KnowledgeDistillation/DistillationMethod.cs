namespace AiDotNet.Compression.KnowledgeDistillation;

/// <summary>
/// Defines the available knowledge distillation methods.
/// </summary>
/// <remarks>
/// <para>
/// Different distillation methods use different approaches to transfer knowledge
/// from the teacher to the student model.
/// </para>
/// <para><b>For Beginners:</b> This defines different ways to teach the student model.
/// 
/// Different methods focus on different aspects of the teacher's behavior:
/// - Some use just the final outputs
/// - Others use intermediate representations
/// - Some focus on attention patterns
/// 
/// Each method has its own strengths and may work better for different models.
/// </para>
/// </remarks>
public enum DistillationMethod
{
    /// <summary>
    /// The original knowledge distillation method from Hinton et al.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Vanilla knowledge distillation uses the teacher model's soft output probabilities
    /// to train the student model.
    /// </para>
    /// <para><b>For Beginners:</b> This is the classic knowledge distillation method.
    /// 
    /// Vanilla distillation:
    /// - Uses softened probability outputs from the teacher
    /// - The student learns to match these probability distributions
    /// - Simple, effective, and widely used
    /// - Works well for classification tasks
    /// </para>
    /// </remarks>
    Vanilla = 0,
    
    /// <summary>
    /// Distillation that also transfers attention patterns from teacher to student.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Attention distillation transfers both output probabilities and attention patterns
    /// from the teacher to the student.
    /// </para>
    /// <para><b>For Beginners:</b> This also teaches attention patterns to the student.
    /// 
    /// Attention distillation:
    /// - Transfers not just what the teacher predicts
    /// - But also how the teacher attends to different parts of the input
    /// - Especially valuable for transformer models
    /// - Helps the student learn more nuanced behaviors
    /// </para>
    /// </remarks>
    Attention = 1,
    
    /// <summary>
    /// Distillation that transfers intermediate feature representations.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Feature distillation transfers intermediate layer outputs from the teacher
    /// to corresponding layers in the student.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches intermediate representations to the student.
    /// 
    /// Feature distillation:
    /// - Has the student learn from the teacher's internal representations
    /// - Not just the final outputs
    /// - Requires alignment between teacher and student layers
    /// - Can transfer more detailed knowledge
    /// </para>
    /// </remarks>
    Feature = 2,
    
    /// <summary>
    /// Combination of multiple distillation approaches.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Combined distillation uses multiple knowledge transfer mechanisms simultaneously.
    /// </para>
    /// <para><b>For Beginners:</b> This uses multiple teaching methods together.
    /// 
    /// Combined distillation:
    /// - Uses a mix of different distillation approaches
    /// - Transfers outputs, attentions, and features
    /// - More complex but often more effective
    /// - Especially good for complex models
    /// </para>
    /// </remarks>
    Combined = 3
}