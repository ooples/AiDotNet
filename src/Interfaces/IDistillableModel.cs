namespace AiDotNet.Interfaces;

using AiDotNet.Interfaces;
using AiDotNet.Compression.KnowledgeDistillation;

/// <summary>
/// Interface for models that support knowledge distillation.
/// </summary>
/// <remarks>
/// <para>
/// This interface should be implemented by models that can act as teachers in knowledge distillation.
/// </para>
/// <para><b>For Beginners:</b> This marks a model as able to teach smaller models.
/// 
/// Models that implement this interface know how to:
/// - Create a smaller student model architecture
/// - Generate soft targets for the student to learn from
/// - Transfer their knowledge to the student model
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TModel">The type of the model.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public interface IDistillableModel<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    /// <summary>
    /// Creates a student model architecture based on this teacher model.
    /// </summary>
    /// <param name="studentSizeRatio">The size ratio for the student model (0-1).</param>
    /// <returns>The student model architecture.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a smaller architecture for the student model based on the
    /// teacher model's architecture and the specified size ratio.
    /// </para>
    /// <para><b>For Beginners:</b> This designs a smaller version of the teacher model.
    /// 
    /// If the teacher model is like a 12-layer network with 768 neurons per layer,
    /// a student with ratio 0.5 might be a 6-layer network with 384 neurons per layer.
    /// </para>
    /// </remarks>
    object CreateStudentArchitecture(double studentSizeRatio);

    /// <summary>
    /// Creates a student model initialized for distillation.
    /// </summary>
    /// <param name="studentArchitecture">The architecture for the student model.</param>
    /// <returns>A new student model initialized for distillation.</returns>
    /// <remarks>
    /// <para>
    /// This method creates and initializes a new student model based on the
    /// provided architecture.
    /// </para>
    /// <para><b>For Beginners:</b> This creates the actual smaller model.
    /// 
    /// Using the architecture designed by CreateStudentArchitecture,
    /// this method:
    /// - Creates the actual student model
    /// - Initializes its weights appropriately
    /// - Prepares it to learn from the teacher
    /// </para>
    /// </remarks>
    TModel CreateStudentModel(object studentArchitecture);

    /// <summary>
    /// Distills knowledge from the teacher model to the student model.
    /// </summary>
    /// <param name="student">The student model to train.</param>
    /// <param name="unlabeledData">Unlabeled data for distillation.</param>
    /// <param name="labeledData">Optional labeled data for distillation.</param>
    /// <param name="parameters">Parameters for the distillation process.</param>
    /// <remarks>
    /// <para>
    /// This method trains the student model to mimic the teacher model using
    /// knowledge distillation.
    /// </para>
    /// <para><b>For Beginners:</b> This teaches the student model to mimic the teacher.
    /// 
    /// The teacher model:
    /// - Generates predictions on the unlabeled data
    /// - Guides the student learning process
    /// - Transfers its "knowledge" to the student
    /// 
    /// After distillation, the student should approach the teacher's accuracy
    /// despite having far fewer parameters.
    /// </para>
    /// </remarks>
    void DistillKnowledgeToStudent(
        IDistillableStudent<T, TInput, TOutput> student,
        TInput[] unlabeledData,
        (TInput[] inputs, TOutput[] outputs)? labeledData,
        DistillationParameters parameters);
}