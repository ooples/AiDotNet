using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation.Teachers;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Factory for creating teacher models from enums and configurations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public static class TeacherModelFactory<T>
{
    /// <summary>
    /// Creates a teacher model from the specified type and configuration.
    /// </summary>
    /// <param name="teacherType">The type of teacher to create.</param>
    /// <param name="model">Optional: IFullModel to wrap as a teacher (required for NeuralNetwork type).</param>
    /// <param name="ensembleModels">Optional: Array of models for ensemble (required for Ensemble type).</param>
    /// <param name="ensembleWeights">Optional: Weights for ensemble models.</param>
    /// <param name="outputDimension">Optional: Output dimension (inferred from model if not provided).</param>
    /// <returns>A configured teacher model.</returns>
    public static ITeacherModel<Vector<T>, Vector<T>> CreateTeacher(
        TeacherModelType teacherType,
        IFullModel<T, Vector<T>, Vector<T>>? model = null,
        ITeacherModel<Vector<T>, Vector<T>>[]? ensembleModels = null,
        double[]? ensembleWeights = null,
        int? outputDimension = null)
    {
        return teacherType switch
        {
            TeacherModelType.NeuralNetwork => CreateNeuralNetworkTeacher(model, outputDimension),
            TeacherModelType.Ensemble => CreateEnsembleTeacher(ensembleModels, ensembleWeights),
            TeacherModelType.Pretrained => CreatePretrainedTeacher(model, outputDimension),
            TeacherModelType.Transformer => CreateTransformerTeacher(model, outputDimension),
            TeacherModelType.MultiModal => CreateMultiModalTeacher(ensembleModels, ensembleWeights),
            TeacherModelType.Adaptive => CreateAdaptiveTeacher(model, outputDimension),
            TeacherModelType.Online => CreateOnlineTeacher(model, outputDimension),
            TeacherModelType.Curriculum => CreateCurriculumTeacher(model, outputDimension),
            TeacherModelType.Self => CreateSelfTeacher(outputDimension ?? 10),
            TeacherModelType.Quantized => CreateQuantizedTeacher(model, outputDimension),
            TeacherModelType.Distributed => CreateDistributedTeacher(ensembleModels),
            _ => throw new ArgumentException($"Unknown teacher type: {teacherType}", nameof(teacherType))
        };
    }

    private static ITeacherModel<Vector<T>, Vector<T>> CreateNeuralNetworkTeacher(
        IFullModel<T, Vector<T>, Vector<T>>? model,
        int? outputDimension)
    {
        if (model == null)
            throw new ArgumentException("Model is required for NeuralNetwork teacher type");

        return new TeacherModelWrapper<T>(model);
    }

    private static ITeacherModel<Vector<T>, Vector<T>> CreateEnsembleTeacher(
        ITeacherModel<Vector<T>, Vector<T>>[]? ensembleModels,
        double[]? ensembleWeights)
    {
        if (ensembleModels == null || ensembleModels.Length == 0)
            throw new ArgumentException("Ensemble models are required for Ensemble teacher type");

        var aggregation = ensembleWeights != null
            ? EnsembleAggregationMode.WeightedAverage
            : EnsembleAggregationMode.WeightedAverage;

        return new EnsembleTeacherModel<T>(ensembleModels, ensembleWeights, aggregation);
    }

    private static ITeacherModel<Vector<T>, Vector<T>> CreatePretrainedTeacher(
        IFullModel<T, Vector<T>, Vector<T>>? model,
        int? outputDimension)
    {
        if (model == null)
            throw new ArgumentException("Model is required for Pretrained teacher type");
        if (!outputDimension.HasValue)
            throw new ArgumentException("Output dimension is required for Pretrained teacher type");

        return new PretrainedTeacherModel<T>(model.Predict, outputDimension.Value);
    }

    private static ITeacherModel<Vector<T>, Vector<T>> CreateTransformerTeacher(
        IFullModel<T, Vector<T>, Vector<T>>? model,
        int? outputDimension)
    {
        if (model == null)
            throw new ArgumentException("Model is required for Transformer teacher type");
        if (!outputDimension.HasValue)
            throw new ArgumentException("Output dimension is required for Transformer teacher type");

        return new TransformerTeacherModel<T>(model.Predict, outputDimension.Value);
    }

    private static ITeacherModel<Vector<T>, Vector<T>> CreateMultiModalTeacher(
        ITeacherModel<Vector<T>, Vector<T>>[]? modalityTeachers,
        double[]? modalityWeights)
    {
        if (modalityTeachers == null || modalityTeachers.Length == 0)
            throw new ArgumentException("Modality teachers are required for MultiModal teacher type");

        return new MultiModalTeacherModel<T>(modalityTeachers, modalityWeights);
    }

    /// <summary>
    /// Creates an adaptive teacher model wrapper.
    /// </summary>
    /// <remarks>
    /// <para><b>Architecture Note:</b> This method creates a simple wrapper around the base model.
    /// For adaptive temperature adjustment based on student performance, use
    /// <see cref="Strategies.AdaptiveDistillationStrategy{T}"/> instead.</para>
    ///
    /// <para>The AdaptiveTeacherModel class is maintained for backward compatibility but does not
    /// contain adaptive logic. Adaptive features (dynamic temperature, performance tracking) belong
    /// in the distillation strategy layer.</para>
    /// </remarks>
    private static ITeacherModel<Vector<T>, Vector<T>> CreateAdaptiveTeacher(
        IFullModel<T, Vector<T>, Vector<T>>? model,
        int? outputDimension)
    {
        if (model == null)
            throw new ArgumentException("Model is required for Adaptive teacher type");

        var baseTeacher = new TeacherModelWrapper<T>(model);
        return new AdaptiveTeacherModel<T>(baseTeacher);
    }

    private static ITeacherModel<Vector<T>, Vector<T>> CreateOnlineTeacher(
        IFullModel<T, Vector<T>, Vector<T>>? model,
        int? outputDimension)
    {
        if (model == null)
            throw new ArgumentException("Model is required for Online teacher type");
        if (!outputDimension.HasValue)
            throw new ArgumentException("Output dimension is required for Online teacher type");

        // Online teacher needs forward and update functions
        return new OnlineTeacherModel<T>(
            model.Predict,
            (pred, target) => { }, // No-op update for now
            outputDimension.Value,
            OnlineUpdateMode.EMA,
            updateRate: 0.999);
    }

    /// <summary>
    /// Creates a curriculum teacher model wrapper.
    /// </summary>
    /// <remarks>
    /// <para><b>Architecture Note:</b> This method creates a simple wrapper around the base model.
    /// For curriculum learning with progressive difficulty adjustment, use
    /// <see cref="Strategies.CurriculumDistillationStrategy{T}"/> instead.</para>
    ///
    /// <para>The CurriculumTeacherModel class is maintained for backward compatibility but does not
    /// contain curriculum logic. Curriculum features (easy-to-hard progression, difficulty-based
    /// temperature adjustment) belong in the distillation strategy layer.</para>
    /// </remarks>
    private static ITeacherModel<Vector<T>, Vector<T>> CreateCurriculumTeacher(
        IFullModel<T, Vector<T>, Vector<T>>? model,
        int? outputDimension)
    {
        if (model == null)
            throw new ArgumentException("Model is required for Curriculum teacher type");

        var baseTeacher = new TeacherModelWrapper<T>(model);
        return new CurriculumTeacherModel<T>(
            baseTeacher,
            CurriculumStrategy.EasyToHard);
    }

    private static ITeacherModel<Vector<T>, Vector<T>> CreateSelfTeacher(int outputDimension)
    {
        return new SelfTeacherModel<T>(outputDimension);
    }

    private static ITeacherModel<Vector<T>, Vector<T>> CreateQuantizedTeacher(
        IFullModel<T, Vector<T>, Vector<T>>? model,
        int? outputDimension)
    {
        if (model == null)
            throw new ArgumentException("Model is required for Quantized teacher type");

        var baseTeacher = new TeacherModelWrapper<T>(model);
        return new QuantizedTeacherModel<T>(baseTeacher, quantizationBits: 8);
    }

    private static ITeacherModel<Vector<T>, Vector<T>> CreateDistributedTeacher(
        ITeacherModel<Vector<T>, Vector<T>>[]? workers)
    {
        if (workers == null || workers.Length == 0)
            throw new ArgumentException("Worker models are required for Distributed teacher type");

        return new DistributedTeacherModel<T>(workers, AggregationMode.Average);
    }
}
