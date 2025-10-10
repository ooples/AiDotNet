namespace AiDotNet.Compression.KnowledgeDistillation;

using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// Implements model compression using knowledge distillation techniques.
/// </summary>
/// <remarks>
/// <para>
/// Knowledge distillation compresses models by training a smaller "student" model
/// to mimic the behavior of a larger "teacher" model.
/// </para>
/// <para><b>For Beginners:</b> This creates a smaller model that learns from a larger one.
/// 
/// Knowledge distillation works like a teacher-student relationship:
/// - The large, complex model (teacher) is already trained and accurate
/// - We create a smaller, simpler model (student)
/// - The student learns to mimic the teacher's outputs
/// - The student ends up with similar accuracy but much smaller size
/// 
/// This is particularly effective for very large models, where we can often create
/// student models that are 5-10x smaller while retaining most of the accuracy.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <typeparam name="TModel">The type of model to compress.</typeparam>
/// <typeparam name="TInput">The input type for the model.</typeparam>
/// <typeparam name="TOutput">The output type for the model.</typeparam>
public class DistillationCompressor<T, TModel, TInput, TOutput> :
    ModelCompressorBase<T, TModel, TInput, TOutput>
    where T : unmanaged
    where TModel : class, IFullModel<T, TInput, TOutput>
{
    private readonly DistillationMethod _distillationMethod = default!;
    private readonly bool _useSoftTargets;
    private readonly int _trainingEpochs;
    private readonly double _alpha; // Weight between soft and hard targets

    /// <summary>
    /// Initializes a new instance of the <see cref="DistillationCompressor{TModel, TInput, TOutput}"/> class.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Creates a new distillation compressor with default settings:
    /// - Using vanilla knowledge distillation
    /// - Using soft targets for distillation
    /// - Training for 100 epochs
    /// - Using a balance of 0.5 between soft and hard targets
    /// </para>
    /// <para><b>For Beginners:</b> This creates a distillation compressor with standard settings.
    /// 
    /// The default configuration works well for many models, but you can use other constructors
    /// if you need more control over the distillation process.
    /// </para>
    /// </remarks>
    public DistillationCompressor()
        : this(new ModelCompressionOptions 
        { 
            Technique = CompressionTechnique.KnowledgeDistillation 
        }, DistillationMethod.Vanilla, true, 100, 0.5)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="DistillationCompressor{TModel, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="options">The options for model compression.</param>
    /// <param name="method">The distillation method to use.</param>
    /// <param name="useSoftTargets">Whether to use soft targets for distillation.</param>
    /// <param name="trainingEpochs">The number of epochs to train the student model.</param>
    /// <param name="alpha">Weight between soft and hard targets (0-1).</param>
    /// <remarks>
    /// <para>
    /// Creates a new distillation compressor with the specified settings.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a distillation compressor with custom settings.
    /// 
    /// You can specify:
    /// - Which distillation method to use (vanilla, attention, etc.)
    /// - Whether to use soft targets (teacher's probability distributions)
    /// - How many training epochs to use
    /// - The balance between mimicking the teacher and learning from hard labels
    /// 
    /// These settings let you fine-tune the distillation process for your specific model.
    /// </para>
    /// </remarks>
    public DistillationCompressor(
        ModelCompressionOptions options,
        DistillationMethod method = DistillationMethod.Vanilla,
        bool useSoftTargets = true,
        int trainingEpochs = 100,
        double alpha = 0.5)
        : base(options)
    {
        _distillationMethod = method;
        _useSoftTargets = useSoftTargets;
        _trainingEpochs = trainingEpochs;
        _alpha = alpha;
    }

    /// <summary>
    /// Compresses a model using knowledge distillation.
    /// </summary>
    /// <param name="model">The teacher model to distill knowledge from.</param>
    /// <returns>The student model after distillation.</returns>
    /// <remarks>
    /// <para>
    /// This method trains a smaller student model to mimic the behavior of the
    /// provided teacher model.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a smaller model that mimics the larger one.
    /// 
    /// The process involves:
    /// 1. Creating a smaller student model
    /// 2. Using the teacher model's outputs as "soft targets"
    /// 3. Training the student to match both soft targets and actual labels
    /// 4. Fine-tuning the student model for best performance
    /// 
    /// The result is a much smaller model that approaches the accuracy of the larger one.
    /// </para>
    /// </remarks>
    protected override TModel CompressModel(TModel model)
    {
        // Validate inputs
        if (model == null) throw new ArgumentNullException(nameof(model));
        if (Options.Technique != CompressionTechnique.KnowledgeDistillation)
        {
            throw new ArgumentException(
                $"Expected CompressionTechnique.KnowledgeDistillation but got {Options.Technique}.",
                nameof(Options));
        }

        // Check if the model supports distillation
        if (!(model is IDistillableModel<T, TModel, TInput, TOutput> distillableModel))
        {
            throw new InvalidOperationException(
                $"Model of type {model.GetType().Name} does not implement IDistillableModel. " +
                "Knowledge distillation requires the teacher model to implement IDistillableModel.");
        }

        // Get unlabeled data for distillation
        var unlabeledData = GetUnlabeledData();
        if (unlabeledData == null || unlabeledData.Length == 0)
        {
            throw new InvalidOperationException(
                "No unlabeled data available for distillation. " +
                "Knowledge distillation requires unlabeled data to transfer knowledge.");
        }

        // Get labeled data for distillation (if using hard targets)
        var labeledData = GetLabeledData();

        // Create the student model architecture
        var studentArchitecture = CreateStudentArchitecture(model, Options.DistillationStudentSize);

        // Create the student model
        var studentModel = distillableModel.CreateStudentModel(studentArchitecture);

        // Set the distillation temperature
        double temperature = Options.DistillationTemperature;
        if (temperature <= 0)
        {
            temperature = CompressionDefaults.DefaultDistillationTemperature;
        }

        // Train the student model using knowledge distillation
        studentModel = TrainStudentModel(
            teacherModel: model,
            studentModel: studentModel,
            unlabeledData: unlabeledData,
            labeledData: labeledData,
            temperature: temperature,
            epochs: _trainingEpochs,
            alpha: _alpha);

        return studentModel;
    }

    /// <summary>
    /// Serializes a distilled model to a file.
    /// </summary>
    /// <param name="model">The distilled model to serialize.</param>
    /// <param name="filePath">The path where the serialized model should be saved.</param>
    /// <remarks>
    /// <para>
    /// This method serializes the distilled student model to a file.
    /// </para>
    /// <para><b>For Beginners:</b> This saves the distilled model to a file.
    /// 
    /// Unlike other compression techniques that need special serialization,
    /// distilled models are complete models in their own right, so they can
    /// use standard serialization methods. This method ensures the distillation
    /// metadata is saved along with the model.
    /// </para>
    /// </remarks>
    public override void SerializeCompressedModel(TModel model, string filePath)
    {
        // Ensure the directory exists
        var directory = Path.GetDirectoryName(filePath);
        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        // Check if model is a distilled model
        if (!(model is IDistilledModel<T, TInput, TOutput> distilledModel))
        {
            throw new ArgumentException(
                $"Model of type {model.GetType().Name} is not a distilled model. " +
                "Use the Compress method before serializing.", nameof(model));
        }

        using (var fileStream = new FileStream(filePath, FileMode.Create))
        using (var writer = new BinaryWriter(fileStream))
        {
            // Write distillation metadata
            writer.Write((int)_distillationMethod);
            writer.Write(_useSoftTargets);
            writer.Write(distilledModel.DistillationTemperature);
            writer.Write(distilledModel.CompressionRatio);

            // Write model-specific data
            distilledModel.SerializeDistilled(writer);
        }
    }

    /// <summary>
    /// Deserializes a distilled model from a file.
    /// </summary>
    /// <param name="filePath">The path where the serialized model is stored.</param>
    /// <returns>The deserialized distilled model.</returns>
    /// <remarks>
    /// <para>
    /// This method deserializes a distilled model from a file that was previously saved
    /// using the SerializeCompressedModel method.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a distilled model from a file.
    /// 
    /// The deserialization process:
    /// 1. Reads the distillation metadata
    /// 2. Creates and returns the deserialized student model
    /// </para>
    /// </remarks>
    public override TModel DeserializeCompressedModel(string filePath)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"Distilled model file not found: {filePath}");
        }

        using (var fileStream = new FileStream(filePath, FileMode.Open))
        using (var reader = new BinaryReader(fileStream))
        {
            // Read distillation metadata
            var method = (DistillationMethod)reader.ReadInt32();
            bool useSoftTargets = reader.ReadBoolean();
            double temperature = reader.ReadDouble();
            double compressionRatio = reader.ReadDouble();

            // Create and return the deserialized model
            return DeserializeDistilledModelInternal(reader, method, useSoftTargets, temperature, compressionRatio);
        }
    }

    /// <summary>
    /// Populates additional metrics specific to knowledge distillation.
    /// </summary>
    /// <param name="metrics">The dictionary to populate with additional metrics.</param>
    /// <param name="originalModel">The original teacher model.</param>
    /// <param name="compressedModel">The distilled student model.</param>
    /// <remarks>
    /// <para>
    /// This method adds knowledge distillation-specific metrics to the compression results.
    /// </para>
    /// <para><b>For Beginners:</b> This adds distillation-specific details to the results.
    /// 
    /// Specifically, it adds:
    /// - The distillation method used
    /// - The temperature parameter used for soft targets
    /// - The student model architecture details
    /// - The parameter reduction achieved
    /// </para>
    /// </remarks>
    protected override void PopulateAdditionalMetrics(
        Dictionary<string, object> metrics,
        TModel originalModel,
        TModel compressedModel)
    {
        // Add distillation-specific metrics
        if (compressedModel is IDistilledModel<T, TInput, TOutput> distilledModel)
        {
            metrics["DistillationMethod"] = _distillationMethod.ToString();
            metrics["DistillationTemperature"] = distilledModel.DistillationTemperature;
            metrics["UsedSoftTargets"] = _useSoftTargets;
            metrics["TeacherModelParamCount"] = GetParameterCount(originalModel);
            metrics["StudentModelParamCount"] = GetParameterCount(compressedModel);
            metrics["ParameterReductionRatio"] = distilledModel.CompressionRatio;

            // Add structural complexity metrics if available
            var structuralMetrics = distilledModel.StructuralMetrics;
            if (structuralMetrics != null)
            {
                foreach (var metric in structuralMetrics)
                {
                    metrics[metric.Key] = metric.Value ?? "N/A";
                }
            }
        }
    }

    /// <summary>
    /// Gets unlabeled data for distillation.
    /// </summary>
    /// <returns>An array of unlabeled input data.</returns>
    /// <remarks>
    /// <para>
    /// This method should be overridden by derived classes to provide unlabeled data
    /// for the distillation process.
    /// </para>
    /// <para><b>For Beginners:</b> This gets data without labels for the teacher to generate predictions.
    /// 
    /// Knowledge distillation needs data for the teacher to make predictions on:
    /// - This data doesn't need labels
    /// - The teacher's predictions become the "soft targets"
    /// - The student learns from these soft targets
    /// 
    /// This method should be customized to provide appropriate data for your domain.
    /// </para>
    /// </remarks>
    protected virtual TInput[] GetUnlabeledData()
    {
        // This should be implemented in a derived class to provide domain-specific data
        // For now, we return a placeholder value
        throw new NotImplementedException(
            "GetUnlabeledData must be implemented in a derived class. " +
            "Knowledge distillation requires unlabeled data to transfer knowledge.");
    }

    /// <summary>
    /// Gets labeled data for distillation.
    /// </summary>
    /// <returns>Arrays of input data and corresponding output labels.</returns>
    /// <remarks>
    /// <para>
    /// This method should be overridden by derived classes to provide labeled data
    /// for the distillation process when using a mix of soft and hard targets.
    /// </para>
    /// <para><b>For Beginners:</b> This gets data with true labels for additional supervision.
    /// 
    /// When using both soft and hard targets:
    /// - The soft targets come from the teacher model
    /// - The hard targets come from actual labeled data
    /// - The student learns from a weighted combination of both
    /// 
    /// This method should be customized to provide appropriate labeled data.
    /// </para>
    /// </remarks>
    protected virtual (TInput[] inputs, TOutput[] outputs)? GetLabeledData()
    {
        // This can be null if only using soft targets
        // Otherwise, should be implemented in a derived class
        return null;
    }

    /// <summary>
    /// Creates a student model architecture based on the teacher model.
    /// </summary>
    /// <param name="teacherModel">The teacher model.</param>
    /// <param name="studentSizeRatio">The size ratio for the student model (0-1).</param>
    /// <returns>The student model architecture.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a smaller architecture for the student model based on the
    /// teacher model's architecture and the specified size ratio.
    /// </para>
    /// <para><b>For Beginners:</b> This designs a smaller version of the original model.
    /// 
    /// The student architecture:
    /// - Is similar to the teacher but with fewer parameters
    /// - Typically has fewer layers or smaller layer dimensions
    /// - Is designed to be proportional to the specified size ratio
    /// 
    /// This method should be customized for your specific model type.
    /// </para>
    /// </remarks>
    protected virtual object CreateStudentArchitecture(TModel teacherModel, double studentSizeRatio)
    {
        // This should be implemented by derived classes to create an appropriate student architecture
        // The implementation depends on the specific model type

        if (teacherModel is IDistillableModel<T, TModel, TInput, TOutput> distillableModel)
        {
            return distillableModel.CreateStudentArchitecture(studentSizeRatio);
        }

        throw new NotSupportedException(
            $"Student architecture creation not supported for model type {teacherModel.GetType().Name}. " +
            "The teacher model must implement IDistillableModel<T, TModel, TInput, TOutput>.");
    }

    /// <summary>
    /// Trains the student model using knowledge distillation.
    /// </summary>
    /// <param name="teacherModel">The teacher model.</param>
    /// <param name="studentModel">The student model to train.</param>
    /// <param name="unlabeledData">Unlabeled data for distillation.</param>
    /// <param name="labeledData">Optional labeled data for distillation.</param>
    /// <param name="temperature">The temperature for softening logits.</param>
    /// <param name="epochs">The number of training epochs.</param>
    /// <param name="alpha">Weight between soft and hard targets (0-1).</param>
    /// <returns>The trained student model.</returns>
    /// <remarks>
    /// <para>
    /// This method trains the student model to mimic the teacher model using
    /// knowledge distillation.
    /// </para>
    /// <para><b>For Beginners:</b> This trains the smaller model to mimic the larger one.
    /// 
    /// The training process:
    /// 1. The teacher model generates predictions ("soft targets") on unlabeled data
    /// 2. These soft targets are adjusted using the temperature parameter
    /// 3. The student model learns to match these soft targets
    /// 4. If labeled data is available, the student also learns from true labels
    /// 5. The alpha parameter balances these two learning objectives
    /// 
    /// After training, the student model can approach the teacher's accuracy
    /// despite having far fewer parameters.
    /// </para>
    /// </remarks>
    protected virtual TModel TrainStudentModel(
        TModel teacherModel,
        TModel studentModel,
        TInput[] unlabeledData,
        (TInput[] inputs, TOutput[] outputs)? labeledData,
        double temperature,
        int epochs,
        double alpha)
    {
        // This should be implemented by derived classes for model-specific training
        // The implementation depends on the specific model type and training framework

        if (teacherModel is IDistillableModel<T, TModel, TInput, TOutput> distillableTeacher &&
            studentModel is IDistillableStudent<T, TInput, TOutput> distillableStudent)
        {
            // Set up distillation parameters
            var distillationParams = new DistillationParameters
            {
                Temperature = temperature,
                Alpha = alpha,
                UseSoftTargets = _useSoftTargets,
                TrainingEpochs = epochs,
                Method = _distillationMethod
            };

            // Train the student using distillation
            distillableTeacher.DistillKnowledgeToStudent(
                distillableStudent,
                unlabeledData,
                labeledData,
                distillationParams);

            return studentModel;
        }

        throw new NotSupportedException(
            "Knowledge distillation training not supported for these model types. " +
            "The teacher model must implement IDistillableModel<T, TModel, TInput, TOutput> " +
            "and the student model must implement IDistillableStudent<T, TInput, TOutput>.");
    }

    /// <summary>
    /// Deserializes a distilled model from a binary reader.
    /// </summary>
    /// <param name="reader">The binary reader containing the serialized model.</param>
    /// <param name="method">The distillation method used.</param>
    /// <param name="useSoftTargets">Whether soft targets were used for distillation.</param>
    /// <param name="temperature">The temperature used for distillation.</param>
    /// <param name="compressionRatio">The achieved compression ratio.</param>
    /// <returns>The deserialized distilled model.</returns>
    /// <remarks>
    /// <para>
    /// This method reconstructs a distilled model from its serialized form.
    /// </para>
    /// <para><b>For Beginners:</b> This rebuilds a distilled model from saved data.
    /// 
    /// The deserialization process:
    /// 1. Reads the serialized student model data
    /// 2. Reconstructs the model with its distillation metadata
    /// 
    /// This is a model-specific process that depends on the type of model being used.
    /// </para>
    /// </remarks>
    protected virtual TModel DeserializeDistilledModelInternal(
        BinaryReader reader,
        DistillationMethod method,
        bool useSoftTargets,
        double temperature,
        double compressionRatio)
    {
        // This implementation depends on the specific model type.
        // For demonstration, we'll assume there's a factory method that can
        // create a model from the serialized data.

        // In a real implementation, this would use model-specific deserialization.
        var factory = DistilledModelFactoryRegistry.GetFactory<T, TModel, TInput, TOutput>();
        if (factory != null)
        {
            return factory.DeserializeDistilledModel(
                reader, method, useSoftTargets, temperature, compressionRatio);
        }

        throw new NotSupportedException(
            $"Distilled model deserialization not supported for model type TModel. " +
            "Register a factory for this model type with DistilledModelFactoryRegistry.");
    }

    /// <summary>
    /// Gets the parameter count for a model.
    /// </summary>
    /// <param name="model">The model to count parameters for.</param>
    /// <returns>The number of parameters in the model.</returns>
    /// <remarks>
    /// <para>
    /// This method counts the total number of parameters (weights, biases, etc.) in a model.
    /// </para>
    /// <para><b>For Beginners:</b> This counts how many numbers are stored in the model.
    /// 
    /// Neural networks contain many parameters:
    /// - Weight matrices connecting layers
    /// - Bias vectors for each layer
    /// - Other learnable parameters
    /// 
    /// This count helps measure the size reduction achieved by distillation.
    /// </para>
    /// </remarks>
    protected virtual long GetParameterCount(TModel model)
    {
        // Implementation for IFullModel
        return model.GetParameters().Length;
    }

    /// <summary>
    /// Gets the compression technique used by this compressor.
    /// </summary>
    /// <returns>CompressionTechnique.KnowledgeDistillation</returns>
    /// <remarks>
    /// <para>
    /// This method returns CompressionTechnique.KnowledgeDistillation to indicate that this
    /// compressor implements knowledge distillation based compression.
    /// </para>
    /// <para><b>For Beginners:</b> This identifies the compressor as using knowledge distillation.
    /// 
    /// This information is used in compression results to indicate which technique was applied.
    /// </para>
    /// </remarks>
    protected override CompressionTechnique GetCompressionTechnique()
    {
        return CompressionTechnique.KnowledgeDistillation;
    }

    /// <summary>
    /// Creates a new compressor with the specified options.
    /// </summary>
    /// <param name="options">The compression options to use.</param>
    /// <returns>A new compressor instance with the specified options.</returns>
    protected override IModelCompressor<TModel, TInput, TOutput> CreateCompressorWithOptions(ModelCompressionOptions options)
    {
        // Create a new distillation compressor with the same configuration but new options
        return new DistillationCompressor<T, TModel, TInput, TOutput>(
            options,
            _distillationMethod,
            _useSoftTargets,
            _trainingEpochs,
            _alpha);
    }

    // SerializeModelToStream is now handled by the base class
    
    /// <summary>
    /// Deserializes a model from a file.
    /// </summary>
    /// <param name="filePath">The file path from which to load the model.</param>
    /// <returns>The deserialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method deserializes a distilled model from a file.
    /// </para>
    /// <para><b>For Beginners:</b> This loads a saved model from a file.
    /// 
    /// This implementation simply calls DeserializeCompressedModel which already
    /// contains the logic for deserializing distilled models from files.
    /// </para>
    /// </remarks>
    protected override TModel DeserializeModelFromFile(string filePath)
    {
        return DeserializeCompressedModel(filePath);
    }
    
    /// <summary>
    /// Runs inference with the model on a single input.
    /// </summary>
    /// <param name="model">The model to use.</param>
    /// <param name="input">The input to process.</param>
    /// <returns>The model's output for the input.</returns>
    protected override TOutput RunInference(TModel model, TInput input)
    {
        return model.Predict(input);
    }

    /// <summary>
    /// Measures the accuracy of a model on the provided test data.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="testInputs">The test inputs.</param>
    /// <param name="expectedOutputs">The expected outputs for the test inputs.</param>
    /// <returns>A value representing the model's accuracy (higher is better).</returns>
    protected override double MeasureAccuracy(TModel model, TInput[] testInputs, TOutput[] expectedOutputs)
    {
        double totalCorrect = 0;
        
        for (int i = 0; i < testInputs.Length; i++)
        {
            var predicted = model.Predict(testInputs[i]);
            
            // Accuracy calculation would depend on the model type and output format
            // For this example, we'll use a dummy check just to fulfill the contract
            if (predicted != null && predicted.Equals(expectedOutputs[i]))
            {
                totalCorrect++;
            }
        }
        
        return totalCorrect / testInputs.Length;
    }

}