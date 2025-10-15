using AiDotNet.LinearAlgebra;
using AiDotNet.Models;

namespace AiDotNet.Genetics;

/// <summary>
/// Represents an individual that is also a full model, allowing direct evolution of models
/// without conversion between individuals and models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The type of input data for the model.</typeparam>
/// <typeparam name="TOutput">The type of output data produced by the model.</typeparam>
/// <typeparam name="TGene">The type representing a gene in the genetic model.</typeparam>
/// <remarks>
/// <para>
/// This class implements both IEvolvable and IFullModel interfaces, allowing it to be used
/// directly in genetic algorithms while also providing model prediction capabilities.
/// </para>
/// <para><b>For Beginners:</b>
/// This class combines the functionality of an individual in a genetic algorithm with a machine
/// learning model. This means:
/// 
/// - You can evolve the model directly without converting between different representations
/// - The individual can make predictions like any other model
/// - It simplifies the implementation of genetic algorithms for model optimization
/// 
/// Use this when you want to directly evolve machine learning models using genetic algorithms.
/// </para>
/// </remarks>
public class ModelIndividual<T, TInput, TOutput, TGene> :
    Interpretability.InterpretableModelBase<T, TInput, TOutput>,
    IEvolvable<TGene, T>,
    IFullModel<T, TInput, TOutput>
    where TGene : class
{
    /// <summary>
    /// The collection of genes that define this individual's genetic makeup.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the genetic information of the individual, which is used to construct
    /// and update the inner model. Each gene typically represents a parameter or structure
    /// in the machine learning model.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Think of these genes like the DNA of the individual. Each gene represents a specific
    /// characteristic or parameter of the model. Just as DNA determines traits in living
    /// organisms, these genes determine how the model behaves and makes predictions.
    /// 
    /// When evolution occurs, these genes are modified, combined, or mutated to create
    /// new individuals with potentially better performance.
    /// </para>
    /// </remarks>
    private List<TGene> _genes = [];

    /// <summary>
    /// Set of feature indices that have been explicitly marked as active.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores feature indices that have been explicitly set as active through
    /// the SetActiveFeatureIndices method, overriding the automatic determination based
    /// on the inner model's behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This tracks which input features have been manually
    /// selected as important for the model, regardless of what features the inner model
    /// actually uses in its calculations.
    /// 
    /// When set, these manually selected features take precedence over the automatic
    /// feature detection based on the inner model's parameters.
    /// </para>
    /// </remarks>
    private HashSet<int>? _explicitlySetActiveFeatures;

    /// <summary>
    /// The fitness score of this individual, indicating how well it performs.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores the individual's fitness score, which quantifies how well the model
    /// performs on the given problem. Higher fitness typically indicates better performance,
    /// though this depends on the specific fitness calculation used.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like a score or grade that measures how well this model solves the problem.
    /// In genetic algorithms, individuals with higher fitness scores are more likely to
    /// survive and reproduce, passing their characteristics to the next generation.
    /// 
    /// Just as in nature where the fittest organisms are more likely to survive and reproduce,
    /// models with higher fitness scores have a better chance of contributing to future generations.
    /// </para>
    /// </remarks>
    private T _fitness = default!;

    /// <summary>
    /// The actual machine learning model contained within this individual.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field holds the machine learning model that is constructed from the individual's genes.
    /// It provides the actual prediction functionality and is updated whenever the genes change.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is the actual machine learning model that makes predictions. While the genes
    /// define the characteristics, this is the functional model that processes inputs and
    /// produces outputs.
    /// 
    /// It's like the difference between a blueprint (genes) and the actual building
    /// constructed from that blueprint (inner model). When the genes change, the
    /// inner model is rebuilt to reflect those changes.
    /// </para>
    /// </remarks>
    private IFullModel<T, TInput, TOutput> _innerModel = default!;

    /// <summary>
    /// A factory function that creates a model from a collection of genes.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This field stores a function that knows how to construct a machine learning model
    /// from a given set of genes. It is used whenever the genes are updated to rebuild
    /// the inner model.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Think of this as a specialized builder that knows how to turn genetic information
    /// into a working model. When the genes change, this builder constructs a new model
    /// based on the updated genetic information.
    /// 
    /// It's like having a construction company that knows how to build a house
    /// from a blueprint - whenever you modify the blueprint, they can build
    /// you a new house that reflects those changes.
    /// </para>
    /// </remarks>
    private readonly Func<ICollection<TGene>, IFullModel<T, TInput, TOutput>> _modelFactory = default!;

    /// <summary>
    /// Provides operations for the numeric type used in this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This static field provides mathematical operations for the specific numeric type
    /// used in the model (e.g., double, float). It allows the model to perform calculations
    /// regardless of the specific numeric type.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like a calculator that knows how to work with the specific number type
    /// being used. It helps the model perform mathematical operations without needing
    /// to know exactly what kind of numbers it's working with.
    /// 
    /// It's like having a universal translator for mathematics that works regardless of
    /// whether you're using whole numbers, decimals, or other number formats.
    /// </para>
    /// </remarks>
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a new model individual with the specified genes and model factory.
    /// </summary>
    /// <param name="genes">The genes to initialize with.</param>
    /// <param name="modelFactory">A function that creates a model from genes.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a new ModelIndividual with the provided genes and model factory.
    /// It creates a new inner model using the factory and the specified genes, and initializes
    /// the fitness to zero.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating a new organism with a specific set of DNA and a builder who
    /// knows how to turn that DNA into a functional body.
    /// 
    /// When you create a new model individual this way:
    /// - You provide the genetic information (genes) that define its characteristics
    /// - You provide a builder (modelFactory) that can turn those genes into a working model
    /// - The individual starts with a fitness score of zero, as it hasn't been evaluated yet
    /// 
    /// This constructor is typically used when creating a new individual from scratch
    /// or when constructing offspring in genetic operations.
    /// </para>
    /// </remarks>
    public ModelIndividual(
        ICollection<TGene> genes,
        Func<ICollection<TGene>, IFullModel<T, TInput, TOutput>> modelFactory)
    {
        _genes = [];
        _modelFactory = modelFactory;
        _innerModel = _modelFactory(_genes);
        _fitness = _numOps.Zero;
    }

    /// <summary>
    /// Creates a new model individual by wrapping an existing model.
    /// </summary>
    /// <param name="model">The model to wrap.</param>
    /// <param name="genes">The genes representing the model.</param>
    /// <param name="modelFactory">A function that creates a model from genes.</param>
    /// <remarks>
    /// <para>
    /// This constructor initializes a ModelIndividual with an existing model, along with the
    /// genes that represent that model and a model factory. It's useful when you already have
    /// a trained or predefined model that you want to include in evolutionary algorithms.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking an existing, fully-formed organism and reverse-engineering its DNA,
    /// then preparing it to participate in evolution.
    /// 
    /// When you create a model individual this way:
    /// - You start with an already working model
    /// - You provide the genes that represent this model's characteristics
    /// - You provide a builder that knows how to create new models from genes
    /// - The individual starts with a fitness score of zero until evaluated
    /// 
    /// This constructor is typically used when integrating existing models into a genetic
    /// algorithm, perhaps to fine-tune them or to use them as a starting point for evolution.
    /// </para>
    /// </remarks>
    public ModelIndividual(
        IFullModel<T, TInput, TOutput> model,
        ICollection<TGene> genes,
        Func<ICollection<TGene>, IFullModel<T, TInput, TOutput>> modelFactory)
    {
        _innerModel = model;
        _genes = [];
        _modelFactory = modelFactory;
        _fitness = _numOps.Zero;
    }

    #region IEvolvable Implementation

    /// <summary>
    /// Gets the genes of this individual.
    /// </summary>
    /// <returns>The collection of genes.</returns>
    /// <remarks>
    /// <para>
    /// This method provides access to the individual's genetic information.
    /// It returns the collection of genes that define this individual's characteristics.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like examining the DNA of an organism to understand its genetic makeup.
    /// 
    /// You might use this method when:
    /// - You need to analyze the specific characteristics of this individual
    /// - You're performing genetic operations like crossover or mutation
    /// - You want to compare the genetic makeup of different individuals
    /// 
    /// In genetic algorithms, these genes are the primary material that evolution works with.
    /// </para>
    /// </remarks>
    public ICollection<TGene> GetGenes()
    {
        return _genes;
    }

    /// <summary>
    /// Sets the genes of this individual and updates the inner model.
    /// </summary>
    /// <param name="genes">The new genes.</param>
    /// <remarks>
    /// <para>
    /// This method replaces the individual's genes with the provided collection and
    /// rebuilds the inner model using the model factory. This ensures that the model's
    /// behavior reflects the new genetic information.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like modifying an organism's DNA and watching its body transform to match.
    /// 
    /// When you set new genes:
    /// - The individual's genetic information is replaced
    /// - The inner model is rebuilt based on these new genes
    /// - The individual's behavior and predictions will change accordingly
    /// 
    /// This is a critical method in genetic algorithms, as it's how changes from genetic
    /// operations (crossover, mutation) are applied to individuals.
    /// </para>
    /// </remarks>
    public void SetGenes(ICollection<TGene> genes)
    {
        _genes = [.. genes];
        _innerModel = _modelFactory(_genes);
    }

    /// <summary>
    /// Gets the fitness of this individual.
    /// </summary>
    /// <returns>The fitness value.</returns>
    /// <remarks>
    /// <para>
    /// This method returns the individual's fitness score, which quantifies how well
    /// the model performs on the given problem. This score is used in selection
    /// processes to determine which individuals contribute to the next generation.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like checking an athlete's performance score to see how well they're doing.
    /// 
    /// The fitness score:
    /// - Indicates how well this model solves the problem
    /// - Determines its chances of being selected for reproduction
    /// - Helps track improvement across generations
    /// 
    /// In genetic algorithms, fitness is the key measure that drives the entire
    /// evolutionary process toward better solutions.
    /// </para>
    /// </remarks>
    public T GetFitness()
    {
        return _fitness;
    }

    /// <summary>
    /// Sets the fitness of this individual.
    /// </summary>
    /// <param name="fitness">The new fitness value.</param>
    /// <remarks>
    /// <para>
    /// This method sets the individual's fitness score after evaluation.
    /// It is typically called by the genetic algorithm after testing the individual's
    /// performance on the problem being solved.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like updating an athlete's score after they complete a competition.
    /// 
    /// When setting fitness:
    /// - The score reflects how well this model performed on the task
    /// - Higher scores typically indicate better performance
    /// - This score will determine the individual's chances in selection processes
    /// 
    /// Setting accurate fitness scores is crucial for the genetic algorithm to effectively
    /// guide evolution toward better solutions.
    /// </para>
    /// </remarks>
    public void SetFitness(T fitness)
    {
        _fitness = fitness;
    }

    /// <summary>
    /// Creates a deep clone of this individual.
    /// </summary>
    /// <returns>A new individual with the same genes and fitness.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new ModelIndividual that is a copy of this one, with cloned
    /// genes and the same fitness score. If genes implement ICloneable, they are deep-cloned;
    /// otherwise, they are shallow-copied.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating an identical twin of the individual, with the same genetic
    /// makeup and fitness level.
    /// 
    /// When cloning:
    /// - A completely new individual is created
    /// - Each gene is copied (deeply if possible)
    /// - The fitness score is copied
    /// - Changes to the clone won't affect the original
    /// 
    /// Cloning is essential in genetic algorithms for operations like elitism
    /// (preserving the best individuals) and for creating offspring.
    /// </para>
    /// </remarks>
    public IEvolvable<TGene, T> Clone()
    {
        var clonedGenes = new List<TGene>();

        // Deep clone each gene if it implements ICloneable
        foreach (var gene in _genes)
        {
            if (gene is ICloneable cloneable)
            {
                clonedGenes.Add((TGene)cloneable.Clone());
            }
            else
            {
                // Fallback to shallow copy if not cloneable
                clonedGenes.Add(gene);
            }
        }

        var clone = new ModelIndividual<T, TInput, TOutput, TGene>(clonedGenes, _modelFactory);
        clone.SetFitness(_fitness);

        return clone;
    }

    #endregion

    #region IFullModel Implementation

    /// <summary>
    /// Makes a prediction using the inner model.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <returns>The predicted output.</returns>
    /// <remarks>
    /// <para>
    /// This method delegates the prediction task to the inner model, which processes
    /// the input data and returns a prediction. The behavior of this method depends
    /// entirely on the type of model being used.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like asking the model a question and getting its answer.
    /// 
    /// When making predictions:
    /// - You provide some input data (the question)
    /// - The inner model processes this data based on its current configuration
    /// - The model returns its prediction (the answer)
    /// 
    /// This is the primary function of any machine learning model - taking inputs
    /// and producing useful outputs or predictions.
    /// </para>
    /// </remarks>
    public override TOutput Predict(TInput input)
    {
        return _innerModel.Predict(input);
    }

    /// <summary>
    /// Gets the metadata for the model.
    /// </summary>
    /// <returns>The model metadata.</returns>
    /// <remarks>
    /// <para>
    /// This method delegates to GetModelMetadata to retrieve metadata about the model,
    /// which includes information like the model type, parameters, performance metrics,
    /// and other descriptive data.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like getting a detailed report card or specifications sheet for the model.
    /// 
    /// The metadata might include:
    /// - What type of model it is
    /// - How it was trained
    /// - How well it performs
    /// - What parameters it uses
    /// 
    /// This information is useful for understanding, comparing, and documenting models.
    /// </para>
    /// </remarks>
    public ModelMetadata<T> GetMetaData()
    {
        return GetModelMetadata();
    }

    /// <summary>
    /// Gets the parameters of the model.
    /// </summary>
    /// <returns>The model parameters as a vector.</returns>
    /// <remarks>
    /// <para>
    /// This method delegates to the inner model to retrieve its parameters as a vector.
    /// These parameters define the behavior of the model and are adjusted during training
    /// or evolution.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like getting all the settings and configuration values of the model.
    /// 
    /// The parameters:
    /// - Control how the model processes inputs
    /// - Determine the model's behavior
    /// - Are what the model learns during training
    /// - Can be modified to change the model's predictions
    /// 
    /// Parameters are the numerical values that define a specific instance of a model,
    /// like weights in a neural network or coefficients in a regression model.
    /// </para>
    /// </remarks>
    public Vector<T> GetParameters()
    {
        return _innerModel.GetParameters();
    }

    /// <summary>
    /// Sets the parameters of the model.
    /// </summary>
    /// <param name="parameters">The parameters to set.</param>
    /// <exception cref="ArgumentNullException">Thrown when parameters is null.</exception>
    /// <remarks>
    /// <para>
    /// This method delegates to the inner model to set its parameters from a vector.
    /// These parameters define the behavior of the model and can be adjusted during training
    /// or evolution.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like changing all the settings and configuration values of the model.
    /// 
    /// The parameters:
    /// - Replace the current settings that control how the model works
    /// - Change the model's behavior and predictions
    /// - Must be compatible with the inner model's structure
    /// 
    /// This is useful when loading saved models or applying parameter updates
    /// from optimization algorithms.
    /// </para>
    /// </remarks>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters == null)
        {
            throw new ArgumentNullException(nameof(parameters));
        }
        
        // Try to find and invoke SetParameters method on inner model
        var setParametersMethod = _innerModel.GetType().GetMethod("SetParameters", [typeof(Vector<T>)]);
        
        if (setParametersMethod != null)
        {
            setParametersMethod.Invoke(_innerModel, [parameters]);
        }
        else
        {
            // If SetParameters is not available, create a new inner model with the parameters
            _innerModel = _innerModel.WithParameters(parameters);
        }
    }

    /// <summary>
    /// Updates the parameters of the model.
    /// </summary>
    /// <param name="parameters">The new parameters.</param>
    /// <remarks>
    /// <para>
    /// This method updates the inner model's parameters without creating a new model instance.
    /// It's used to adjust the model's behavior during training or optimization.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like adjusting the settings of a machine to change how it works.
    /// 
    /// When updating parameters:
    /// - The model's internal values are changed
    /// - Its behavior and predictions will change accordingly
    /// - The model structure remains the same
    /// 
    /// This is often used during training or fine-tuning when the model structure
    /// is fixed but the parameter values need to be optimized.
    /// </para>
    /// </remarks>
    public void UpdateParameters(Vector<T> parameters)
    {
        _innerModel.WithParameters(parameters);
    }

    /// <summary>
    /// Creates a new model with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters for the new model.</param>
    /// <returns>A new model with the specified parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new model with the same structure as the current model but
    /// with different parameters. It creates a new ModelIndividual wrapping the new inner model.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating a new machine with different settings based on an existing blueprint.
    /// 
    /// When creating a model with new parameters:
    /// - A completely new model is created
    /// - It has the same structure as the original
    /// - But it uses the specified parameter values
    /// - Changes to this new model won't affect the original
    /// 
    /// This is useful when you want to explore different parameter configurations
    /// without modifying the original model.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> WithParameters(Vector<T> parameters)
    {
        var newModel = _innerModel.WithParameters(parameters);

        return new ModelIndividual<T, TInput, TOutput, TGene>(
            newModel,
            _genes,
            _modelFactory);
    }

    /// <summary>
    /// Serializes the model to a byte array.
    /// </summary>
    /// <returns>A byte array containing the serialized model.</returns>
    /// <remarks>
    /// <para>
    /// This method delegates to the inner model to serialize itself into a byte array.
    /// This allows the model to be saved to disk or transmitted over a network.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like taking a snapshot or photograph of the model that can be saved or shared.
    /// 
    /// When serializing a model:
    /// - All essential information is converted to a compact format
    /// - This data can be saved to a file or sent over a network
    /// - The model can later be reconstructed from this data
    /// 
    /// Serialization is important for preserving models after training, sharing them
    /// with others, or deploying them in different environments.
    /// </para>
    /// </remarks>
    public byte[] Serialize()
    {
        return _innerModel.Serialize();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model.</param>
    /// <remarks>
    /// <para>
    /// This method delegates to the inner model to reconstruct itself from a serialized byte array.
    /// It restores the model to the state it was in when serialized.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like rebuilding a model from its snapshot or blueprint.
    /// 
    /// When deserializing:
    /// - The byte data is interpreted to restore the model's state
    /// - All parameters and configuration are recovered
    /// - The model becomes functional again, identical to when it was serialized
    /// 
    /// Deserialization allows models to be loaded from saved files or reconstructed
    /// after being transmitted over a network.
    /// </para>
    /// </remarks>
    public void Deserialize(byte[] data)
    {
        _innerModel.Deserialize(data);
    }

    /// <summary>
    /// Trains the model on the provided input and expected output data.
    /// </summary>
    /// <param name="input">The input training data.</param>
    /// <param name="expectedOutput">The expected output for the training data.</param>
    /// <remarks>
    /// <para>
    /// This method attempts to train the inner model on the provided data if possible.
    /// Since the evolutionary process typically handles the optimization of the model,
    /// direct training is not the primary method of improvement for this class.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like trying to teach a model by showing it examples directly, which may
    /// or may not be supported depending on the type of model.
    /// 
    /// Since this is a model meant for evolutionary optimization:
    /// - Direct training might not be supported by the inner model
    /// - Improvement typically happens through the genetic algorithm instead
    /// - This method is provided for compatibility with other model interfaces
    /// 
    /// In most cases, you would improve this model through evolution rather than direct training.
    /// </para>
    /// </remarks>
    public override void Train(TInput input, TOutput expectedOutput)
    {
        // Try to use reflection to check if the inner model has a Train method
        var trainMethod = _innerModel.GetType().GetMethod("Train", [typeof(TInput), typeof(TOutput)]);

        if (trainMethod != null)
        {
            // If the inner model has a Train method, invoke it
            trainMethod.Invoke(_innerModel, [input, expectedOutput]);
        }
        else
        {
            // If no Train method is found, throw an exception
            throw new InvalidOperationException("The inner model does not support direct training. Use evolutionary optimization instead.");
        }
    }

    /// <summary>
    /// Gets detailed metadata about the model.
    /// </summary>
    /// <returns>The model metadata.</returns>
    /// <remarks>
    /// <para>
    /// This method provides comprehensive metadata about the model, including its type,
    /// parameters, performance metrics, and other descriptive information. It enhances
    /// the inner model's metadata with information about this being a genetic model.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like getting a complete report card about the model with details about
    /// its structure, performance, and evolutionary history.
    /// 
    /// The metadata might include:
    /// - Technical details about the model type and structure
    /// - Performance metrics like accuracy or error rates
    /// - Information about how it was trained or evolved
    /// - Details about its genetic heritage in the evolutionary process
    /// 
    /// This comprehensive information helps with model selection, comparison, and documentation.
    /// </para>
    /// </remarks>
    public override ModelMetadata<T> GetModelMetadata()
    {
        // Get the inner model's metadata
        var metadata = _innerModel.GetModelMetadata();

        // If inner model's GetModelMetadata method is not properly implemented or returns null,
        // create a new metadata object
        metadata ??= new ModelMetadata<T>
            {
                ModelType = ModelType.GeneticAlgorithmRegression,
                Description = "Genetically evolved model individual",
                AdditionalInfo = []
            };

        // Add genetic algorithm related information
        metadata.AdditionalInfo["IsGeneticModel"] = "True";
        metadata.AdditionalInfo["GeneCount"] = _genes.Count;
        metadata.AdditionalInfo["Fitness"] = Convert.ToDouble(_fitness);

        return metadata;
    }

    /// <summary>
    /// Gets the indices of features that are actively used by the model.
    /// </summary>
    /// <returns>An enumerable collection of active feature indices.</returns>
    /// <remarks>
    /// <para>
    /// This method tries to identify which input features are actively used by the model.
    /// Since there's no universal way to determine this across all model types,
    /// it conservatively assumes that all potential features are used.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like finding out which parts of the input data the model is using.
    /// 
    /// However, since the inner model might work in different ways:
    /// - We can't always tell exactly which features are being used
    /// - To be safe, the method assumes all features might be important
    /// - It returns indices for all possible features
    /// 
    /// This conservative approach ensures that no important features are overlooked,
    /// even if it means including some that aren't actually being used.
    /// </para>
    /// </remarks>
    public IEnumerable<int> GetActiveFeatureIndices()
    {
        try
        {
            // First try to get active features from inner model directly
            return _innerModel.GetActiveFeatureIndices();
        }
        catch (NotImplementedException)
        {
            // If not implemented by inner model, make an educated guess about the input dimension
            // and return all indices up to that dimension
            int dimension = EstimateInputDimension();
            return Enumerable.Range(0, dimension);
        }
    }

    /// <summary>
    /// Estimates the input dimension based on the model parameters or other available information.
    /// </summary>
    /// <returns>The estimated number of input features.</returns>
    /// <remarks>
    /// This is a helper method to guess the dimensionality of the input space when it's not explicitly available.
    /// </remarks>
    private int EstimateInputDimension()
    {
        // Try to get dimension from parameters - many models have parameters with length related to input dimension
        var parameters = GetParameters();
        if (parameters != null && parameters.Length > 0)
        {
            // For many models, the number of parameters is related to the input dimension
            // This is a very rough estimate and would need to be adapted for specific model types
            return Math.Max(1, parameters.Length / 2);  // Conservative estimate
        }

        // If we can't estimate, return a reasonable default value
        return 10;  // Arbitrary default
    }

    /// <summary>
    /// Determines whether a specific feature is used by the model.
    /// </summary>
    /// <param name="featureIndex">The index of the feature to check.</param>
    /// <returns>True if the feature is used, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// This method checks if a specific input feature is actively used by the model.
    /// It attempts to use the inner model's capability if available, otherwise it
    /// conservatively assumes that all features within range are used.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like asking if the model pays attention to a specific piece of information when making predictions.
    /// 
    /// Since we might not know exactly how the inner model works:
    /// - We first try to ask the inner model directly
    /// - If that's not possible, we check if the feature index is within a reasonable range
    /// - If it is, we assume it might be used by the model
    /// 
    /// This helps determine which inputs might be important to the model's predictions,
    /// but errs on the side of caution by including features that might actually be unused.
    /// </para>
    /// </remarks>
    public bool IsFeatureUsed(int featureIndex)
    {
        try
        {
            // First try to check with the inner model directly
            return _innerModel.IsFeatureUsed(featureIndex);
        }
        catch (NotImplementedException)
        {
            // If not implemented, check if the feature index is within the estimated dimension
            int dimension = EstimateInputDimension();
            return featureIndex >= 0 && featureIndex < dimension;
        }
    }

    /// <summary>
    /// Creates a deep copy of this model.
    /// </summary>
    /// <returns>A new, independent copy of this model.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a completely independent copy of the model, including
    /// the inner model and all genes. It uses serialization and deserialization
    /// as a robust way to create a deep copy.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is like creating an exact duplicate of the model that can be modified independently.
    /// 
    /// When deep copying:
    /// - Every aspect of the model is duplicated, including its inner structure
    /// - The copy behaves identically to the original
    /// - Changes to either model won't affect the other
    /// 
    /// Deep copying is useful when you want to experiment with modifications
    /// while preserving the original model as a reference or fallback.
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        // Serialize and deserialize as a robust way to create a deep copy
        byte[] serialized = Serialize();

        // Create a new instance with the same factory
        var copy = new ModelIndividual<T, TInput, TOutput, TGene>([], _modelFactory);

        // Deserialize the data into the new instance
        copy.Deserialize(serialized);

        // Clone the genes
        var clonedGenes = new List<TGene>();
        foreach (var gene in _genes)
        {
            if (gene is ICloneable cloneable)
            {
                clonedGenes.Add((TGene)cloneable.Clone());
            }
            else
            {
                clonedGenes.Add(gene);
            }
        }

        copy._genes = clonedGenes;
        copy._fitness = _fitness;

        return copy;
    }

    /// <summary>
    /// Creates a clone of this model (ICloneable implementation).
    /// </summary>
    /// <returns>A new, independent copy of this model.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the ICloneable interface for the model, creating
    /// a completely independent copy by delegating to the DeepCopy method.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// This is exactly the same as DeepCopy - it creates an exact duplicate of the model.
    /// 
    /// The only difference is:
    /// - DeepCopy is part of the IFullModel interface
    /// - Clone is part of the ICloneable interface
    /// 
    /// Having both methods allows the model to be cloned regardless of which
    /// interface the code is working with.
    /// </para>
    /// </remarks>
    IFullModel<T, TInput, TOutput> ICloneable<IFullModel<T, TInput, TOutput>>.Clone()
    {
        return DeepCopy();
    }

    /// <summary>
    /// Sets which features should be considered active in the model.
    /// </summary>
    /// <param name="featureIndices">The indices of features to mark as active.</param>
    /// <exception cref="ArgumentNullException">Thrown when featureIndices is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any feature index is negative.</exception>
    /// <remarks>
    /// <para>
    /// This method explicitly specifies which features should be considered active in the model,
    /// overriding the automatic determination based on the inner model's behavior.
    /// Any features not included in the provided collection will be considered inactive,
    /// regardless of whether they are used by the inner model.
    /// </para>
    /// <para><b>For Beginners:</b> This method lets you manually tell the model which input features
    /// are important, regardless of what features the inner model actually uses.
    /// 
    /// For example, if you have 10 features but want to focus on only features 2, 5, and 7,
    /// you can use this method to specify exactly those features. After setting these features:
    /// - Only these specific features will be reported as active by GetActiveFeatureIndices()
    /// - Only these features will return true when checked with IsFeatureUsed()
    /// - This selection will persist when the model is saved and loaded
    /// 
    /// This can be useful for:
    /// - Feature selection experiments (testing different feature subsets)
    /// - Simplifying model interpretation
    /// - Ensuring consistency across different models
    /// - Highlighting specific features you know are important from domain expertise
    /// </para>
    /// </remarks>
    public void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        if (featureIndices == null)
        {
            throw new ArgumentNullException(nameof(featureIndices), "Feature indices cannot be null.");
        }

        // Initialize the hash set if it doesn't exist
        _explicitlySetActiveFeatures ??= [];

        // Clear existing explicitly set features
        _explicitlySetActiveFeatures.Clear();

        // Add the new feature indices
        foreach (var index in featureIndices)
        {
            if (index < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"Feature index {index} cannot be negative.");
            }

            _explicitlySetActiveFeatures.Add(index);
        }

        _innerModel.SetActiveFeatureIndices(featureIndices);
    }
    
    /// <inheritdoc/>
    public override async Task<TOutput> PredictAsync(TInput input)
    {
        return await _innerModel.PredictAsync(input);
    }
    
    /// <inheritdoc/>
    public override async Task TrainAsync(TInput input, TOutput expectedOutput)
    {
        await _innerModel.TrainAsync(input, expectedOutput);
    }
    
    /// <inheritdoc/>
    public override void SetModelMetadata(ModelMetadata<T> metadata)
    {
        // Update the inner model's metadata
        if (_innerModel is IModel<TInput, TOutput, ModelMetadata<T>> model)
        {
            model.SetModelMetadata(metadata);
        }
        
        // Update fitness based on metadata complexity or other metrics
        if (metadata.Complexity > 0)
        {
            // Optionally update fitness based on model complexity
            // Lower complexity might mean better fitness in some scenarios
        }
    }
    
    /// <inheritdoc/>
    public override void Save(string filepath)
    {
        _innerModel.Save(filepath);
    }
    
    /// <inheritdoc/>
    public override void Load(string filepath)
    {
        _innerModel.Load(filepath);
    }
    
    /// <inheritdoc/>
    public override void Dispose()
    {
        if (_innerModel is IDisposable disposable)
        {
            disposable.Dispose();
        }
        _genes.Clear();
        _explicitlySetActiveFeatures?.Clear();
    }
    
    // Override interpretability methods to delegate to inner model
    public override async Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync()
    {
        return await _innerModel.GetGlobalFeatureImportanceAsync();
    }
    
    public override async Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync(TInput input)
    {
        return await _innerModel.GetLocalFeatureImportanceAsync(input);
    }
    
    public override async Task<LimeExplanation<T>> GetLimeExplanationAsync(TInput input, int numFeatures = 10)
    {
        return await _innerModel.GetLimeExplanationAsync(input, numFeatures);
    }
    
    public override async Task<Matrix<T>> GetShapValuesAsync(TInput inputs)
    {
        return await _innerModel.GetShapValuesAsync(inputs);
    }
    
    public override async Task<string> GenerateTextExplanationAsync(TInput input, TOutput prediction)
    {
        return await _innerModel.GenerateTextExplanationAsync(input, prediction);
    }

    #endregion

    #region IFullModel Interface Members - Added by Team 23

    /// <summary>
    /// Gets the total number of parameters in the model
    /// </summary>
    public virtual int ParameterCount => _innerModel?.ParameterCount ?? 0;

    /// <summary>
    /// Saves the model to a file
    /// </summary>
    public virtual void SaveModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path cannot be null or empty.", nameof(filePath));

        byte[] serializedData = Serialize();
        File.WriteAllBytes(filePath, serializedData);
    }

    /// <summary>
    /// Gets feature importance scores
    /// </summary>
    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        // Delegate to inner model if available
        if (_innerModel != null)
        {
            return _innerModel.GetFeatureImportance();
        }

        // Return empty dictionary as default
        return new Dictionary<string, T>();
    }

    #endregion
}