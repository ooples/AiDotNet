﻿namespace AiDotNet.Genetics;

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
    IEvolvable<TGene, T>,
    IFullModel<T, TInput, TOutput>
    where TGene : class
{
    private List<TGene> _genes = [];
    private T _fitness;
    private IFullModel<T, TInput, TOutput> _innerModel;
    private readonly Func<ICollection<TGene>, IFullModel<T, TInput, TOutput>> _modelFactory;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Creates a new model individual with the specified genes and model factory.
    /// </summary>
    /// <param name="genes">The genes to initialize with.</param>
    /// <param name="modelFactory">A function that creates a model from genes.</param>
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
    public ICollection<TGene> GetGenes()
    {
        return _genes;
    }

    /// <summary>
    /// Sets the genes of this individual and updates the inner model.
    /// </summary>
    /// <param name="genes">The new genes.</param>
    public void SetGenes(ICollection<TGene> genes)
    {
        _genes = [.. genes];
        // Recreate the model with the new genes
        _innerModel = _modelFactory(_genes);
    }

    /// <summary>
    /// Gets the fitness of this individual.
    /// </summary>
    /// <returns>The fitness value.</returns>
    public T GetFitness()
    {
        return _fitness;
    }

    /// <summary>
    /// Sets the fitness of this individual.
    /// </summary>
    /// <param name="fitness">The new fitness value.</param>
    public void SetFitness(T fitness)
    {
        _fitness = fitness;
    }

    /// <summary>
    /// Creates a deep clone of this individual.
    /// </summary>
    /// <returns>A new individual with the same genes and fitness.</returns>
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
    public TOutput Predict(TInput input)
    {
        return _innerModel.Predict(input);
    }

    /// <summary>
    /// Gets the metadata for the model.
    /// </summary>
    /// <returns>The model metadata.</returns>
    public ModelMetaData<T> GetMetaData()
    {
        return _innerModel.GetModelMetaData();
    }

    /// <summary>
    /// Gets the parameters of the model.
    /// </summary>
    /// <returns>The model parameters as a vector.</returns>
    public Vector<T> GetParameters()
    {
        return _innerModel.GetParameters();
    }

    /// <summary>
    /// Updates the parameters of the model.
    /// </summary>
    /// <param name="parameters">The new parameters.</param>
    public void UpdateParameters(Vector<T> parameters)
    {
        // Replace the inner model with a new instance containing the updated parameters
        _innerModel = _innerModel.WithParameters(parameters);
    }

    /// <summary>
    /// Creates a new model with the specified parameters.
    /// </summary>
    /// <param name="parameters">The parameters for the new model.</param>
    /// <returns>A new model with the specified parameters.</returns>
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
    public byte[] Serialize()
    {
        return _innerModel.Serialize();
    }

    /// <summary>
    /// Deserializes the model from a byte array.
    /// </summary>
    /// <param name="data">The byte array containing the serialized model.</param>
    public void Deserialize(byte[] data)
    {
        _innerModel.Deserialize(data);
    }

    public void Train(TInput input, TOutput expectedOutput)
    {
        _innerModel.Train(input, expectedOutput);
    }

    public ModelMetaData<T> GetModelMetaData()
    {
        return _innerModel.GetModelMetaData();
    }

    public IEnumerable<int> GetActiveFeatureIndices()
    {
        return _innerModel.GetActiveFeatureIndices();
    }

    public bool IsFeatureUsed(int featureIndex)
    {
        return _innerModel.IsFeatureUsed(featureIndex);
    }

    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        // Deep copy the inner model and wrap it in a new ModelIndividual to preserve genes and factory
        var copiedInner = _innerModel.DeepCopy();
        return new ModelIndividual<T, TInput, TOutput, TGene>(copiedInner, _genes, _modelFactory);
    }

    IFullModel<T, TInput, TOutput> ICloneable<IFullModel<T, TInput, TOutput>>.Clone()
    {
        // Clone the inner model and wrap it in a new ModelIndividual to preserve genes and factory
        var clonedInner = _innerModel is ICloneable<IFullModel<T, TInput, TOutput>> cl
            ? cl.Clone()
            : _innerModel.DeepCopy();

        return new ModelIndividual<T, TInput, TOutput, TGene>(clonedInner, _genes, _modelFactory);
    }

    /// <summary>
    /// Gets the total number of parameters for this model.
    /// </summary>
    public virtual int ParameterCount
    {
        get
        {
            var parameters = _innerModel.GetParameters();
            return parameters?.Length ?? 0;
        }
    }

    /// <summary>
    /// Sets parameters on the inner model by creating a new model instance with the given parameters.
    /// </summary>
    /// <param name="parameters">The parameters to apply.</param>
    public void SetParameters(Vector<T> parameters)
    {
        _innerModel = _innerModel.WithParameters(parameters);
    }

    #endregion
}
