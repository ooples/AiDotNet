using System;
using System.Collections.Generic;

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
        // Initialize with a copy of provided genes to avoid shared references
        _genes = [.. genes];
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

    public virtual Dictionary<string, T> GetFeatureImportance()
    {
        return _innerModel.GetFeatureImportance();
    }

    public virtual void SetActiveFeatureIndices(IEnumerable<int> featureIndices)
    {
        _innerModel.SetActiveFeatureIndices(featureIndices);
    }

    public bool IsFeatureUsed(int featureIndex)
    {
        return _innerModel.IsFeatureUsed(featureIndex);
    }

    public IFullModel<T, TInput, TOutput> DeepCopy()
    {
        var copiedInner = _innerModel.DeepCopy();
        // Deep copy genes where possible
        var clonedGenes = new List<TGene>(_genes.Count);
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
        return new ModelIndividual<T, TInput, TOutput, TGene>(copiedInner, clonedGenes, _modelFactory);
    }

    IFullModel<T, TInput, TOutput> ICloneable<IFullModel<T, TInput, TOutput>>.Clone()
    {
        var cloned = _innerModel.Clone();
        // Deep copy genes where possible
        var clonedGenes = new List<TGene>(_genes.Count);
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
        return new ModelIndividual<T, TInput, TOutput, TGene>(cloned, clonedGenes, _modelFactory);
    }

    public virtual void SetParameters(Vector<T> parameters)
    {
        _innerModel = _innerModel.WithParameters(parameters);
        _parameterCountCache = null; // invalidate cache
    }

    private int? _parameterCountCache;
    public virtual int ParameterCount
        => _parameterCountCache ??= _innerModel.GetParameters()?.Length ?? 0;

    public virtual void SaveModel(string filePath)
    {
        if (_innerModel == null)
        {
            throw new InvalidOperationException("Inner model is not initialized.");
        }
        _innerModel.SaveModel(filePath);
    }

    public virtual void LoadModel(string filePath)
    {
        if (_innerModel == null)
        {
            _innerModel = _modelFactory(_genes);
        }
        _innerModel.LoadModel(filePath);
    }

    #endregion
}
