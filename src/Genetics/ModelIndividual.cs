using System;
using System.Collections.Generic;
using System.IO;
using AiDotNet.Autodiff;

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
    private List<TGene> _genes = new List<TGene>();
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
        _genes = [.. genes];
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
    public ModelMetadata<T> GetMetaData()
    {
        return _innerModel.GetModelMetadata();
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

    public ModelMetadata<T> GetModelMetadata()
    {
        return _innerModel.GetModelMetadata();
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
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));

        try
        {
            var data = Serialize();
            var directory = Path.GetDirectoryName(filePath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllBytes(filePath, data);
        }
        catch (IOException ex) { throw new InvalidOperationException($"Failed to save model to '{filePath}': {ex.Message}", ex); }
        catch (UnauthorizedAccessException ex) { throw new InvalidOperationException($"Access denied when saving model to '{filePath}': {ex.Message}", ex); }
        catch (System.Security.SecurityException ex) { throw new InvalidOperationException($"Security error when saving model to '{filePath}': {ex.Message}", ex); }
    }

    public virtual void LoadModel(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
            throw new ArgumentException("File path must not be null or empty.", nameof(filePath));

        try
        {
            var data = File.ReadAllBytes(filePath);
            Deserialize(data);
        }
        catch (FileNotFoundException ex) { throw new FileNotFoundException($"The specified model file does not exist: {filePath}", filePath, ex); }
        catch (IOException ex) { throw new InvalidOperationException($"File I/O error while loading model from '{filePath}': {ex.Message}", ex); }
        catch (UnauthorizedAccessException ex) { throw new InvalidOperationException($"Access denied when loading model from '{filePath}': {ex.Message}", ex); }
        catch (System.Security.SecurityException ex) { throw new InvalidOperationException($"Security error when loading model from '{filePath}': {ex.Message}", ex); }
        catch (Exception ex) { throw new InvalidOperationException($"Failed to deserialize model from file '{filePath}'. The file may be corrupted or incompatible: {ex.Message}", ex); }
    }

    /// <summary>
    /// Gets the default loss function for gradient computation by delegating to the inner model.
    /// </summary>
    public ILossFunction<T> DefaultLossFunction => _innerModel.DefaultLossFunction;

    /// <summary>
    /// Computes gradients by delegating to the inner model.
    /// </summary>
    public Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null)
    {
        return _innerModel.ComputeGradients(input, target, lossFunction);
    }

    /// <summary>
    /// Applies gradients by delegating to the inner model.
    /// </summary>
    public void ApplyGradients(Vector<T> gradients, T learningRate)
    {
        _innerModel.ApplyGradients(gradients, learningRate);
    }

    /// <summary>
    /// Saves the model's current state to a stream.
    /// </summary>
    public void SaveState(Stream stream)
    {
        _innerModel.SaveState(stream);
    }

    /// <summary>
    /// Loads the model's state from a stream.
    /// </summary>
    public void LoadState(Stream stream)
    {
        _innerModel.LoadState(stream);
    }


    #region IJitCompilable Implementation

    /// <summary>
    /// Gets whether this model currently supports JIT compilation.
    /// </summary>
    /// <value>True if the inner model supports JIT compilation, false otherwise.</value>
    /// <remarks>
    /// <para>
    /// Model individuals delegate JIT compilation support to their inner model.
    /// Genetic evolution does not affect JIT compilability - it depends on the wrapped model type.
    /// </para>
    /// <para><b>For Beginners:</b> Genetically evolved models can be JIT compiled if their inner model supports it.
    ///
    /// The genetic algorithm modifies the model's genes (parameters/structure), but:
    /// - The underlying computation graph can still be JIT compiled
    /// - Evolution happens at the model level, JIT compilation at the execution level
    /// - Both work together: evolution finds good parameters, JIT makes them run fast
    /// </para>
    /// </remarks>
    public virtual bool SupportsJitCompilation
    {
        get
        {
            if (_innerModel is null)
                return false;

            return _innerModel.SupportsJitCompilation;
        }
    }

    /// <summary>
    /// Exports the computation graph for JIT compilation by delegating to the inner model.
    /// </summary>
    /// <param name="inputNodes">List to populate with input computation nodes.</param>
    /// <returns>The output computation node representing the model's prediction.</returns>
    /// <remarks>
    /// <para>
    /// Model individuals delegate graph export to their inner model.
    /// The graph represents the current evolved model's computation.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a computation graph from the evolved model.
    ///
    /// When genetic algorithms evolve a model:
    /// - The genes determine the model's parameters or structure
    /// - The inner model is rebuilt from those genes
    /// - That inner model can then be JIT compiled for fast execution
    ///
    /// This allows you to:
    /// - Evolve models to find good architectures
    /// - JIT compile the best evolved models for production use
    /// - Get both the benefits of evolution and fast execution
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentNullException">Thrown when inputNodes is null.</exception>
    /// <exception cref="InvalidOperationException">Thrown when inner model is null.</exception>
    /// <exception cref="NotSupportedException">
    /// Thrown when the inner model does not support JIT compilation.
    /// </exception>
    public virtual ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        if (_innerModel is null)
            throw new InvalidOperationException(
                "Cannot export computation graph: Inner model is null.");

        if (!_innerModel.SupportsJitCompilation)
            throw new NotSupportedException(
                $"The inner model of type {_innerModel.GetType().Name} does not support JIT compilation. " +
                "JIT compilation availability depends on the inner model's capabilities.");

        return _innerModel.ExportComputationGraph(inputNodes);
    }

    #endregion
    #endregion
}
