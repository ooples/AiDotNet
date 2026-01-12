using AiDotNet.Interfaces;

namespace AiDotNet.Postprocessing;

/// <summary>
/// Chains multiple data transformers into a sequential postprocessing pipeline.
/// </summary>
/// <remarks>
/// <para>
/// A postprocessing pipeline applies transformers in sequence, passing the output
/// of each transformer as input to the next. This enables composable postprocessing
/// workflows similar to sklearn's Pipeline, applied to model outputs.
/// </para>
/// <para><b>For Beginners:</b> Think of a pipeline as a series of steps for processing
/// model output. For example:
/// 1. First, apply softmax to get probabilities
/// 2. Then, decode indices to labels
/// 3. Finally, format the output
///
/// The pipeline ensures all these steps happen in the right order.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
public class PostprocessingPipeline<T, TInput, TOutput> : IDataTransformer<T, TInput, TOutput>
{
    private readonly List<(string Name, IDataTransformer<T, TInput, TInput> Transformer)> _steps;
    private IDataTransformer<T, TInput, TOutput>? _finalTransformer;
    private bool _isFitted;

    /// <summary>
    /// Gets whether this pipeline has been fitted to data.
    /// </summary>
    public bool IsFitted => _isFitted;

    /// <summary>
    /// Gets the column indices this transformer operates on (null for pipelines).
    /// </summary>
    public int[]? ColumnIndices => null;

    /// <summary>
    /// Gets whether this pipeline supports inverse transformation.
    /// </summary>
    /// <remarks>
    /// A pipeline supports inverse transform if ALL its transformers support it.
    /// </remarks>
    public bool SupportsInverseTransform
    {
        get
        {
            foreach (var step in _steps)
            {
                if (!step.Transformer.SupportsInverseTransform)
                {
                    return false;
                }
            }
            if (_finalTransformer is not null && !_finalTransformer.SupportsInverseTransform)
            {
                return false;
            }
            return true;
        }
    }

    /// <summary>
    /// Gets the number of steps in the pipeline.
    /// </summary>
    public int Count => _steps.Count + (_finalTransformer is not null ? 1 : 0);

    /// <summary>
    /// Gets the named steps in the pipeline.
    /// </summary>
    public IReadOnlyList<(string Name, IDataTransformer<T, TInput, TInput> Transformer)> Steps => _steps.AsReadOnly();

    /// <summary>
    /// Creates a new empty postprocessing pipeline.
    /// </summary>
    public PostprocessingPipeline()
    {
        _steps = new List<(string, IDataTransformer<T, TInput, TInput>)>();
        _isFitted = false;
    }

    /// <summary>
    /// Adds a transformer step to the pipeline.
    /// </summary>
    /// <param name="transformer">The transformer to add.</param>
    /// <returns>This pipeline for method chaining.</returns>
    public PostprocessingPipeline<T, TInput, TOutput> Add(IDataTransformer<T, TInput, TInput> transformer)
    {
        return Add($"step_{_steps.Count}", transformer);
    }

    /// <summary>
    /// Adds a named transformer step to the pipeline.
    /// </summary>
    /// <param name="name">The name for this step.</param>
    /// <param name="transformer">The transformer to add.</param>
    /// <returns>This pipeline for method chaining.</returns>
    public PostprocessingPipeline<T, TInput, TOutput> Add(string name, IDataTransformer<T, TInput, TInput> transformer)
    {
        if (string.IsNullOrWhiteSpace(name))
        {
            throw new ArgumentException("Step name cannot be null or whitespace.", nameof(name));
        }

        if (transformer is null)
        {
            throw new ArgumentNullException(nameof(transformer));
        }

        // Check for duplicate names
        foreach (var step in _steps)
        {
            if (step.Name.Equals(name, StringComparison.Ordinal))
            {
                throw new ArgumentException($"A step with name '{name}' already exists.", nameof(name));
            }
        }

        _steps.Add((name, transformer));
        _isFitted = false;

        return this;
    }

    /// <summary>
    /// Sets the final transformer that may change the output type.
    /// </summary>
    /// <param name="transformer">The final transformer.</param>
    /// <returns>This pipeline for method chaining.</returns>
    public PostprocessingPipeline<T, TInput, TOutput> SetFinalTransformer(IDataTransformer<T, TInput, TOutput> transformer)
    {
        _finalTransformer = transformer ?? throw new ArgumentNullException(nameof(transformer));
        _isFitted = false;
        return this;
    }

    /// <summary>
    /// Gets a transformer step by name.
    /// </summary>
    /// <param name="name">The step name.</param>
    /// <returns>The transformer, or null if not found.</returns>
    public IDataTransformer<T, TInput, TInput>? GetStep(string name)
    {
        foreach (var step in _steps)
        {
            if (step.Name.Equals(name, StringComparison.Ordinal))
            {
                return step.Transformer;
            }
        }
        return null;
    }

    /// <summary>
    /// Fits all transformers in the pipeline to the data.
    /// </summary>
    /// <param name="data">The data to fit.</param>
    /// <remarks>
    /// <para>
    /// For postprocessing, fitting learns any parameters needed for transformation.
    /// Many postprocessors are stateless and don't require fitting.
    /// </para>
    /// </remarks>
    public void Fit(TInput data)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        // Fit each step in sequence, transforming as we go
        TInput current = data;

        foreach (var step in _steps)
        {
            step.Transformer.Fit(current);
            current = step.Transformer.Transform(current);
        }

        // Fit final transformer if present
        if (_finalTransformer is not null)
        {
            _finalTransformer.Fit(current);
        }

        _isFitted = true;
    }

    /// <summary>
    /// Transforms data through all pipeline steps.
    /// </summary>
    /// <param name="data">The data to transform.</param>
    /// <returns>The transformed data.</returns>
    /// <exception cref="InvalidOperationException">Thrown if not fitted.</exception>
    public TOutput Transform(TInput data)
    {
        EnsureFitted();

        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        // Transform through each step
        TInput current = data;

        foreach (var step in _steps)
        {
            current = step.Transformer.Transform(current);
        }

        // Apply final transformer or cast
        if (_finalTransformer is not null)
        {
            return _finalTransformer.Transform(current);
        }

        // If TInput == TOutput, we can cast
        if (current is TOutput output)
        {
            return output;
        }

        throw new InvalidOperationException(
            $"Pipeline has no final transformer and cannot convert {typeof(TInput).Name} to {typeof(TOutput).Name}.");
    }

    /// <summary>
    /// Fits the pipeline and transforms the data in a single step.
    /// </summary>
    /// <param name="data">The data to fit and transform.</param>
    /// <returns>The transformed data.</returns>
    public TOutput FitTransform(TInput data)
    {
        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        // FitTransform each step in sequence
        TInput current = data;

        foreach (var step in _steps)
        {
            current = step.Transformer.FitTransform(current);
        }

        // Apply final transformer or cast
        if (_finalTransformer is not null)
        {
            _isFitted = true;
            return _finalTransformer.FitTransform(current);
        }

        _isFitted = true;

        // If TInput == TOutput, we can cast
        if (current is TOutput output)
        {
            return output;
        }

        throw new InvalidOperationException(
            $"Pipeline has no final transformer and cannot convert {typeof(TInput).Name} to {typeof(TOutput).Name}.");
    }

    /// <summary>
    /// Inverse transforms data through all pipeline steps in reverse order.
    /// </summary>
    /// <param name="data">The transformed data.</param>
    /// <returns>The original-scale data.</returns>
    /// <exception cref="NotSupportedException">Thrown if any step doesn't support inverse.</exception>
    /// <exception cref="InvalidOperationException">Thrown if not fitted.</exception>
    public TInput InverseTransform(TOutput data)
    {
        EnsureFitted();

        if (!SupportsInverseTransform)
        {
            throw new NotSupportedException(
                "One or more pipeline steps do not support inverse transformation.");
        }

        if (data is null)
        {
            throw new ArgumentNullException(nameof(data));
        }

        TInput current;

        // Inverse transform through final transformer first
        if (_finalTransformer is not null)
        {
            current = _finalTransformer.InverseTransform(data);
        }
        else if (data is TInput input)
        {
            current = input;
        }
        else
        {
            throw new InvalidOperationException(
                $"Cannot convert {typeof(TOutput).Name} to {typeof(TInput).Name} for inverse transform.");
        }

        // Inverse transform through steps in reverse order
        for (int i = _steps.Count - 1; i >= 0; i--)
        {
            current = _steps[i].Transformer.InverseTransform(current);
        }

        return current;
    }

    /// <summary>
    /// Gets the output feature names after all transformations.
    /// </summary>
    /// <param name="inputFeatureNames">The input feature names (optional).</param>
    /// <returns>The output feature names.</returns>
    public string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        string[] names = inputFeatureNames ?? Array.Empty<string>();

        foreach (var step in _steps)
        {
            names = step.Transformer.GetFeatureNamesOut(names);
        }

        if (_finalTransformer is not null)
        {
            names = _finalTransformer.GetFeatureNamesOut(names);
        }

        return names;
    }

    /// <summary>
    /// Creates a clone of this pipeline without fitted state.
    /// </summary>
    /// <returns>A new unfitted pipeline with the same structure.</returns>
    public PostprocessingPipeline<T, TInput, TOutput> Clone()
    {
        var clone = new PostprocessingPipeline<T, TInput, TOutput>();

        // Note: This creates a shallow clone - transformers are not cloned
        // For deep cloning, transformers would need to implement ICloneable
        foreach (var step in _steps)
        {
            clone._steps.Add(step);
        }

        clone._finalTransformer = _finalTransformer;

        return clone;
    }

    private void EnsureFitted()
    {
        if (!_isFitted)
        {
            throw new InvalidOperationException(
                "Pipeline has not been fitted. Call Fit() or FitTransform() first.");
        }
    }
}
