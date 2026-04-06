
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Implementation of Associative Memory for nested learning using modern continuous
/// Hopfield retrieval (Ramsauer et al. 2021). Stores key-value pairs and retrieves
/// via softmax attention: new_state = softmax(β * keys^T @ query) @ values.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class AssociativeMemory<T> : NestedLearningBase<T>, IAssociativeMemory<T>
{
    private readonly int _capacity;
    private readonly int _dimension;
    private readonly double _inverseTemperature;
    private readonly List<(Vector<T> Input, Vector<T> Target)> _memories;
    private Matrix<T>? _cachedAssociationMatrix;

    public AssociativeMemory(int dimension, int capacity = 1000, double inverseTemperature = 8.0)
    {
        if (dimension < 1)
            throw new ArgumentOutOfRangeException(nameof(dimension), dimension, "Dimension must be at least 1.");
        if (capacity < 1)
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be at least 1.");
        if (inverseTemperature <= 0 || double.IsNaN(inverseTemperature) || double.IsInfinity(inverseTemperature))
            throw new ArgumentOutOfRangeException(nameof(inverseTemperature), inverseTemperature, "Must be a finite positive value.");
        _dimension = dimension;
        _capacity = capacity;
        _inverseTemperature = inverseTemperature;
        _memories = new List<(Vector<T>, Vector<T>)>();
    }

    public void Associate(Vector<T> input, Vector<T> target)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (target == null) throw new ArgumentNullException(nameof(target));
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        _memories.Add((input.Clone(), target.Clone()));
        _cachedAssociationMatrix = null;

        if (_memories.Count > _capacity)
        {
            _memories.RemoveAt(0);
        }
    }

    public Vector<T> Retrieve(Vector<T> query)
    {
        if (query == null) throw new ArgumentNullException(nameof(query));
        if (query.Length != _dimension)
            throw new ArgumentException($"Query length {query.Length} must match memory dimension {_dimension}.", nameof(query));

        if (_memories.Count == 0)
            return new Vector<T>(_dimension);

        // Modern continuous Hopfield retrieval per Ramsauer et al. 2021:
        // Scores keys (Input) against query, returns weighted sum of values (Target).
        // new_state = softmax(β * keys^T @ query) @ values
        var scores = new double[_memories.Count];
        double maxScore = double.NegativeInfinity;

        for (int m = 0; m < _memories.Count; m++)
        {
            double dot = 0;
            var key = _memories[m].Input;
            for (int d = 0; d < _dimension; d++)
                dot += NumOps.ToDouble(NumOps.Multiply(key[d], query[d]));
            scores[m] = _inverseTemperature * dot;
            if (scores[m] > maxScore) maxScore = scores[m];
        }

        double sumExp = 0;
        for (int m = 0; m < _memories.Count; m++)
        {
            scores[m] = Math.Exp(scores[m] - maxScore);
            sumExp += scores[m];
        }
        for (int m = 0; m < _memories.Count; m++)
            scores[m] /= (sumExp + 1e-10);

        var result = new Vector<T>(_dimension);
        for (int m = 0; m < _memories.Count; m++)
        {
            T weight = NumOps.FromDouble(scores[m]);
            var value = _memories[m].Target;
            for (int d = 0; d < _dimension; d++)
                result[d] = NumOps.Add(result[d], NumOps.Multiply(weight, value[d]));
        }
        return result;
    }

    public void Update(Vector<T> input, Vector<T> target, T learningRate)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (target == null) throw new ArgumentNullException(nameof(target));
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        // If a matching key already exists, replace its target (accumulate learning signal)
        // instead of appending a duplicate that splits softmax attention
        for (int m = 0; m < _memories.Count; m++)
        {
            double dot = 0;
            var key = _memories[m].Input;
            for (int d = 0; d < _dimension; d++)
                dot += NumOps.ToDouble(NumOps.Multiply(key[d], input[d]));

            double keyNormSq = 0, inputNormSq = 0;
            for (int d = 0; d < _dimension; d++)
            {
                keyNormSq += NumOps.ToDouble(NumOps.Multiply(key[d], key[d]));
                inputNormSq += NumOps.ToDouble(NumOps.Multiply(input[d], input[d]));
            }

            double cosine = dot / (Math.Sqrt(keyNormSq * inputNormSq) + 1e-10);
            if (cosine > 0.99)
            {
                // Blend existing target with new target using learning rate
                var existing = _memories[m].Target;
                var updated = new Vector<T>(_dimension);
                for (int d = 0; d < _dimension; d++)
                {
                    T keep = NumOps.Multiply(NumOps.Subtract(NumOps.One, learningRate), existing[d]);
                    T add = NumOps.Multiply(learningRate, target[d]);
                    updated[d] = NumOps.Add(keep, add);
                }
                _memories[m] = (key, updated);
                _cachedAssociationMatrix = null;
                return;
            }
        }

        // No matching key — store as new association
        Associate(input, target);
    }

    public int Capacity => _capacity;

    public void Clear()
    {
        _memories.Clear();
        _cachedAssociationMatrix = null;
    }

    /// <summary>
    /// Computes the Hebbian association matrix from stored memories: W = Σ target_i * input_i^T.
    /// Cached and invalidated on Associate/Update/Clear. For diagnostics/testing only.
    /// </summary>
    public Matrix<T> GetAssociationMatrix()
    {
        if (_cachedAssociationMatrix != null)
            return _cachedAssociationMatrix;

        var matrix = new Matrix<T>(_dimension, _dimension);
        foreach (var (input, target) in _memories)
        {
            for (int i = 0; i < _dimension; i++)
                for (int j = 0; j < _dimension; j++)
                    matrix[i, j] = NumOps.Add(matrix[i, j], NumOps.Multiply(target[i], input[j]));
        }
        _cachedAssociationMatrix = matrix;
        return matrix;
    }

    /// <summary>
    /// Gets the number of stored memories.
    /// </summary>
    public int MemoryCount => _memories.Count;
}
