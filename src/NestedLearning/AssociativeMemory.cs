
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Implementation of Associative Memory for nested learning.
/// Models both backpropagation (data point → local error) and
/// attention mechanisms (query → key-value) as associative memory.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class AssociativeMemory<T> : IAssociativeMemory<T>
{
    private readonly int _capacity;
    private readonly int _dimension;
    private readonly double _inverseTemperature;
    private readonly List<(Vector<T> Input, Vector<T> Target)> _memories;
    private Matrix<T> _associationMatrix;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public AssociativeMemory(int dimension, int capacity = 1000, double inverseTemperature = 8.0)
    {
        if (capacity < 1)
            throw new ArgumentOutOfRangeException(nameof(capacity), capacity, "Capacity must be at least 1.");
        if (inverseTemperature <= 0 || double.IsNaN(inverseTemperature) || double.IsInfinity(inverseTemperature))
            throw new ArgumentOutOfRangeException(nameof(inverseTemperature), inverseTemperature, "Must be a finite positive value.");
        _dimension = dimension;
        _capacity = capacity;
        _inverseTemperature = inverseTemperature;
        _memories = new List<(Vector<T>, Vector<T>)>();
        _associationMatrix = new Matrix<T>(dimension, dimension);
    }

    public void Associate(Vector<T> input, Vector<T> target)
    {
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        // Add to memory buffer
        _memories.Add((input.Clone(), target.Clone()));

        // Maintain capacity limit (FIFO)
        if (_memories.Count > _capacity)
        {
            _memories.RemoveAt(0);
        }

        // Update association matrix using Hebbian-like learning
        UpdateAssociationMatrix(input, target, _numOps.FromDouble(0.01));
    }

    public Vector<T> Retrieve(Vector<T> query)
    {
        if (query.Length != _dimension)
            throw new ArgumentException($"Query length {query.Length} must match memory dimension {_dimension}.", nameof(query));

        // Modern continuous Hopfield retrieval per Ramsauer et al. 2021:
        // Scores keys (Input) against query, returns weighted sum of values (Target).
        // new_state = softmax(β * keys^T @ query) @ values
        // Falls back to association matrix when no memories are stored.
        if (_memories.Count > 0)
        {
            var scores = new double[_memories.Count];
            double maxScore = double.NegativeInfinity;

            // Compute similarity scores: β * <key_i, query>
            for (int m = 0; m < _memories.Count; m++)
            {
                double dot = 0;
                var key = _memories[m].Input;
                for (int d = 0; d < _dimension; d++)
                    dot += _numOps.ToDouble(_numOps.Multiply(key[d], query[d]));
                scores[m] = _inverseTemperature * dot;
                if (scores[m] > maxScore) maxScore = scores[m];
            }

            // Softmax (numerically stable)
            double sumExp = 0;
            for (int m = 0; m < _memories.Count; m++)
            {
                scores[m] = Math.Exp(scores[m] - maxScore);
                sumExp += scores[m];
            }
            for (int m = 0; m < _memories.Count; m++)
                scores[m] /= (sumExp + 1e-10);

            // Weighted sum of stored values (Target)
            var result = new Vector<T>(_dimension);
            for (int m = 0; m < _memories.Count; m++)
            {
                T weight = _numOps.FromDouble(scores[m]);
                var value = _memories[m].Target;
                for (int d = 0; d < _dimension; d++)
                    result[d] = _numOps.Add(result[d], _numOps.Multiply(weight, value[d]));
            }
            return result;
        }

        // No memories stored — fall back to association matrix retrieval
        return _associationMatrix.Multiply(query);
    }

    public void Update(Vector<T> input, Vector<T> target, T learningRate)
    {
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        // Update both the association matrix and the memory buffer
        // so changes are reflected in both Retrieve paths
        UpdateAssociationMatrix(input, target, learningRate);
        Associate(input, target);
    }

    private void UpdateAssociationMatrix(Vector<T> input, Vector<T> target, T learningRate)
    {
        // Hebbian learning rule: Δw_ij = η * target_i * input_j
        // This models how backpropagation maps data points to local errors
        for (int i = 0; i < _dimension; i++)
        {
            for (int j = 0; j < _dimension; j++)
            {
                T update = _numOps.Multiply(_numOps.Multiply(target[i], input[j]), learningRate);
                _associationMatrix[i, j] = _numOps.Add(_associationMatrix[i, j], update);
            }
        }
    }

    private T ComputeSimilarity(Vector<T> a, Vector<T> b)
    {
        return _numOps.FromDouble(VectorHelper.CosineSimilarity(a, b));
    }

    public int Capacity => _capacity;

    public void Clear()
    {
        _memories.Clear();
        _associationMatrix = new Matrix<T>(_dimension, _dimension);
    }

    /// <summary>
    /// Gets the association matrix for inspection/debugging.
    /// </summary>
    public Matrix<T> GetAssociationMatrix() => _associationMatrix;

    /// <summary>
    /// Gets the number of stored memories.
    /// </summary>
    public int MemoryCount => _memories.Count;
}
