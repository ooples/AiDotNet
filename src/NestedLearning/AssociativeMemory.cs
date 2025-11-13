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
    private readonly List<(Vector<T> Input, Vector<T> Target)> _memories;
    private Matrix<T> _associationMatrix;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public AssociativeMemory(int dimension, int capacity = 1000)
    {
        _dimension = dimension;
        _capacity = capacity;
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
            throw new ArgumentException("Query must match memory dimension");

        // Retrieve using association matrix (similar to attention mechanism)
        var retrieved = _associationMatrix.Multiply(query);

        // Also check for exact or near matches in memory buffer
        T bestSimilarity = _numOps.FromDouble(double.NegativeInfinity);
        Vector<T>? bestMatch = null;

        foreach (var (input, target) in _memories)
        {
            T similarity = ComputeSimilarity(query, input);
            if (_numOps.GreaterThan(similarity, bestSimilarity))
            {
                bestSimilarity = similarity;
                bestMatch = target;
            }
        }

        // Blend matrix-based retrieval with buffer-based retrieval
        if (bestMatch != null && _numOps.GreaterThan(bestSimilarity, _numOps.FromDouble(0.8)))
        {
            T blendFactor = _numOps.FromDouble(0.3);
            var blended = new Vector<T>(_dimension);

            for (int i = 0; i < _dimension; i++)
            {
                T matrixPart = _numOps.Multiply(retrieved[i],
                    _numOps.Subtract(_numOps.One, blendFactor));
                T bufferPart = _numOps.Multiply(bestMatch[i], blendFactor);
                blended[i] = _numOps.Add(matrixPart, bufferPart);
            }

            return blended;
        }

        return retrieved;
    }

    public void Update(Vector<T> input, Vector<T> target, T learningRate)
    {
        if (input.Length != _dimension || target.Length != _dimension)
            throw new ArgumentException("Input and target must match memory dimension");

        UpdateAssociationMatrix(input, target, learningRate);
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
        // Cosine similarity
        T dotProduct = _numOps.Zero;
        T normA = _numOps.Zero;
        T normB = _numOps.Zero;

        for (int i = 0; i < _dimension; i++)
        {
            dotProduct = _numOps.Add(dotProduct, _numOps.Multiply(a[i], b[i]));
            normA = _numOps.Add(normA, _numOps.Square(a[i]));
            normB = _numOps.Add(normB, _numOps.Square(b[i]));
        }

        normA = _numOps.Sqrt(normA);
        normB = _numOps.Sqrt(normB);

        T denominator = _numOps.Multiply(normA, normB);

        if (_numOps.Equals(denominator, _numOps.Zero))
            return _numOps.Zero;

        return _numOps.Divide(dotProduct, denominator);
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
