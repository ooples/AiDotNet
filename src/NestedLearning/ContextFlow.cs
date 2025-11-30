using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.NestedLearning;

/// <summary>
/// Implementation of Context Flow mechanism for nested learning.
/// Maintains distinct information pathways for each optimization level,
/// enabling deeper computational depth in learning components.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public class ContextFlow<T> : IContextFlow<T>
{
    private readonly int _numLevels;
    private readonly int _contextDimension;
    private readonly Vector<T>[] _contextStates;
    private readonly Matrix<T>[] _transformationMatrices;
    private Matrix<T>[] _compressionMatrices;
    private static readonly INumericOperations<T> _numOps = MathHelper.GetNumericOperations<T>();

    public ContextFlow(int contextDimension, int numLevels = 3)
    {
        _contextDimension = contextDimension;
        _numLevels = numLevels;

        _contextStates = new Vector<T>[numLevels];
        _transformationMatrices = new Matrix<T>[numLevels];
        _compressionMatrices = new Matrix<T>[numLevels];

        InitializeContextFlow();
    }

    private void InitializeContextFlow()
    {
        var random = RandomHelper.CreateSeededRandom(42);

        for (int i = 0; i < _numLevels; i++)
        {
            // Initialize context states to zero
            _contextStates[i] = new Vector<T>(_contextDimension);

            // Initialize transformation matrices with Xavier/Glorot initialization
            _transformationMatrices[i] = new Matrix<T>(_contextDimension, _contextDimension);
            _compressionMatrices[i] = new Matrix<T>(_contextDimension, _contextDimension);

            double scale = Math.Sqrt(2.0 / _contextDimension);
            T scaleFactor = _numOps.FromDouble(scale);

            for (int row = 0; row < _contextDimension; row++)
            {
                for (int col = 0; col < _contextDimension; col++)
                {
                    double randVal = (random.NextDouble() - 0.5) * 2.0;
                    _transformationMatrices[i][row, col] = _numOps.Multiply(
                        _numOps.FromDouble(randVal), scaleFactor);

                    randVal = (random.NextDouble() - 0.5) * 2.0;
                    _compressionMatrices[i][row, col] = _numOps.Multiply(
                        _numOps.FromDouble(randVal), scaleFactor);
                }
            }
        }
    }

    public Vector<T> PropagateContext(Vector<T> input, int currentLevel)
    {
        if (currentLevel < 0 || currentLevel >= _numLevels)
            throw new ArgumentException($"Invalid level: {currentLevel}");

        // Transform input through level-specific transformation
        var transformed = _transformationMatrices[currentLevel].Multiply(input);

        // Update context state with exponential moving average (momentum-like)
        T momentum = _numOps.FromDouble(0.9);
        T oneMinusMomentum = _numOps.Subtract(_numOps.One, momentum);

        var currentContext = _contextStates[currentLevel];
        var newContext = new Vector<T>(_contextDimension);

        for (int i = 0; i < _contextDimension; i++)
        {
            T momentumTerm = _numOps.Multiply(currentContext[i], momentum);
            T updateTerm = _numOps.Multiply(transformed[i], oneMinusMomentum);
            newContext[i] = _numOps.Add(momentumTerm, updateTerm);
        }

        _contextStates[currentLevel] = newContext;
        return newContext;
    }

    public Vector<T> ComputeContextGradients(Vector<T> upstreamGradient, int level)
    {
        if (level < 0 || level >= _numLevels)
            throw new ArgumentException($"Invalid level: {level}");

        // Backpropagate through transformation matrix
        var transposed = _transformationMatrices[level].Transpose();
        return transposed.Multiply(upstreamGradient);
    }

    public void UpdateFlow(Vector<T>[] gradients, T[] learningRates)
    {
        if (gradients.Length != _numLevels)
            throw new ArgumentException("Number of gradients must match number of levels");

        if (learningRates.Length != _numLevels)
            throw new ArgumentException("Number of learning rates must match number of levels");

        for (int i = 0; i < _numLevels; i++)
        {
            var contextVector = _contextStates[i];
            var gradVector = gradients[i];

            // Compute gradient for transformation matrix using outer product
            var matrixGrad = new Matrix<T>(_contextDimension, _contextDimension);

            for (int row = 0; row < _contextDimension; row++)
            {
                for (int col = 0; col < _contextDimension; col++)
                {
                    T grad = _numOps.Multiply(gradVector[row], contextVector[col]);
                    matrixGrad[row, col] = grad;
                }
            }

            // Update transformation matrix with gradient descent
            var scaled = matrixGrad.Multiply(learningRates[i]);
            _transformationMatrices[i] = _transformationMatrices[i].Subtract(scaled);
        }
    }

    public Vector<T> GetContextState(int level)
    {
        if (level < 0 || level >= _numLevels)
            throw new ArgumentException($"Invalid level: {level}");

        return _contextStates[level];
    }

    public Vector<T> CompressContext(Vector<T> context, int targetLevel)
    {
        if (targetLevel < 0 || targetLevel >= _numLevels)
            throw new ArgumentException($"Invalid target level: {targetLevel}");

        // Compress internal context flows as described in research
        // This enables deeper computational depth
        return _compressionMatrices[targetLevel].Multiply(context);
    }

    public void Reset()
    {
        for (int i = 0; i < _numLevels; i++)
        {
            _contextStates[i] = new Vector<T>(_contextDimension);
        }
    }

    public int NumberOfLevels => _numLevels;

    /// <summary>
    /// Gets the transformation matrices for inspection/debugging.
    /// </summary>
    public Matrix<T>[] GetTransformationMatrices() => _transformationMatrices;

    /// <summary>
    /// Gets the compression matrices for inspection/debugging.
    /// </summary>
    public Matrix<T>[] GetCompressionMatrices() => _compressionMatrices;
}
