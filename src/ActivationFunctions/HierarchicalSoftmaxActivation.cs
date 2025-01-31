namespace AiDotNet.ActivationFunctions;

public class HierarchicalSoftmaxActivation<T> : ActivationFunctionBase<T>
{
    private readonly int _numClasses;
    private readonly int _treeDepth;
    private readonly Matrix<T> _nodeWeights;

    public HierarchicalSoftmaxActivation(int numClasses)
    {
        _numClasses = numClasses;
        _treeDepth = (int)Math.Ceiling(MathHelper.Log2(numClasses));
        _nodeWeights = new Matrix<T>(_treeDepth, numClasses);
        InitializeWeights();
    }

    protected override bool SupportsScalarOperations() => false;

    public override Vector<T> Activate(Vector<T> input)
    {
        Vector<T> output = new Vector<T>(_numClasses);
        for (int i = 0; i < _numClasses; i++)
        {
            output[i] = ComputePathProbability(input, i);
        }

        return output;
    }

    public override Matrix<T> Derivative(Vector<T> input)
    {
        Matrix<T> jacobian = new Matrix<T>(_numClasses, input.Length);

        for (int i = 0; i < _numClasses; i++)
        {
            Vector<T> pathDerivative = ComputePathDerivative(input, i);
            jacobian.SetRow(i, pathDerivative);
        }

        return jacobian;
    }

    private Vector<T> ComputePathDerivative(Vector<T> input, int classIndex)
    {
        Vector<T> derivative = new Vector<T>(input.Length);
        int node = 1;

        for (int depth = 0; depth < _treeDepth; depth++)
        {
            Vector<T> nodeWeights = _nodeWeights.GetRow(depth);
            T nodeOutput = MathHelper.Sigmoid(input.DotProduct(nodeWeights));
            bool goRight = (classIndex & (1 << (_treeDepth - depth - 1))) != 0;

            T multiplier = goRight ? nodeOutput : NumOps.Subtract(NumOps.One, nodeOutput);
            T derivativeFactor = NumOps.Multiply(nodeOutput, NumOps.Subtract(NumOps.One, nodeOutput));

            if (goRight)
            {
                derivative = derivative.Add(nodeWeights.Multiply(derivativeFactor));
            }
            else
            {
                derivative = derivative.Subtract(nodeWeights.Multiply(derivativeFactor));
            }

            node = node * 2 + (goRight ? 1 : 0);
            if (node >= _numClasses) break;
        }

        return derivative;
    }

    private void InitializeWeights()
    {
        Random random = new Random();
        for (int i = 0; i < _treeDepth; i++)
        {
            for (int j = 0; j < _numClasses; j++)
            {
                _nodeWeights[i, j] = NumOps.FromDouble(random.NextDouble() - 0.5);
            }
        }
    }

    private T ComputePathProbability(Vector<T> input, int classIndex)
    {
        T probability = NumOps.One;
        int node = 1;
        for (int depth = 0; depth < _treeDepth; depth++)
        {
            T nodeOutput = MathHelper.Sigmoid(input.DotProduct(_nodeWeights.GetRow(depth)));
            bool goRight = (classIndex & (1 << (_treeDepth - depth - 1))) != 0;
            probability = NumOps.Multiply(probability, goRight ? nodeOutput : NumOps.Subtract(NumOps.One, nodeOutput));
            node = node * 2 + (goRight ? 1 : 0);
            if (node >= _numClasses) break;
        }

        return probability;
    }
}