namespace AiDotNet.Interfaces;

public interface ISymbolicModel<T> : IFullModel<T>
{
    int Complexity { get; }
    void Fit(Matrix<T> X, Vector<T> y);
    T Evaluate(Vector<T> input);
    ISymbolicModel<T> Mutate(double mutationRate, INumericOperations<T> numOps);
    ISymbolicModel<T> Crossover(ISymbolicModel<T> other, double crossoverRate, INumericOperations<T> numOps);
    ISymbolicModel<T> Clone();
    int FeatureCount { get; }
    bool IsFeatureUsed(int featureIndex);
}