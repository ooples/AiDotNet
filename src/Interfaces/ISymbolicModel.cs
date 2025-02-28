namespace AiDotNet.Interfaces;

public interface ISymbolicModel<T> : IFullModel<T>
{
    int Complexity { get; }
    void Fit(Matrix<T> X, Vector<T> y);
    T Evaluate(Vector<T> input);
    ISymbolicModel<T> Mutate(double mutationRate);
    ISymbolicModel<T> Crossover(ISymbolicModel<T> other, double crossoverRate);
    ISymbolicModel<T> Copy();
    int FeatureCount { get; }
    bool IsFeatureUsed(int featureIndex);
    Vector<T> Coefficients { get; }
    ISymbolicModel<T> UpdateCoefficients(Vector<T> newCoefficients);
}