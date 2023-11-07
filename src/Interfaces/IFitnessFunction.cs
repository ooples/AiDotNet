namespace AiDotNet.Interfaces;

public interface IFitnessFunction
{
    /// <summary>
    ///    /// The fitness function to use for the genetic algorithm.
    ///    /// </summary>
    ///    /// <param name="model">The model to evaluate.</param>
    ///    /// <param name="oosData">The out of sample data to use for evaluation.</param>
    ///    /// <returns>The fitness score.</returns>
    ///    double Evaluate(IModel model, Matrix<double> oosData);
}