using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.LinearAlgebra;

namespace AiDotNet.Interfaces;

/// <summary>
/// Non-generic base interface for all models providing common properties and methods.
/// </summary>
public interface IModel
{
    /// <summary>
    /// Gets the type of the model.
    /// </summary>
    ModelType Type { get; }
    
    /// <summary>
    /// Gets the statistics of the model.
    /// </summary>
    ModelStats<double, Matrix<double>, Vector<double>> GetStats();
}