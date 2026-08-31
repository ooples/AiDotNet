using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the NeuralTuringMachine.
/// </summary>
public class NeuralTuringMachineOptions : NeuralNetworkOptions
{
    private double _learningRate = 1e-4;
    private double _rmsPropDecay = 0.95;
    private double _rmsPropMomentum = 0.9;
    private double _rmsPropEpsilon = 1e-4;
    private double _gradientClipValue = 10.0;

    /// <summary>
    /// Gets or sets the RMSProp learning rate. The paper uses 1e-4 for the copy,
    /// repeat-copy, and associative-recall experiments.
    /// </summary>
    public double LearningRate
    {
        get => _learningRate;
        set => _learningRate = RequirePositiveFinite(value, nameof(LearningRate));
    }

    /// <summary>
    /// Gets or sets the decay used by the centered RMSProp running averages.
    /// The RMSProp formulation referenced by the NTM paper uses 0.95.
    /// </summary>
    public double RmsPropDecay
    {
        get => _rmsPropDecay;
        set
        {
            if (double.IsNaN(value) || double.IsInfinity(value) || value < 0.0 || value >= 1.0)
                throw new ArgumentOutOfRangeException(nameof(RmsPropDecay), value, "RMSProp decay must be finite and in [0, 1).");
            _rmsPropDecay = value;
        }
    }

    /// <summary>
    /// Gets or sets the RMSProp velocity coefficient. Section 4.6 of the NTM
    /// paper specifies momentum 0.9 for every experiment.
    /// </summary>
    public double RmsPropMomentum
    {
        get => _rmsPropMomentum;
        set
        {
            if (double.IsNaN(value) || double.IsInfinity(value) || value < 0.0 || value >= 1.0)
                throw new ArgumentOutOfRangeException(nameof(RmsPropMomentum), value, "RMSProp momentum must be finite and in [0, 1).");
            _rmsPropMomentum = value;
        }
    }

    /// <summary>
    /// Gets or sets the stabilizer inside the centered RMSProp denominator.
    /// The referenced Graves RMSProp formulation uses 1e-4.
    /// </summary>
    public double RmsPropEpsilon
    {
        get => _rmsPropEpsilon;
        set => _rmsPropEpsilon = RequirePositiveFinite(value, nameof(RmsPropEpsilon));
    }

    /// <summary>
    /// Gets or sets the absolute element-wise gradient limit. The NTM paper
    /// clips every gradient component to (-10, 10).
    /// </summary>
    public double GradientClipValue
    {
        get => _gradientClipValue;
        set => _gradientClipValue = RequirePositiveFinite(value, nameof(GradientClipValue));
    }

    private static double RequirePositiveFinite(double value, string propertyName)
    {
        if (double.IsNaN(value) || double.IsInfinity(value) || value <= 0.0)
            throw new ArgumentOutOfRangeException(propertyName, value, $"{propertyName} must be positive and finite.");
        return value;
    }
}
