namespace AiDotNet.Enums;

/// <summary>
/// Specifies the type of spiking neuron model to use in neuromorphic computing simulations.
/// </summary>
/// <remarks>
/// <b>For Beginners:</b> Spiking neurons are AI components that work more like real brain cells.
/// 
/// Traditional AI neurons output continuous values (like 0.7), but spiking neurons work with
/// discrete "spikes" or pulses of activity (like a real neuron firing). This makes them more
/// biologically realistic and potentially more efficient for certain tasks.
/// 
/// Think of regular AI neurons as light bulbs with dimmers that can be set to any brightness,
/// while spiking neurons are more like light bulbs that either flash brightly or stay off.
/// 
/// Different spiking neuron types represent different mathematical models of how real neurons work,
/// with varying levels of biological accuracy and computational complexity.
/// </remarks>
public enum SpikingNeuronType
{
    /// <summary>
    /// A simplified neuron model that accumulates input and "leaks" voltage over time.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is like a leaky bucket collecting water (electrical charge).
    /// 
    /// How it works:
    /// 1. The neuron collects incoming signals (like water filling a bucket)
    /// 2. The bucket slowly leaks over time (the "leaky" part)
    /// 3. When the water level reaches a certain height, the bucket tips over (neuron fires)
    /// 4. After firing, the bucket is emptied and starts collecting again
    /// 
    /// This model is:
    /// - Computationally efficient (fast to simulate)
    /// - Simple to understand and implement
    /// - Good for large-scale neural networks
    /// 
    /// It captures the basic behavior of real neurons while being much simpler than more detailed models.
    /// </remarks>
    LeakyIntegrateAndFire,

    /// <summary>
    /// A basic neuron model that accumulates input until reaching a threshold, then fires.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is like a bucket collecting water without any leaks.
    /// 
    /// How it works:
    /// 1. The neuron collects incoming signals (like water filling a bucket)
    /// 2. When the water level reaches a certain height, the bucket tips over (neuron fires)
    /// 3. After firing, the bucket is emptied and starts collecting again
    /// 
    /// The key difference from LeakyIntegrateAndFire is that this model doesn't have any "leak" -
    /// once charge is added, it stays there until the neuron fires.
    /// 
    /// This model is:
    /// - The simplest spiking neuron model
    /// - Very computationally efficient
    /// - Less biologically accurate than other models
    /// 
    /// It's good for educational purposes and very basic simulations.
    /// </remarks>
    IntegrateAndFire,

    /// <summary>
    /// A computationally efficient model that can reproduce many behaviors of biological neurons.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This model strikes a balance between biological realism and computational efficiency.
    /// 
    /// Named after Eugene Izhikevich who developed it, this model can simulate many different
    /// firing patterns seen in real neurons (like bursting, chattering, or regular spiking)
    /// while being much faster to compute than fully detailed models.
    /// 
    /// Think of it like a sophisticated light switch that can be programmed to blink in
    /// different patterns that closely resemble real brain activity.
    /// 
    /// This model is:
    /// - More biologically realistic than the simpler models
    /// - Still computationally efficient
    /// - Able to reproduce many different neural firing patterns
    /// 
    /// It's popular for large-scale brain simulations where both biological realism and
    /// computational efficiency are important.
    /// </remarks>
    Izhikevich,

    /// <summary>
    /// A detailed biophysical model that accurately represents ion channel dynamics in neurons.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is the most biologically accurate model, but also the most complex.
    /// 
    /// Named after Alan Hodgkin and Andrew Huxley who won a Nobel Prize for this work,
    /// this model precisely describes how ions flow through channels in the neuron's membrane.
    /// 
    /// Think of it like having a detailed engineering blueprint of a neuron that models
    /// all the important chemical and electrical processes happening inside.
    /// 
    /// This model is:
    /// - Extremely biologically accurate
    /// - Computationally intensive (slow to simulate)
    /// - Able to capture subtle details of neural behavior
    /// 
    /// It's primarily used in neuroscience research when biological accuracy is more important
    /// than computational efficiency.
    /// </remarks>
    HodgkinHuxley,

    /// <summary>
    /// A model that combines exponential spike generation with adaptive threshold mechanisms.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This model adds adaptability to neuron behavior.
    /// 
    /// The "Adaptive" part means the neuron can change its sensitivity based on recent activity.
    /// The "Exponential" part refers to how quickly the neuron responds when close to firing.
    /// 
    /// Think of it like a smart thermostat that becomes less sensitive after detecting several
    /// temperature changes, preventing it from overreacting to small fluctuations.
    /// 
    /// This model is:
    /// - More biologically realistic than basic models
    /// - Able to capture adaptation behaviors (neurons getting "tired" after firing a lot)
    /// - Moderately computationally efficient
    /// 
    /// It's useful for simulations where you need more realistic neural behavior than simple
    /// models provide, but can't afford the computational cost of the most detailed models.
    /// </remarks>
    AdaptiveExponential
}
