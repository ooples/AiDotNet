namespace AiDotNet.Enums;

/// <summary>
/// Defines optimization levels for model performance tuning.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Like choosing quality settings in a video game - higher optimization 
/// means more effort to make the model faster or smaller, but might take longer to prepare.
/// </para>
/// </remarks>
public enum OptimizationLevel
{
    /// <summary>
    /// No optimization applied.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The model runs as-is without any speed or size improvements - 
    /// fastest to deploy but may be slow or large.
    /// </remarks>
    None,

    /// <summary>
    /// Basic optimization for slight improvements.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Quick and easy optimizations that give small speed boosts without 
    /// much effort - like cleaning up obvious inefficiencies.
    /// </remarks>
    Basic,

    /// <summary>
    /// Level 1 optimization (alias for Basic).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Same as Basic optimization - provides quick improvements.
    /// </remarks>
    O1 = Basic,

    /// <summary>
    /// Moderate optimization balancing speed and effort.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> A good middle ground - applies common optimizations that significantly 
    /// improve performance without too much complexity.
    /// </remarks>
    Moderate,

    /// <summary>
    /// Aggressive optimization for maximum performance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Pulls out all the stops to make the model as fast as possible - 
    /// may take a long time to optimize and could slightly reduce accuracy.
    /// </remarks>
    Aggressive,

    /// <summary>
    /// Optimization focused on inference speed.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Makes the model predict faster - important when you need quick 
    /// responses, like real-time video processing.
    /// </remarks>
    SpeedFocused,

    /// <summary>
    /// Optimization focused on memory usage.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Makes the model use less memory - important for devices with 
    /// limited RAM, like phones or embedded systems.
    /// </remarks>
    MemoryFocused,

    /// <summary>
    /// Optimization focused on model size.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Makes the model file smaller - important for downloading to devices 
    /// or fitting in limited storage.
    /// </remarks>
    SizeFocused,

    /// <summary>
    /// Optimization focused on power consumption.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Makes the model use less battery power - crucial for mobile devices 
    /// and IoT sensors that run on batteries.
    /// </remarks>
    PowerFocused,

    /// <summary>
    /// Optimization for specific hardware.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Tunes the model to run best on particular chips or devices - like 
    /// optimizing specifically for iPhones or NVIDIA GPUs.
    /// </remarks>
    HardwareSpecific,

    /// <summary>
    /// Balanced optimization across all metrics.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Tries to improve everything a bit - speed, size, and memory usage 
    /// all get some attention without focusing on just one.
    /// </remarks>
    Balanced,

    /// <summary>
    /// Adaptive optimization based on runtime.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> The model adjusts its optimization strategy while running - like 
    /// switching to power-saving mode when battery is low.
    /// </remarks>
    Adaptive,

    /// <summary>
    /// Profile-guided optimization.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Watches how the model is actually used, then optimizes the parts 
    /// that matter most - like a coach focusing training on weak spots.
    /// </remarks>
    ProfileGuided,

    /// <summary>
    /// Custom optimization strategy.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Allows you to define your own optimization approach for special requirements.
    /// </remarks>
    Custom
}