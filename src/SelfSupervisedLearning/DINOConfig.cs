namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// DINO-specific configuration settings.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> DINO (self-DIstillation with NO labels) uses self-distillation
/// with centering and sharpening to learn emergent attention patterns in Vision Transformers.</para>
/// </remarks>
public class DINOConfig
{
    /// <summary>
    /// Gets or sets the student temperature.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.1</c></para>
    /// </remarks>
    public double? StudentTemperature { get; set; }

    /// <summary>
    /// Gets or sets the initial teacher temperature.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.04</c></para>
    /// <para>Teacher temperature is typically scheduled from 0.04 to 0.07.</para>
    /// </remarks>
    public double? TeacherTemperatureStart { get; set; }

    /// <summary>
    /// Gets or sets the final teacher temperature.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.07</c></para>
    /// </remarks>
    public double? TeacherTemperatureEnd { get; set; }

    /// <summary>
    /// Gets or sets the number of warmup epochs for teacher temperature.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>30</c></para>
    /// </remarks>
    public int? TeacherTemperatureWarmupEpochs { get; set; }

    /// <summary>
    /// Gets or sets the momentum for the centering mechanism.
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>0.9</c></para>
    /// </remarks>
    public double? CenterMomentum { get; set; }

    /// <summary>
    /// Gets or sets the number of local crops (small crops).
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>8</c></para>
    /// </remarks>
    public int? NumLocalCrops { get; set; }

    /// <summary>
    /// Gets or sets the number of global crops (large crops).
    /// </summary>
    /// <remarks>
    /// <para>Default: <c>2</c></para>
    /// </remarks>
    public int? NumGlobalCrops { get; set; }

    /// <summary>
    /// Gets the configuration as a dictionary.
    /// </summary>
    public IDictionary<string, object> GetConfiguration()
    {
        var config = new Dictionary<string, object>();
        if (StudentTemperature.HasValue) config["studentTemperature"] = StudentTemperature.Value;
        if (TeacherTemperatureStart.HasValue) config["teacherTemperatureStart"] = TeacherTemperatureStart.Value;
        if (TeacherTemperatureEnd.HasValue) config["teacherTemperatureEnd"] = TeacherTemperatureEnd.Value;
        if (CenterMomentum.HasValue) config["centerMomentum"] = CenterMomentum.Value;
        if (NumLocalCrops.HasValue) config["numLocalCrops"] = NumLocalCrops.Value;
        if (NumGlobalCrops.HasValue) config["numGlobalCrops"] = NumGlobalCrops.Value;
        return config;
    }
}
