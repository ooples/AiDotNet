namespace AiDotNet.Enums;

/// <summary>How a NeuS-style reconstruction places the samples along each ray.</summary>
public enum NeuSRaySampler
{
    /// <summary>
    /// NeuS's sampler (Wang et al. 2021, Sec. 3.4): uniform samples between the sphere's near and far points, then
    /// hierarchical up-sampling toward the zero level set.
    /// </summary>
    Hierarchical,

    /// <summary>
    /// instant-nsr-pl's nerfacc ray marching: equal steps through the scene box, skipping the cells an occupancy grid
    /// marks empty (Müller et al. 2022, Sec. 5.4; nerfacc's <c>OccupancyGrid</c> and <c>ray_marching</c>).
    /// </summary>
    OccupancyGrid,
}
