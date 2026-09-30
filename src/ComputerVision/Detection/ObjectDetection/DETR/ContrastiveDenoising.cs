using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.DETR;

/// <summary>
/// The contrastive-denoising (CDN) queries of one training step (DINO Sec. 3.3; reference
/// <c>prepare_for_cdn</c>). Every target is copied into <see cref="Groups"/> groups, each holding a positive
/// copy (box noise below lambda) and a negative copy (between lambda and 2 lambda). Positives reconstruct
/// their target; negatives and padding are background.
/// </summary>
internal sealed class ContrastiveDenoisingPlan<T>
{
    public ContrastiveDenoisingPlan(int groups, int singlePad, int[] labels, double[] unsigmoidBoxes, bool[] occupied,
        DetectionTrainingBatch<T> targets, int[][] assignments)
    {
        Groups = groups;
        SinglePad = singlePad;
        Labels = labels;
        UnsigmoidBoxes = unsigmoidBoxes;
        Occupied = occupied;
        Targets = targets;
        Assignments = assignments;
    }

    /// <summary>Denoising groups (dn_number after the reference's rescaling).</summary>
    public int Groups { get; }

    /// <summary>Slots per copy: the largest target count in the batch.</summary>
    public int SinglePad { get; }

    /// <summary>Denoising queries per image: 2 x groups x single pad.</summary>
    public int PadSize => 2 * Groups * SinglePad;

    /// <summary>Noised label of every slot, <c>[batch * padSize]</c> (0 for padding).</summary>
    public int[] Labels { get; }

    /// <summary>Inverse-sigmoid noised box of every slot, <c>[batch * padSize * 4]</c> (0 for padding).</summary>
    public double[] UnsigmoidBoxes { get; }

    /// <summary>Whether each slot holds a target copy (padding slots are zero queries).</summary>
    public bool[] Occupied { get; }

    /// <summary>The targets each positive slot reconstructs: each image's targets repeated once per group.</summary>
    public DetectionTrainingBatch<T> Targets { get; }

    /// <summary>For each image and repeated target, the positive slot it is pinned to.</summary>
    public int[][] Assignments { get; }
}

/// <summary>
/// Contrastive denoising shared by the DETR family (DINO Sec. 3.3, RT-DETR): builds the noised positive and
/// negative query groups and their attention mask, following the reference <c>prepare_for_cdn</c>.
/// </summary>
internal static class ContrastiveDenoising<T>
{
    /// <summary>
    /// Builds the CDN queries for a batch (reference <c>prepare_for_cdn</c>). Returns null when no image has a
    /// target: the reference then adds no denoising queries and its denoising losses are zero.
    /// </summary>
    public static ContrastiveDenoisingPlan<T>? Plan(DetectionTrainingBatch<T> targets, int numClasses,
        int denoisingQueries, double labelNoiseRatio, double boxNoiseScale, Random random)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        int batch = targets.ImageCount;
        int singlePad = Enumerable.Range(0, batch).Max(b => targets[b].Count);
        if (singlePad == 0 || denoisingQueries <= 0) return null;
        int dnNumber = 2 * denoisingQueries;
        int groups = dnNumber >= 100 ? dnNumber / (2 * singlePad) : Math.Max(1, dnNumber);
        if (groups == 0) groups = 1;
        int pad = 2 * groups * singlePad;
        var labels = new int[batch * pad];
        var boxes = new double[batch * pad * 4];
        var occupied = new bool[batch * pad];

        // Reference order: every (copy r, concatenated target) entry, copy-major; even copies are positive.
        for (int r = 0; r < 2 * groups; r++)
        {
            bool negative = (r % 2) == 1;
            for (int b = 0; b < batch; b++)
            {
                for (int t = 0; t < targets[b].Count; t++)
                {
                    var gt = targets[b][t];
                    int slot = (b * pad) + (singlePad * r) + t;
                    occupied[slot] = true;
                    int label = gt.ClassId;
                    if (labelNoiseRatio > 0 && random.NextDouble() < labelNoiseRatio * 0.5) label = random.Next(numClasses);
                    labels[slot] = label;

                    double cx = ops.ToDouble(gt.CenterX), cy = ops.ToDouble(gt.CenterY), w = ops.ToDouble(gt.Width), h = ops.ToDouble(gt.Height);
                    double[] corners = { cx - (w / 2), cy - (h / 2), cx + (w / 2), cy + (h / 2) };
                    if (boxNoiseScale > 0)
                    {
                        double[] diff = { w / 2, h / 2, w / 2, h / 2 };
                        for (int c = 0; c < 4; c++)
                        {
                            double sign = random.Next(2) * 2.0 - 1.0;
                            double part = random.NextDouble() + (negative ? 1.0 : 0.0);
                            corners[c] = Math.Min(1.0, Math.Max(0.0, corners[c] + (part * sign * diff[c] * boxNoiseScale)));
                        }
                    }
                    double[] noised = { (corners[0] + corners[2]) / 2, (corners[1] + corners[3]) / 2, corners[2] - corners[0], corners[3] - corners[1] };
                    for (int c = 0; c < 4; c++) boxes[(slot * 4) + c] = DetrEmbeddings.InverseSigmoid(noised[c]);
                }
            }
        }

        // Positives reconstruct their target: group g's copy of target t sits at slot 2 * g * singlePad + t.
        var repeated = new DetectionTrainingTarget<T>[batch][];
        var assignments = new int[batch][];
        for (int b = 0; b < batch; b++)
        {
            int count = targets[b].Count;
            repeated[b] = new DetectionTrainingTarget<T>[groups * count];
            assignments[b] = new int[groups * count];
            for (int g = 0; g < groups; g++)
                for (int t = 0; t < count; t++)
                {
                    repeated[b][(g * count) + t] = targets[b][t];
                    assignments[b][(g * count) + t] = (2 * g * singlePad) + t;
                }
        }

        return new ContrastiveDenoisingPlan<T>(groups, singlePad, labels, boxes, occupied, new DetectionTrainingBatch<T>(repeated), assignments);
    }

    /// <summary>
    /// The reference CDN attention mask, as the engine's "may attend" mask. Matching queries cannot see any
    /// denoising query, and each denoising group sees only itself and the matching queries.
    /// </summary>
    public static Tensor<bool> AttendMask(int batch, int heads, int total, ContrastiveDenoisingPlan<T> plan)
    {
        int pad = plan.PadSize, group = 2 * plan.SinglePad;
        var mask = new Tensor<bool>(new[] { batch, heads, total, total });
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int q = 0; q < total; q++)
                    for (int key = 0; key < total; key++)
                    {
                        bool blocked;
                        if (q >= pad) blocked = key < pad;
                        else blocked = key < pad && (key / group) != (q / group);
                        mask[b, h, q, key] = !blocked;
                    }
        return mask;
    }
}
