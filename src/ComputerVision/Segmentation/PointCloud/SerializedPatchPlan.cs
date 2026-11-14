namespace AiDotNet.ComputerVision.Segmentation.PointCloud;

/// <summary>
/// How the serialized points of a batch are cut into attention patches, grouped by patch length so
/// each group runs as one batched matrix multiply.
/// </summary>
/// <remarks>
/// <para>
/// Reproduces Pointcept's flash-attention path (<c>get_padding_and_inverse</c> with
/// <c>cu_seqlens</c>), which is PTv3's default:
/// </para>
/// <list type="bullet">
/// <item>A cloud with at most <c>K</c> points is one patch of its own length.</item>
/// <item>A larger cloud is cut into consecutive patches of <c>K</c> serialized points. When <c>K</c>
/// does not divide its size, Pointcept pads the last patch by repeating the points just before it,
/// so the last patch is exactly the cloud's final <c>K</c> serialized points; only the points not
/// already covered by an earlier patch take their output from it.</item>
/// </list>
/// <para>All indices are <see cref="int"/>; nothing here touches the network's numeric type.</para>
/// </remarks>
internal sealed class SerializedPatchPlan
{
    private SerializedPatchPlan(IReadOnlyList<SerializedPatchGroup> groups, int[] inverse)
    {
        Groups = groups;
        Inverse = inverse;
    }

    /// <summary>Patch groups, ordered by patch length.</summary>
    internal IReadOnlyList<SerializedPatchGroup> Groups { get; }

    /// <summary>For every point, its row in the concatenation of all groups' outputs.</summary>
    internal int[] Inverse { get; }

    internal static SerializedPatchPlan Build(int[] order, int[] batch, int batchCount, int patchSize)
    {
        if (patchSize <= 0) throw new ArgumentOutOfRangeException(nameof(patchSize), patchSize, "Patch size must be positive.");
        int count = order.Length;
        var perCloud = new int[batchCount];
        for (int i = 0; i < count; i++) perCloud[batch[i]]++;

        var gathers = new SortedDictionary<int, List<int>>();
        // Owner rows are first recorded as (patch length, row within that length's group).
        var ownerLength = new int[count];
        var ownerRow = new int[count];

        void AddPatch(int start, int length, int firstOwner)
        {
            if (!gathers.TryGetValue(length, out var list))
            {
                list = new List<int>();
                gathers[length] = list;
            }

            int baseRow = list.Count;
            for (int t = 0; t < length; t++)
            {
                int point = order[start + t];
                list.Add(point);
                if (t < firstOwner) continue;
                ownerLength[point] = length;
                ownerRow[point] = baseRow + t;
            }
        }

        int offset = 0;
        for (int b = 0; b < batchCount; b++)
        {
            int n = perCloud[b];
            if (n == 0) continue;
            if (n <= patchSize)
            {
                AddPatch(offset, n, 0);
            }
            else
            {
                int full = n / patchSize, remainder = n % patchSize;
                for (int j = 0; j < full; j++) AddPatch(offset + j * patchSize, patchSize, 0);
                if (remainder > 0) AddPatch(offset + n - patchSize, patchSize, patchSize - remainder);
            }

            offset += n;
        }

        var groups = new List<SerializedPatchGroup>(gathers.Count);
        var groupBase = new Dictionary<int, int>();
        int rows = 0;
        foreach (var pair in gathers)
        {
            groupBase[pair.Key] = rows;
            groups.Add(new SerializedPatchGroup(pair.Key, pair.Value.ToArray()));
            rows += pair.Value.Count;
        }

        var inverse = new int[count];
        for (int i = 0; i < count; i++) inverse[i] = groupBase[ownerLength[i]] + ownerRow[i];
        return new SerializedPatchPlan(groups, inverse);
    }
}

/// <summary>Patches of one length: <see cref="Gather"/> lists point indices patch after patch.</summary>
internal sealed class SerializedPatchGroup
{
    internal SerializedPatchGroup(int patchLength, int[] gather)
    {
        PatchLength = patchLength;
        Gather = gather;
    }

    internal int PatchLength { get; }

    internal int[] Gather { get; }

    internal int PatchCount => Gather.Length / PatchLength;
}
