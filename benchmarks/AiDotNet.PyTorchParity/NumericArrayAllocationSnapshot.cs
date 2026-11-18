using Newtonsoft.Json.Linq;

/// <summary>Diagnostic interpretation of the version2 primitive-array callback counters.</summary>
internal static class NumericArrayAllocationSnapshot
{
    internal const int MaximumRank = 32;
    internal const int BinsBegin = 26;
    internal static readonly string[] DTypes =
        ["bool", "char", "i1", "u1", "i2", "u2", "i4", "u4", "i8", "u8", "r4", "r8", "native_i", "native_u"];
    internal const int Length = BinsBegin + 14 * MaximumRank * 3;

    internal static JObject Describe(ulong[] values)
    {
        if (values.Length != Length || values[7] != 2 || values[8] != Length)
            throw new InvalidDataException("Primitive-array profiler ABI/length mismatch.");
        ulong count = 0, payload = 0, objectBytes = 0;
        var bins = new JArray();
        checked
        {
            for (int dtype = 0; dtype < DTypes.Length; dtype++)
                for (int rank = 1; rank <= MaximumRank; rank++)
                {
                    int index = BinsBegin + (dtype * MaximumRank + rank - 1) * 3;
                    count += values[index]; payload += values[index + 1]; objectBytes += values[index + 2];
                    if (values[index] != 0 || values[index + 1] != 0 || values[index + 2] != 0)
                        bins.Add(new JObject { ["dtype"] = DTypes[dtype], ["rank"] = rank, ["count"] = values[index],
                            ["payload_bytes"] = values[index + 1], ["full_object_bytes"] = values[index + 2] });
                }
            bool reconciled = count == values[10] && payload == values[11]
                && objectBytes == values[12] - Math.Min(values[12], values[17])
                && values[17] <= values[12];
            if (values[5] == 0 && values[25] == 0)
                reconciled &= values[9] + values[18] + values[20] == values[1]
                    && values[12] + values[19] + values[21] == values[2]
                    && values[22] == values[3] && values[23] == values[4]
                    && values[9] == values[10] + values[13] + values[14] + values[15] + values[16];
            bool bounded = values[6] == 1 && values[5] == 0 && values[25] == 0 && values[24] == 0 && reconciled;
            bool exactPrimitive = bounded && values[13] == 0 && values[14] == 0 && values[15] == 0
                && values[16] == 0 && values[22] == 0;
            return new JObject {
                ["coverage"] = "clr_primitive_array_callbacks_v2", ["abi"] = 2,
                ["legacy_prefix_atomic"] = true, ["snapshot"] = JArray.FromObject(values),
                ["primitive_candidate_count"] = values[9], ["valid_primitive_count"] = values[10],
                ["primitive_payload_bytes"] = values[11], ["primitive_full_object_bytes"] = values[12],
                ["reference_array_count"] = values[18], ["reference_array_full_object_bytes"] = values[19],
                ["unsupported_array_count"] = values[20], ["unsupported_array_full_object_bytes"] = values[21],
                ["classification_failures"] = values[22], ["classification_failure_full_object_bytes"] = values[23],
                ["failures"] = new JObject { ["legacy_size"] = values[5], ["size"] = values[25],
                    ["array_info"] = values[13], ["rank"] = values[14], ["payload_overflow"] = values[15],
                    ["payload_exceeds_object"] = values[16], ["counter_overflow"] = values[24],
                    ["counter_reconciliation"] = !reconciled },
                ["exact_primitive_payload"] = exactPrimitive, ["candidate_upper_bound_available"] = bounded,
                ["fresh_candidate_count_upper_bound"] = values[9] + values[20] + values[22],
                ["fresh_candidate_payload_bytes_upper_bound"] = values[11] + values[17] + values[21] + values[23],
                ["fresh_candidate_object_bytes_upper_bound"] = values[12] + values[21] + values[23],
                ["by_dtype_rank"] = bins,
                ["scope"] = "Fresh managed primitive-array payloads; bool/char included. Unsupported value-type and unknown-class objects are conservatively included in candidate bounds. Reference-element numeric storage requires a separate active dtype audit or the legacy all-array bound.",
                ["pool_borrow_coverage"] = false, ["native_storage_coverage"] = false,
                ["physical_gc_heap_coverage"] = false, ["gc_alignment_bytes_covered"] = false,
                ["acceptance_claim"] = false,
            };
        }
    }
}
