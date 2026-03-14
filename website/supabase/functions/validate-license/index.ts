import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

interface ValidateRequest {
  license_key: string;
  machine_id_hash: string;
  hostname?: string;
  os_description?: string;
}

serve(async (req: Request) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return new Response(
      JSON.stringify({ valid: false, error: "method_not_allowed" }),
      { status: 405, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }

  try {
    const body: ValidateRequest = await req.json();

    if (!body.license_key || !body.machine_id_hash) {
      return new Response(
        JSON.stringify({
          valid: false,
          error: "missing_fields",
          message: "license_key and machine_id_hash are required.",
        }),
        { status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    // Create Supabase client with service role key for RPC access
    const supabaseUrl = Deno.env.get("SUPABASE_URL") ?? "";
    const supabaseServiceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
    const supabase = createClient(supabaseUrl, supabaseServiceKey);

    // Call the validate_license_key database function
    const { data, error } = await supabase.rpc("validate_license_key", {
      p_license_key: body.license_key,
      p_machine_id_hash: body.machine_id_hash,
      p_hostname: body.hostname ?? null,
      p_os_description: body.os_description ?? null,
    });

    if (error) {
      console.error("RPC error:", error);
      return new Response(
        JSON.stringify({
          valid: false,
          error: "server_error",
          message: "License validation failed. Please try again.",
        }),
        { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
      );
    }

    const result = data as Record<string, unknown>;
    const httpStatus = result.valid ? 200 : 403;

    return new Response(JSON.stringify(result), {
      status: httpStatus,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("Unexpected error:", err);
    return new Response(
      JSON.stringify({
        valid: false,
        error: "server_error",
        message: "An unexpected error occurred.",
      }),
      { status: 500, headers: { ...corsHeaders, "Content-Type": "application/json" } }
    );
  }
});
