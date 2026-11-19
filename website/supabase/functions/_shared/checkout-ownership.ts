import type Stripe from "https://esm.sh/stripe@14.14.0?target=deno";
import type { SupabaseClient } from "https://esm.sh/@supabase/supabase-js@2.39.3";

export function checkoutCustomer(session: Stripe.Checkout.Session): string | null {
  return typeof session.customer === "string" ? session.customer : session.customer?.id ?? null;
}

/** Public Payment Link fields identify an opaque authenticated intent, never an account ID. */
export async function checkoutOwner(client: SupabaseClient, session: Stripe.Checkout.Session): Promise<string | null> {
  const { data: binding, error: bindingError } = await client.from("billing_checkout_bindings")
    .select("user_id,customer_id").eq("session_id", session.id).maybeSingle();
  if (bindingError) throw bindingError;
  if (binding) {
    if (binding.customer_id !== checkoutCustomer(session)) throw new Error("Checkout customer does not match its verified binding");
    return binding.user_id;
  }
  const reference = session.client_reference_id;
  if (!reference || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(reference)) return null;
  const { data: intent, error } = await client.from("billing_checkout_intents")
    .select("user_id,tier,created_at,expires_at").eq("id", reference).maybeSingle();
  if (error) throw error;
  if (!intent) return null;
  // Stripe's signed creation time permits delayed payment/replay after intent expiry.
  const created = session.created * 1000;
  if (!Number.isFinite(created) || created < new Date(intent.created_at).getTime() - 1000 || created > new Date(intent.expires_at).getTime()) return null;
  // Fulfillment derives the purchased tier from Stripe, not from this intent.
  return intent.user_id;
}

export async function rememberCheckoutOwner(client: SupabaseClient, userId: string, session: Stripe.Checkout.Session) {
  const { error } = await client.rpc("bind_aidotnet_checkout", {
    p_user_id: userId, p_session_id: session.id, p_customer_id: checkoutCustomer(session),
  });
  if (error) throw error;
}

// Legacy profile links can be ambiguous. Only this service-owned ledger grants access;
// a stale preference must not prevent access to the user's other verified customers.
// Resolve an unowned legacy link through claim_checkout's verified Stripe session/email
// and immutable binding RPC, or support review. Never infer ownership from the profile.
export async function ownedBillingCustomers(client: SupabaseClient, userId: string): Promise<string[]> {
  const { data, error } = await client.from("billing_customers").select("customer_id").eq("user_id", userId);
  if (error) throw error;
  const customers = (data ?? []).map((row: { customer_id: string }) => row.customer_id);
  return customers;
}
