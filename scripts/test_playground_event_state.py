"""Real PostgreSQL regression tests for the Playground webhook migration (requires Docker)."""
import concurrent.futures
import pathlib
import subprocess
import sys
import time
import unittest
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTAINER = "playground-stripe-tests-" + uuid.uuid4().hex[:12]
PROFILE = "00000000-0000-0000-0000-00000000abcd"
TOKEN = "00000000-0000-0000-0000-000000000001"
OTHER = "00000000-0000-0000-0000-000000000002"


def sql(query, check=True):
    result = subprocess.run(
        ["docker", "exec", "-i", CONTAINER, "psql", "-U", "postgres", "-At", "-v", "ON_ERROR_STOP=1"],
        input=query, text=True, capture_output=True, timeout=30)
    if check and result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.strip() if check else result


def claim(event, created, token=TOKEN):
    return sql(f"select public.claim_playground_stripe_event('{PROFILE}', '{event}', {created}, '{token}');")


def complete(status="active", token=TOKEN, subscription="sub_current", created=100):
    return sql(f"select public.complete_playground_stripe_event('{PROFILE}', '{token}', 'cus_test', '{subscription}', {created}, '{status}');")


def profile():
    return sql(f"select subscription_tier || '/' || subscription_status || '/' || coalesce(stripe_customer_id, '') from public.profiles where id='{PROFILE}';")


class EventStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sql("""
            create role anon;
            create role authenticated;
            create role service_role;
            create table public.profiles (
                id uuid primary key, subscription_tier text not null default 'free',
                subscription_status text not null default 'active', stripe_customer_id text,
                updated_at timestamptz not null default now()
            );
        """)
        for name in ["20261027000000_playground_stripe_event_state.sql",
                     "20261027010000_playground_stripe_ordering_and_tiers.sql"]:
            sql((ROOT / "website/supabase/migrations" / name).read_text())

    def setUp(self):
        sql("truncate public.playground_stripe_events, public.playground_stripe_state, public.profiles;")
        sql(f"insert into public.profiles(id) values ('{PROFILE}');")

    def test_duplicate_and_delayed_delivery_cannot_restore_canceled_profile(self):
        self.assertEqual(claim("evt_cancel", 200), "claimed")
        self.assertEqual(complete("canceled"), "applied")
        self.assertEqual(claim("evt_cancel", 200, OTHER), "duplicate")
        self.assertEqual(claim("evt_old_update", 199), "claimed")
        # The handler reads current Stripe state while holding the lease.
        self.assertEqual(complete("canceled"), "applied")
        self.assertEqual(profile(), "free/canceled/cus_test")
        self.assertEqual(sql("select count(*) from public.playground_stripe_events;"), "2")

    def test_delayed_replacement_checkout_after_old_cancellation(self):
        self.assertEqual(claim("evt_cancel_a", 300), "claimed")
        self.assertEqual(complete("canceled", subscription="sub_a", created=100), "applied")
        self.assertEqual(profile(), "free/canceled/cus_test")
        self.assertEqual(claim("evt_checkout_b", 299), "claimed")
        self.assertEqual(complete("active", subscription="sub_b", created=299), "applied")
        self.assertEqual(profile(), "pro/active/cus_test")
        self.assertEqual(sql("select subscription_id || '/' || last_event_created from public.playground_stripe_state;"), "sub_b/300")
        self.assertEqual(claim("evt_checkout_b", 299, OTHER), "duplicate")
        self.assertEqual(sql("select count(*) from public.playground_stripe_events;"), "2")

    def test_only_active_and_trialing_subscriptions_grant_pro(self):
        for i, status in enumerate(["active", "trialing", "past_due", "unpaid",
                                    "incomplete", "incomplete_expired", "paused", "canceled"]):
            with self.subTest(status=status):
                self.assertEqual(claim(f"evt_status_{i}", 200 + i), "claimed")
                self.assertEqual(complete(status), "applied")
                tier = "pro" if status in ("active", "trialing") else "free"
                expected_status = "active" if status == "trialing" else status
                self.assertEqual(profile(), f"{tier}/{expected_status}/cus_test")

    def test_concurrent_workers_have_one_owner_and_one_receipt(self):
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda token: claim("evt_same", 200, token), [TOKEN, OTHER]))
        self.assertEqual(sorted(results), ["busy", "claimed"])
        owner = [TOKEN, OTHER][results.index("claimed")]
        self.assertEqual(complete(token=owner), "applied")
        self.assertEqual(claim("evt_same", 200), "duplicate")
        self.assertEqual(sql("select count(*) from public.playground_stripe_events;"), "1")
        self.assertEqual(profile(), "pro/active/cus_test")

    def test_distinct_same_second_events_can_reconcile_current_state(self):
        self.assertEqual(claim("evt_active", 200), "claimed")
        self.assertEqual(complete(), "applied")
        self.assertEqual(claim("evt_past_due", 200, OTHER), "claimed")
        self.assertEqual(complete("past_due", OTHER), "applied")
        self.assertEqual(profile(), "free/past_due/cus_test")
        self.assertEqual(sql("select count(*) from public.playground_stripe_events;"), "2")

    def test_lease_expiry_fences_old_worker_and_allows_recovery(self):
        self.assertEqual(claim("evt_one", 200), "claimed")
        sql("update public.playground_stripe_state set lease_expires_at=clock_timestamp()-interval '1 second';")
        self.assertEqual(claim("evt_two", 201, OTHER), "claimed")
        failed = sql(f"select public.complete_playground_stripe_event('{PROFILE}', '{TOKEN}', 'cus_test', 'sub_current', 100, 'canceled');", check=False)
        self.assertNotEqual(failed.returncode, 0)
        sql(f"select public.release_playground_stripe_event('{PROFILE}', '{TOKEN}');")
        self.assertEqual(claim("evt_three", 202), "busy")
        self.assertEqual(complete(token=OTHER), "applied")
        self.assertEqual(profile(), "pro/active/cus_test")
        self.assertEqual(sql("select event_id from public.playground_stripe_events;"), "evt_two")

    def test_failed_profile_update_rolls_back_receipt_and_watermark(self):
        sql("alter table public.profiles add constraint test_reject_pro check (subscription_tier <> 'pro');")
        try:
            self.assertEqual(claim("evt_retry", 200), "claimed")
            failed = sql(f"select public.complete_playground_stripe_event('{PROFILE}', '{TOKEN}', 'cus_test', 'sub_current', 100, 'active');", check=False)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(sql("select count(*) from public.playground_stripe_events;"), "0")
            self.assertEqual(sql("select last_event_created from public.playground_stripe_state;"), "0")
            self.assertEqual(profile(), "free/active/")
        finally:
            sql("alter table public.profiles drop constraint test_reject_pro;")
        sql(f"select public.release_playground_stripe_event('{PROFILE}', '{TOKEN}');")
        self.assertEqual(claim("evt_retry", 200, OTHER), "claimed")
        self.assertEqual(complete(token=OTHER), "applied")
        self.assertEqual(profile(), "pro/active/cus_test")

    def test_old_subscription_cancellation_does_not_cancel_replacement(self):
        self.assertEqual(claim("evt_new", 200), "claimed")
        self.assertEqual(complete(subscription="sub_new", created=150), "applied")
        self.assertEqual(claim("evt_old_cancel", 201), "claimed")
        self.assertEqual(complete("canceled", subscription="sub_old", created=100), "stale")
        self.assertEqual(profile(), "pro/active/cus_test")
        self.assertEqual(sql("select subscription_id || '/' || last_event_created from public.playground_stripe_state;"), "sub_new/200")

    def test_missing_profile_cannot_be_acknowledged_as_processed(self):
        sql("delete from public.profiles;")
        failed = sql(f"select public.claim_playground_stripe_event('{PROFILE}', 'evt_missing', 200, '{TOKEN}');", check=False)
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(sql("select count(*) from public.playground_stripe_events;"), "0")

    def test_only_service_role_can_access_event_state(self):
        for role in ["anon", "authenticated"]:
            for query in [
                "select * from public.playground_stripe_events",
                "select * from public.playground_stripe_state",
                f"select public.claim_playground_stripe_event('{PROFILE}', 'evt_denied', 200, '{TOKEN}')",
                f"select public.complete_playground_stripe_event('{PROFILE}', '{TOKEN}', 'cus_test', 'sub_current', 100, 'active')",
                f"select public.release_playground_stripe_event('{PROFILE}', '{TOKEN}')",
            ]:
                with self.subTest(role=role, query=query):
                    self.assertNotEqual(sql(f"set role {role}; {query};", check=False).returncode, 0)
        result = sql(f"set role service_role; select public.claim_playground_stripe_event('{PROFILE}', 'evt_allowed', 200, '{TOKEN}');")
        self.assertTrue(result.endswith("claimed"), result)
        self.assertEqual(complete("trialing"), "applied")
        self.assertEqual(profile(), "pro/active/cus_test")


if __name__ == "__main__":
    try:
        subprocess.run(["docker", "run", "--detach", "--rm", "--name", CONTAINER,
                        "--env", "POSTGRES_PASSWORD=local-regression-only", "postgres:17-alpine"],
                       check=True, capture_output=True, text=True)
        for attempt in range(30):
            if subprocess.run(["docker", "exec", CONTAINER, "pg_isready", "-U", "postgres"], capture_output=True).returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("Test PostgreSQL did not start")
        unittest.main()
    finally:
        subprocess.run(["docker", "stop", CONTAINER], capture_output=True, timeout=30)
