do $$
begin
  if not exists (
    select 1 from pg_index i join pg_class c on c.oid = i.indexrelid
    where c.oid = to_regclass('public.idx_license_keys_subscription_product')
      and i.indrelid = 'public.license_keys'::regclass and i.indisunique and i.indisvalid and i.indisready
      and i.indnkeyatts = 2
      and pg_get_indexdef(i.indexrelid,1,true) = 'stripe_subscription_id'
      and pg_get_indexdef(i.indexrelid,2,true) = 'product'
      and pg_get_expr(i.indpred,i.indrelid) = '(stripe_subscription_id IS NOT NULL)'
  ) or not exists (
    select 1 from pg_index i where i.indexrelid = to_regclass('public.idx_license_keys_checkout_product')
      and i.indrelid = 'public.license_keys'::regclass
      and i.indisunique and i.indisvalid and i.indisready and i.indnkeyatts = 2
      and pg_get_indexdef(i.indexrelid,1,true) = 'stripe_checkout_session_id'
      and pg_get_indexdef(i.indexrelid,2,true) = 'product'
      and pg_get_expr(i.indpred,i.indrelid) = '(stripe_checkout_session_id IS NOT NULL)'
  ) or to_regclass('public.idx_license_keys_one_active_per_user_product_tier') is not null
    or to_regprocedure('public.sync_aidotnet_billing_profile(text,text,text,uuid)') is null
    or to_regprocedure('public.issue_aidotnet_community_license(uuid,text)') is null then
    raise exception 'Entitlement prerequisites are missing or invalid. Run the reviewed migrations before deploying billing functions.';
  end if;
  if not has_function_privilege('service_role','public.sync_aidotnet_billing_profile(text,text,text,uuid)','EXECUTE')
    or has_function_privilege('authenticated','public.sync_aidotnet_billing_profile(text,text,text,uuid)','EXECUTE')
    or has_function_privilege('anon','public.sync_aidotnet_billing_profile(text,text,text,uuid)','EXECUTE') then
    raise exception 'Billing mirror execution permissions are invalid. Run the reviewed migrations.';
  end if;
  if not has_function_privilege('service_role','public.issue_aidotnet_community_license(uuid,text)','EXECUTE')
    or has_function_privilege('authenticated','public.issue_aidotnet_community_license(uuid,text)','EXECUTE')
    or has_function_privilege('anon','public.issue_aidotnet_community_license(uuid,text)','EXECUTE') then
    raise exception 'Community issuance execution permissions are invalid. Run the reviewed migrations.';
  end if;
end $$;

do $$ begin
  if not exists (select 1 from pg_proc where oid='public.sync_aidotnet_billing_profile(text,text,text,uuid)'::regprocedure
    and proargnames=array['p_customer_id','p_fallback_tier','p_fallback_status','p_user_id'] and pronargdefaults=1
    and pg_get_expr(proargdefaults,0)='NULL::uuid')
    or not exists (select 1 from pg_proc where oid='public.issue_aidotnet_community_license(uuid,text)'::regprocedure
      and proargnames=array['p_user_id','p_license_key'])
    or to_regclass('public.billing_checkout_intents') is null
    or to_regclass('public.billing_customers') is null
    or to_regclass('public.billing_checkout_bindings') is null
    or to_regprocedure('public.bind_aidotnet_checkout(uuid,text,text)') is null then
    raise exception 'RPC argument contract or verified billing ownership schema is missing. Run the reviewed migrations.';
  end if;
  if not has_function_privilege('service_role','public.bind_aidotnet_checkout(uuid,text,text)','EXECUTE')
    or has_function_privilege('authenticated','public.bind_aidotnet_checkout(uuid,text,text)','EXECUTE')
    or has_function_privilege('anon','public.bind_aidotnet_checkout(uuid,text,text)','EXECUTE') then
    raise exception 'Checkout binding execution permissions are invalid. Run the reviewed migrations.';
  end if;
end $$;
