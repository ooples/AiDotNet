const SERVING_URL = import.meta.env.PUBLIC_SERVING_URL || 'https://YOUR_SERVING_URL';

export async function startCheckout(tier: string, interval: string = 'month') {
  const button = document.querySelector(`[data-tier="${tier}"]`) as HTMLButtonElement | null;
  if (button) {
    button.disabled = true;
    button.textContent = 'Redirecting...';
  }

  try {
    const response = await fetch(`${SERVING_URL}/api/checkout/session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: '',
        customerName: '',
        tier: tier,
        seats: 1,
        billingInterval: interval,
      }),
    });

    if (!response.ok) {
      throw new Error(`Checkout failed: ${response.status}`);
    }

    const data = await response.json();
    if (data.checkoutUrl) {
      window.location.href = data.checkoutUrl;
    } else {
      throw new Error('No checkout URL returned');
    }
  } catch (err) {
    console.error('Checkout error:', err);
    if (button) {
      button.disabled = false;
      button.textContent = 'Subscribe';
    }
    alert('Unable to start checkout. Please try again or contact support.');
  }
}
