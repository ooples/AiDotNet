import { expect, type Locator, type Page } from '@playwright/test';

/** Reach an overflowed dialog action through actual phone touch gestures. */
export async function touchScrollIntoView(page: Page, panel: Locator, target: Locator): Promise<number> {
  const session=await page.context().newCDPSession(page);
  let gestures=0;
  try {
    for(let i=0;i<8;i++) {
      try { await expect(target).toBeInViewport({ratio:1,timeout:100}); break; }
      catch { /* swipe the dialog to reach its offscreen action */ }
      const box=await panel.boundingBox();
      if(!box) throw new Error('Dialog has no touch target');
      const height=page.viewportSize()!.height;
      const x=box.x+box.width-20, startY=Math.min(box.y+box.height-35,height-40);
      const distance=Math.min(300,startY-Math.max(box.y+25,40));
      await session.send('Input.dispatchTouchEvent',{type:'touchStart',touchPoints:[{x,y:startY}]});
      for(let step=1;step<=12;step++) {
        await session.send('Input.dispatchTouchEvent',{type:'touchMove',touchPoints:[{x,y:startY-distance*step/12}]});
        await page.waitForTimeout(16); // Gesture frames, not a substitute for the viewport assertion.
      }
      await session.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});
      gestures++;
    }
    await expect(target).toBeInViewport({ratio:1});
    return gestures;
  } finally { await session.detach(); }
}
