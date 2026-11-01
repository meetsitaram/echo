# Reference images: from the web to shapes

The look is only as good as the pictures you measure. echo's first proportions came from a small, low-resolution
frame: the head was 4 cm too tall and the arm 4-8 cm too low. Its "black head" turned out, in a close-up, to be a
grey knit hood with a black visor. Spend the time here.

## 1. Find the sources

- The maker's product page: published specs (height, mass, payload, runtime, speed) and the hero images.
- The launch or reveal video: usually the only place the robot is seen whole, from several sides, standing still.
- News and press pages from the maker: more stills, and sometimes the battery, hands or charging details.
- Write down each URL and the date you read it. They go in `spec/targets.json` (`_about`) and in your sources list.

echo used figure.ai's Figure 03 product page (`figure.ai/figure`) and its reveal post (`figure.ai/news/introducing-figure-03`).

## 2. Capture frames from a video, at exact times

Use a browser the agent controls (Claude's built-in browser, or Chrome through its extension) on the maker's own page.
Seek the page's video player with JavaScript, enlarge the player to fill the window, and take a screenshot.

1. Open the page and find the video. Count the elements with `document.querySelectorAll('video')`, and log each
   one's `duration` and on-screen rectangle to tell them apart.
2. Seek and hold. Players often resume by themselves, so pause on every `play` and wait for `seeked`:

   ```js
   const v = document.querySelectorAll('video')[1];       // the one you identified
   v.loop = false; v.autoplay = false; v.muted = true;
   v.onplay = () => { if (window.__hold) v.pause(); };
   window.__hold = true;
   window.__seek = async (t) => {
     v.pause();
     await new Promise(res => { v.onseeked = () => res(); v.currentTime = t; });
     v.pause();
     await new Promise(r => setTimeout(r, 400));          // let the frame paint
     return v.currentTime;
   };
   await window.__seek(21.5);
   ```

3. Make the frame as large as the window allows: fix the player over the page and hide headers.

   ```js
   Object.assign(v.style, {position: 'fixed', zIndex: 99999, left: '0', top: '0', width: innerWidth + 'px',
                           height: innerHeight + 'px', maxWidth: 'none', objectFit: 'contain', background: '#fff'});
   document.querySelectorAll('header, nav').forEach(e => e.style.display = 'none');
   ```

4. Scrub to choose: seek to a dozen times and take small screenshots to find the views. Then capture the chosen
   times at full size. Pausing exactly where you want also catches the robot standing still.
5. Save each frame as `reports/compare/<reference>_<view>.jpg` (side, 3/4, front). Note its time code in
   `spec/targets.json`; echo's are side 0:21.5, 3/4 0:24.5 and front 1:50.

Drawing the video to a canvas and exporting it usually fails on a maker's site: a cross-origin video "taints" the
canvas. The screenshot route always works. For stills, take the largest image the page offers (the biggest
`srcset` entry) rather than the thumbnail.

Only use what the page shows publicly: don't bypass downloads that are blocked, DRM or logins. The frames are the
maker's copyright, so keep them local and out of the repo (`.gitignore` them from the first commit). Measure from
them, credit them, and never publish them or anything cut from them.

## 3. Choose the frames that measure well

- **Heights from level views only.** A camera above the robot lifts every top surface: echo's front frame, shot from
  above, put the shoulders about 4 cm higher than the side and 3/4 frames did.
- **The whole robot, standing.** Mid-stride frames make the legs read short. If a frame cuts the robot off,
  scale it from two landmarks you can see (echo's front frame uses the chin and the knee: `FRAME_BOX` in
  `tools/compare.py`).
- **Widths from the front, depths from the side.** Measure each section in the view it is seen in. An arm's depth
  comes from the side, its width from the front.
- **Fit the camera of a 3/4 frame before comparing.** Render your model at several angles and keep the one whose
  silhouette widths match. echo's "3/4" was 60° from the front, not 40°: at 40° every width read 10 cm too wide.

## 4. Turn pictures into numbers

`python -I tools/compare.py` does the routine part.
- It finds the robot in each frame against the background, scales it to the published height, and puts it beside
  your render at the same height.
- It draws every target height as a line across both pictures, so anything off its line is off the proportions.
- It writes a table of each landmark against your model's own geometry (not against a detector run on the render),
  and close-up crops of each region side by side.

For a section the tool does not measure, scan rows of the silhouette at the heights you need: the width of the dark
pixels in a row, times metres per pixel. Write each landmark to `spec/targets.json` with its value, tolerance and
source frame. The `targets` check then holds the model to it.

## 5. Turn numbers into shapes

- **Heights become anchors** (`HZ`, `KZ`, `EZ` ... at the top of the spec), and parts are placed relative to them.
- **Sections become loft sections.** At each height, the side depth gives `a`, the front width gives `b`, and the
  centre offsets give `cx`, `cy`. The squareness of the outline gives `n`: 2 is an ellipse; 2.4-3 is a soft box,
  which most covers are.
- **Silhouettes become section stacks.** Sample the outline every few centimetres where the shape changes (where a
  torso tapers to the waist, or a calf swells), and fewer where it doesn't.
- **Details come from close-ups.** Zoom into each region before modelling it: the head and visor, the shoulder cap,
  the elbow disc, the hips, the knee pads, the shoes. Figure's sneaker (outsole with toe spring, a stripe to the
  heel over a window, an angled collar) was invisible at full-body scale.

## 6. Compare, and let the user judge

Render before and after at the same scale beside the reference, in each view and in close-up crops. Landmarks can
all be in tolerance while the shape still reads wrong ("a boxy butt", "a bean for a head", "feet so off"). Look at
silhouettes and proportions between sections, not only the measured points. Show the user the side-by-side
pictures, change one thing per round, and keep the frames and crops that settled each decision. They are the
argument for the shape.
