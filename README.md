# Gradient Boosting – Minimal Vanilla JS Demo

This is a tiny, framework-free web app that visualizes gradient boosting applied to 1D regression, now with a two-panel view showing residuals.

What it does:

- Generates a green parabola y = 1 + x^2 and samples noisy points from it
- Briefly displays the parabola (green), then hides it
- Left panel: fits a single shallow regression tree (depth 2) to the noisy data and plots its prediction in purple; the green parabola appears briefly.
- Right panel: plots residuals (red points) defined as y - y_hat_left; then fits another depth-2 tree to residuals and plots the residual prediction in purple.
- Hover over each canvas to probe the corresponding model at x. Click "Regenerate data" to sample a new dataset and retrain both trees.

No build steps, no frameworks—just a single HTML page with CSS and JS.

## Run locally

Open `index.html` in a browser, or serve the folder with any static file server.

Optional quick server (Python 3):

```
python3 -m http.server 8000
```

Then visit http://localhost:8000.

## Files

- `index.html` – app shell
- `style.css` – minimal styling
- `main.js` – data generation, tiny CART trees, residual computation, and canvas rendering

## Notes

- Trees split only on x; this is intentionally simple for clarity.
- Hyperparameters (depth is fixed at 2 for this demo, noise, etc.) are set near the top of `main.js` and can be tweaked easily.
