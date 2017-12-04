# EnergyMinimizingBezierDemo
Interactive demonstration of energy-minimizing Bézier curves


`demo.py` is a a `matplotlib`-powered demonstration which a user can construct a Bézier 
curve by clicking repeatedly on the plot. The energy of the curve is shown on a seperate 
plot once three points have been selected. You can press `e` with your mouse over a point
on the control polygon to move it to a position which minimizes the energy. 

Commands:
 * `click` in the plot: Add a point to the curve's control polygon
 * `click on a point and drag`: Move a point
 * `d` key while mouse is hovering over a point: Delete that point
 * `e` key while mouse is hovering over a point: Move that point to a position which minimizes the energy
 * `1`, `2`, and `3` keys: Change the degree of the energy calculated and minimized
   * `1`: Stretch energy (first degree)
   * `2`: Strain energy (second degree)
   * `3`: Jerk energy (third degree)

The included Jupyter notebook investigates the matrices which generate points on a Bézier curve's 
control polygon which minimize the energy of the curve. 

