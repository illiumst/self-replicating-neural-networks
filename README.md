# self-rep NN paper - ALIFE journal edition

- [x] Plateau / Pillar sizeWhat does happen to the fixpoints after noise introduction and retraining?Options beeing: Same Fixpoint, Similar Fixpoint (Basin), 
    - Different Fixpoint?
      Yes, we did not found same (10-5)
    - Do they do the clustering thingy?
      Kind of: Small movement towards (MIM-Distance getting smaller) parent fixpoint.
      Small movement for everyone? -> Distribution

    - see `journal_basins.py` for the "train -> spawn with noise -> train again and see where they end up" functionality. Apply noise follows the `vary` function that was used in the paper robustness test with `+- prng() * eps`. Change if desired.

    - there is also a distance matrix for all-to-all particle comparisons (with distance parameter one of: `MSE`, `MAE` (mean absolute error = mean manhattan) and `MIM` (mean position invariant manhattan))


- [ ] Same Thing with Soup interactionWe would expect the same behaviour...Influence of interaction with near and far away particles.

- [ ] How are basins / "attractor areas" shaped?
      - Weired.... tbc...

- [x] Robustness test with a trained NetworkTraining for high quality fixpoints, compare with the "perfect" fixpoint. Average Loss per application step
    
    - see `journal_robustness.py` for robustness test modeled after cristians robustness-exp (with the exeption that we put noise on the weights). Has `synthetic` bool to switch to hand-modeled perfect fixpoint instead of naturally trained ones. 

    - Also added two difference between the "time-as-fixpoint" and "time-to-verge" (i.e. to divergence / zero).

    - We might need to consult about the "average loss per application step", as I think application loss get gradually higher the worse the weights get. So the average might not tell us much here.

- [ ] Adjust Self Training so that it favors second order fixpoints-> Second order test implementation (?)

- [x] Barplot over clones -> how many become a fixpoint cs how many diverge per noise level

- [ ] Box-Plot of Avg. Distance of clones from parent

# Future Todos:

- [ ] Find a statistik over weight space that provides a better init function
- [ ] Test this init function on a mnist classifier - just for the lolz
- [ ] 
 
---
## Notes: 

- In the spawn-experiment we now fit and transform the PCA over *ALL* trajectories, instead of each net-history by its own. This can be toggled by the `plot_pca_together` parameter in `visualisation.py/plot_3d_self_train() & plot_3d()` (default: `False` but set `True` in the spawn-experiment class).

- I have also added a `start_time` property for the nets (default: `1`). This is intended to be set flexibly for e.g., clones (when they are spawned midway through the experiment), such that the PCA can start the plotting trace from this timestep. When we spawn clones we deepcopy their parent's saved weight_history too, so that the PCA transforms same lenght trajectories. With `plot_pca_together` that means that clones and their parents will literally be plotted perfectly overlayed on top, up until the spawn-time, where you can see the offset / noise we apply. By setting the start_time, you can avoid this overlap and avoid hiding the parent's trace color which gets plotted first (because the parent is always added to self.nets first). **But more importantly, you can effectively zoom into the plot, by setting the parents start-time to just shy of the end of first epoch (where they get checked on fixpoint-property and spawn clones) and the start-times of clones to the second epoch. This will make the plot begin at spawn time, cutting off the parents initial trajectory and zoom-in to the action (see. `journal_basins.py/spawn_and_continue()`).**

- Now saving the whole experiment class as pickle dump (`experiment_pickle.p`, just like cristian), hope thats fine.

- Added a `requirement.txt` for quick venv / pip -r installs. Append as necessary.  