# code ALIFE paper journal edition

- see journal_basins.py for the "train -> spawn with noise -> train again and see where they end up" first draft. Apply noise follows the `vary` function that was used in the paper robustness test with `+- prng() * eps`. Change if desired.
- has some interesting results, but maybe due to PCA the newly spawned weights + noise get plotted quite a bit aways from the parent particle, even though the weights are similar to 10e-8?
- see journal_basins.py for an attempt at a distance matrix between the nets/end_weight_states of an experiment. Also has the position-invariant-manhattan-distance as option. (checks n^2, neither fast nor elegant ;D )
- i forgot what "sanity check for the beacon" meant but we leave that out anyway, right?


## Some changes from cristian's code (or suggestions, rather)

This is just my understanding, I might be wrong here. Just a short writeup of what I noticed from trying to implement the new experiments. 
EDIT: I also saw that you updated your branch, so some of these things might have already been adressed.  

- I think, id_function is only training to reproduce the *very first weight* configuration, right? Now I see where the confusion is. But according to my understanding the selfrep networks gets trained with the task to ouput the *current weights at each training timestep*, so dynamic targets as the weight learns until it stabilizes/converges. I have changed that accordingly in the experiments to produce one input/target **per step** and train on that once (batch_size 1) for e.g. ST_step many times (not ST_many times on the inital one input/target).

- Not sure about this one but: Train only seems to save the *output* (i.e, the prediction, not the net weight states)? Semantically this changes the 3d trajectories from the papers:
    - "the trajectory dont change anymore because the *weights* are always the same" , ie. the backprop gradient doesnt change anything because the loss of the prediction is basically nonexistant,
    - to "the net has learned to return the input vector 1:1 (id_function, yes) and the *output* prediction is the *same* everytime". Eventually weights == output == target_data, but we are interested in the weight states trajectory during learning and not really the output, i guess (because we know that the output will eventually converge to the right prediction of the weighs, but not how the weights develop during training to accomodate this ability). Logging target_data would be better because that is basically the weights at each step we are aiming for. Thats what i am using now at least.

- robustness test doesnt seem to self apply the prediction currently, it only changes the weights (apply weights ≠ self.apply), right? Thats why the Readme has the notice "never fails for the smaller values", because it only adds epsilon small enough to not destroy the fixpoint property (10e-6 onwards) and not actually tries to self-apply. If the changed weights + noise small enough = fixpoint, then it will always hold without change (i.e., without the actual self application). Also the noise is *on the input*, which is a robustness-test for id_function, yes, while the paper experiment has the noise *on the weights*. Semantically, noise on the input asks "can the same net/weights produce the same input/output even when we change the output", which of course not. But the output may be changed (small enough) that its within epsilon-degree of change and therefore not looses the fixpoint property. 
The robustness exp in the paper tests self-application resistance, which means how much faster do the nets loose prediction-accuracy on self-application when weights get x-amount of noise. They all loose precision even without noise (see the paper, self-application is by nature a value degrading operation/predicion closer to 0-values, "easier to predict"), its "just" the visualisation of how much faster it collapses to the 0-fixpoint with different amounts of noise on weight (and since the nets sample from within their own weights, on the input as well; weights => input).

- getting randdigit for the path destroys save-order, no? Makes finding stuff tricky. IRC thats why steffen used timestamps, they are ordered ascendingly?

- the normalize() is different from the paper, right? It gets normalized over len(state_dict) = 14, not over the positional encoding of each layer / cell / weight_value?

- test_for_fixpoint doesnt return/or set the id_functions array? How does that work? Do you then just filter all nets with the fitting "string" property somewhere?
