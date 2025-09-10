from classifier_utils import train, test

import jax
import jax.numpy as jnp
import os

def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)

def tree_unstack(tree):
    leaves, treedef = jax.tree.flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

def run(seeds, training_data, training_indices, test_data, test_indices, max_input, extra_args,
        validation_data=None, validation_indices=None, chi0=None, mu_res_0=None, c_l1=0., path=None):
    
    if type(seeds)!=int and (validation_data==None or validation_indices==None):
        raise TypeError("When training across many random seeds, validation data and indices must be supplied.")
    
    # Define constants
    boundary_type, batch_length, n_hidden, threshold, paramsType, max_chi, min_out_chi = extra_args

    print(boundary_type)

    n_inputs = (training_data.shape)[-1]
    n_outputs = len(jnp.unique(training_indices))
    sizes = [n_inputs, n_outputs, n_hidden]

    # Define export path
    if path==None:
        path = os.getcwd() + f'/results/optimization_results/{boundary_type}/{n_hidden}_hidden/batch_length_{batch_length}'

    def training_fn(s, training_data, training_indices):
        sol_opt, aux_opt = train.run(seed = s,
                                     training_data = training_data,
                                     training_indices = training_indices,
                                     n_hidden = n_hidden,
                                     batch_length = batch_length,
                                     threshold = threshold,
                                     paramsType = paramsType,
                                     max_chi = max_chi,
                                     min_out_chi = min_out_chi,
                                     chi0 = chi0,
                                     mu_res_0 = mu_res_0,
                                     c_l1 = c_l1,
                                     save_intermediate = save_intermediate,
                                     boundary_type = boundary_type,
                                     path = path
                                     )
        return sol_opt, aux_opt
    batch_training_fn = jax.vmap(training_fn, in_axes=(0,None,None))

    if type(seeds)==int:

        save_intermediate = True

        # Training
        sol_opt, aux_opt = training_fn(seeds, training_data, training_indices) # run optimization
        # Testing
        test_results = test.run(*sol_opt, test_data, test_indices, sizes)
        test.export(test_results, boundary_type, sizes, max_input, threshold, path)

    else:

        # Train

        save_intermediate = False

        sols, auxs = batch_training_fn(seeds, training_data, training_indices)
        sols = tree_unstack(sols)
        auxs = tree_unstack(auxs)

        # Validate and Export

        validation_success = jnp.zeros(len(seeds))

        for i in range(len(seeds)):
            path_seeds = path + f'/seeds/{i}'
            train.export(0, 0, sols[i], auxs[i], boundary_type, sizes, path_seeds)

            validation_results = test.run(*sols[i], validation_data, validation_indices, sizes)
            _, _, indices_true, indices_pred, _, _, _ = validation_results

            validation_success.at[i].set(jnp.sum(indices_pred==indices_true)/len(indices_pred))

            test.export(validation_results, boundary_type, sizes, max_input, threshold, path_seeds)
         
        max_arg = jnp.argmax(validation_success)
        sol_opt = sols[max_arg]
        aux_opt = auxs[max_arg]

        # Test and Export

        test_results = test.run(*sol_opt, test_data, test_indices, sizes)

        path_opt = path + f'/opt'
        train.export(0, 0, sol_opt, aux_opt, boundary_type, sizes, path_opt)
        test.export(test_results, boundary_type, sizes, max_input, threshold, path_opt)

    # return optimization parameters

    return sol_opt, aux_opt