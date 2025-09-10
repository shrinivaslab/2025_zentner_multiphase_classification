from fh_utils import model_A_dynamics, fh_quantities
from classifier_utils import classifier_setup, utils
from gen_utils import plot_gen

import jax
import jax.numpy as jnp
from jax.nn import relu
import optax

from jax_tqdm import scan_tqdm
jax.config.update("jax_enable_x64", True)

import equinox

import os

def min_custom(x):
    return 1-relu(1-x)

def tree_stack(running_aux, aux):
    return jax.tree_util.tree_map(lambda x, y: jnp.concatenate([x, y]), running_aux, aux)

#----------Function to Train Classifier----------#

def run(seed, training_data, training_indices, n_hidden,
        batch_length=64, opt_steps = 1000, learning_rate = 0.01, c_l1=0.,
        paramsType = 'chi_and_mu', chi0 = None, mu_res_0 = None,
        threshold = 0.1, max_chi=15, min_out_chi=10,
        save_intermediate=False, boundary_type=None, path=None):
    '''
    Description: function for training the classifier on input data.

    Parameters:
        seed (int): the random seed to use in generating the initial parameters
        training_data (array): input concentration vectors used for training
        training_indices (array): target indices (corresponding to the output species that should be enriched) of the training data.
        n_hidden (int): the number of output species
        batch_length (int): number of training points to use in each optimization epoch
        opt_steps (int): number of optimization epochs
        learning_rate (float): the learning rate to use in the optimizer
        paramsType (string): the parameters over which to optimize. Options are 'chi_and_mu', 'chi', 'mu'.
        chi0 (array): fixed interaction matrix
        mu_res_0 (array): fixed chemical potential of the reservoir
        threshold (float): the factor by which the target output species should be greater than 1/N.
        min_out_chi (float): the minimum allowed value for output-output interactions
        max_chi (float): the maximum absolute value allowed for interactions.

    Return: the optimized interaction matrix and reservoir chemical potential, and auxiliary data
    '''

    if save_intermediate==True and (path==None or boundary_type==None):
        raise TypeError("If save_intermediate==True, the function must also be provided with an export path and a boundary_type")

    # Define system size
    n_inputs = (training_data.shape)[-1]
    n_outputs = len(jnp.unique(training_indices))
    n_components = n_inputs + n_outputs + n_hidden
    sizes=[n_inputs,n_outputs,n_hidden]

    setup = classifier_setup.DataSetup_Custom(training_data, training_indices, n_outputs=n_outputs, n_hidden=n_hidden, seed=100)

    # Define constants
    phi_max = (1+threshold)/n_components
    lower_thresh = 0.25
    phi_min = lower_thresh/n_components
    #c_l1 = 0.00005

    possible_targets = jnp.arange(n_outputs)
    off_targets = jnp.array([jnp.delete(possible_targets,idx) for idx in possible_targets])
    array_idx = jnp.arange(n_outputs*int(batch_length/n_outputs))

    # Define simplifying functions
    diffrax_integrate = model_A_dynamics.get_integrator()
    convert_to_mat = lambda chi: utils.chivec_to_chimat(chi, min_out_chi, max_chi, n_inputs, n_outputs, n_hidden)

    def initialize_params(seed, chi0, mu_res_0):
        '''
        Description: format the data depending on the parameters to be optimized. Options are to...
                        1) optimize over interaction matrix and fix chemical potential of the reservoir
                        2) fix interaction matrix and optimize over chemical potential of the reservoir
        
        Parameters:
            seed (int): the random seed to use in generating the initial parameters
            chi0 (array): initial interaction matrix
            mu_res_0 (array): initial chemical potential of the reservoir
            paramsType (string): the parameters over which to optimize. Options are 'chi_and_mu', 'chi', 'mu'.
            n_inputs (int): the number of input species
            n_components (int): the total number of species

        Returns: length-2 array, where the first entry is the initial vector of parameters to be optimized, and the second entry is the vector of parameters to be fixed.
        '''
        key = jax.random.PRNGKey(seed)
        key, split = jax.random.split(key)

        # Initialize a random interacton matrix:
        if chi0 == None:
            n_exclude = n_inputs
            n_include = n_components - n_exclude
            vec_len = int(n_include*n_exclude + n_include*(n_include - 1)/2)
            chi0 = jax.random.uniform(split, (1,vec_len), minval = -1., maxval = 1.)[0]
            if paramsType == 'mu':
                chi0 = convert_to_mat(chi0)

        # Initialize a reference chemical potential vector for the reservoir:
        if mu_res_0 == None:
            # Input components are taken to be 0
            mu_res_0 = -5.0*jnp.ones(n_components-n_inputs)

        # Organize into params to be altered vs fixed:
        if paramsType == 'chi_and_mu':
            return jnp.concatenate([chi0, mu_res_0]), None
        elif paramsType == 'chi':
            return chi0, mu_res_0
        elif paramsType == 'mu':
            return mu_res_0, chi0

    def format_params(params, fixed_quantities):
        '''
        Description: function for formatting the optimization parameters

        Parameters:
            params (array): vector of optimization parameters
            fixed_quantities (array): vector of fixed parameters

        Returns: formatted interaction matrix and chemical potential of the reservoir
        '''
        if paramsType == 'chi_and_mu':
            len_mu = n_components-n_inputs
            chi_mat = convert_to_mat(params[:-len_mu])
            mu = params[-len_mu:]
            if n_hidden>0:
                mu = mu.at[:n_outputs].set(mu[0]*jnp.ones(n_outputs)) #equal output mu's
            return chi_mat, mu
        elif paramsType == 'chi':
            chi_mat = convert_to_mat(params)
            mu = fixed_quantities
            return chi_mat, mu
        elif paramsType == 'mu':
            chi_mat = fixed_quantities
            mu = params
            if n_hidden>0:
                mu = mu.at[:n_outputs].set(mu[0]*jnp.ones(n_outputs)) #equal output mu's
                if mu_res_0!=None:
                    mu = mu.at[:n_outputs].set(mu_res_0[0]*jnp.ones(n_outputs)) #fix output mu's to initial vector
            return chi_mat, mu

    def predict(params, phi_in, phi_oh):
        '''
        Description: perform Model-A Dynamics with fixed input concentrations to get final output and hidden concentrations
        
        Parameters:
            params (array): vector of optimization parameters
            phi_in (array): initial vector of input concentrations
            phi_out_and_hidden (array): initial vector of output and hidden concentrations

        Returns: final output concentration vector
        '''
        chi_mat, mu_res = params
        phi_oh, (stats, _, _, _) = diffrax_integrate(phi_oh, phi_in, chi_mat[n_inputs:], mu_res)
        phi_full = jnp.concatenate([phi_in, phi_oh])
        return phi_oh[:n_outputs], (jnp.array(list(stats.values())[1:3]), jnp.sum(phi_full), jnp.max(jnp.abs(fh_quantities.mu_oh(phi_oh, phi_in, chi_mat[n_inputs:])-mu_res)))
    batch_predict = equinox.filter_vmap(predict, in_axes=(None, 0, 0))

    @equinox.filter_jit
    def loss(params, seed):
        '''
        Description: compute loss based on output concentration vectors after Model-A dynamics
        
        Parameters:
            params (array): vector of optimization parameters
            seed (int): random seed for selecting a different batch of training data for each optimization epoch

        Returns: the loss associated with the given batch
        '''
        formatted_params = format_params(params, fixed_quantities)
        chi_mat = formatted_params[0]
        phi_inputs, phi_out_and_hidden, target = setup.get_batch(seed, batch_length)
        # Run forward dynamics to get outputs
        phi_out_opt, predict_aux = batch_predict(formatted_params, phi_inputs, phi_out_and_hidden)
        target_elem_loss = 1. + (1+threshold)*relu(1 - phi_out_opt[array_idx,target]/phi_max)
        off_target_elem_loss = 1. + lower_thresh*relu(phi_out_opt[array_idx, off_targets[target].T].T/phi_min-1)
        return (jnp.sum(jnp.log(target_elem_loss)) + jnp.sum(jnp.log(off_target_elem_loss)))/len(target) + c_l1*jnp.sum(jnp.abs(chi_mat)), predict_aux

    def get_params_opt(params, opt_steps):
        '''
        Description: perform optimization

        Parameters:
            params (array): vector of optimization parameters

        Returns: the optimized parameters and auxiliary data
        '''
        loss_and_grad = equinox.filter_value_and_grad(loss, has_aux=True)

        sched = optax.piecewise_constant_schedule(init_value=learning_rate, boundaries_and_scales={int(2*opt_steps/3): 0.25, int(5*opt_steps/6): 0.1})

        solver = optax.rmsprop(learning_rate=sched, nesterov=True)
        opt_state = solver.init(params)

        @scan_tqdm(opt_steps, print_rate=50)
        def step_fn(state, t):
            opt_state, params = state
            (l, (stats_aux, phi_tot_aux, mu_diff_aux)), g = loss_and_grad(params, t)
            updates, opt_state = solver.update(g, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (opt_state, params), (stats_aux, phi_tot_aux, mu_diff_aux, l)
        
        save_every=50
        for i in range(int(opt_steps/save_every)):
            # Time each scan over 50 optimization epochs
            (opt_state, params), aux_i = jax.lax.scan(step_fn, (opt_state, params), jnp.arange(save_every*i,save_every*(i+1)))
            aux = tree_stack(aux, aux_i) if i>0 else aux_i
            if save_intermediate==True:
                export(save_every*(i+1), opt_steps, format_params(params, fixed_quantities), aux, boundary_type, sizes, path=path)

        return params, aux

    init_params, fixed_quantities = initialize_params(seed, chi0, mu_res_0)
    params_opt, aux = get_params_opt(init_params, opt_steps)
    sol_opt = format_params(params_opt, fixed_quantities)

    return sol_opt, aux

# Function to export data files for training results

def export(step, opt_steps, sol_opt, aux, boundary_type, sizes, path):

    chimat_opt, res_opt = sol_opt
    stats_aux, phi_tot_vals, mu_diff_vals, loss_vals = aux
    n_hidden = sizes[-1]

    if step<opt_steps:
        # Export data from intermediate optimization epochs
        path_intermediate = path + f'/data_files/intermediate_data/{step}/'
        path_logging = path + '/data_files/data_logging/'
        for p in [path_intermediate, path_logging]:
            if not os.path.exists(p):
                os.makedirs(p)

        jnp.save(path_intermediate + f'chi_matrix_{boundary_type}_{n_hidden}hidden', chimat_opt) # Interaction Matrix
        jnp.save(path_intermediate + f'mu_reservoir_{boundary_type}_{n_hidden}hidden', res_opt) # Reservoir Chemical Potential

        jnp.save(path + f'/data_files/loss_vals_{boundary_type}_{n_hidden}hidden', loss_vals)
        jnp.save(path_logging + f'num_steps_{boundary_type}_{n_hidden}hidden', stats_aux)

    elif step==opt_steps:
        # Export data for final optimization epoch
        path_data = path + '/data_files'
        path_plots = path + '/output_plots'
        for p in [path_data, path_plots]:
            if not os.path.exists(p):
                os.makedirs(p)

        jnp.save(path_data + f'/loss_vals_{boundary_type}_{n_hidden}hidden', loss_vals) # Loss Values
        jnp.save(path_data + f'/chi_matrix_{boundary_type}_{n_hidden}hidden', chimat_opt) # Interaction Matrix
        jnp.save(path_data + f'/mu_reservoir_{boundary_type}_{n_hidden}hidden', res_opt) # Reservoir Chemical Potential

        # Plot loss and outputs
        fig = plot_gen.plot_loss_and_outputs(loss_vals, (chimat_opt, res_opt), sizes)
        plot_gen.fig_export(fig, path_plots + f'/loss_{boundary_type}_{n_hidden}hidden')

        # Plot logging mu - mu_ref
        fig = plot_gen.plot_success_tracker(mu_diff_vals, phi_tot_vals)
        plot_gen.fig_export(fig, path_plots + f'/ModelA_SuccessTracker_{boundary_type}_{n_hidden}hidden')