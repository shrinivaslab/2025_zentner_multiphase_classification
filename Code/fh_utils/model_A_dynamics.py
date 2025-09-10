from fh_utils import fh_quantities

import jax
import jax.numpy as jnp
import equinox
import diffrax
jax.config.update("jax_enable_x64", True)

# Solver Parameters
mobility=True

#--------------------x reparametrization functions--------------------#

def get_x_from_phi(phi, phi_max):
    '''
    Description: get the x parametrization of a concentration vector

    Parameters:
        phi (array): concentration vector
        phi_max (float): the maximum total volume fraction allowed for the concentration vector

    Returns: parametrized concentration vector
    '''
    return jnp.log(phi/(phi_max-jnp.sum(phi)))

def get_phi_from_x(x, phi_max):
    '''
    Description: the inverse transformation of `get_x_from_phi`; get the phi parametrization of a x-parametrized concentration vector

    Parameters:
        params (array): parametrized concentration vector
        phi_max (float): the maximum total volume fraction allowed for the concentration vector

    Returns: concentration vector
    '''
    return phi_max/(1+jnp.sum(jnp.exp(x))) * jnp.exp(x)

#--------------------Model A ODE RHS--------------------#

def phi_ODE(t, phi_p, args):
    '''
    Description: obtain the RHS of unparametrized Model-A dynamics (specifically formatted for Diffrax)

    Parameters:
        phi_p (array): vector of output and hidden concentrations
        phi_in (array): vector of input concentrations
        chi (array): interaction matrix
        mu_res (array): chemical potential of reservoir
    
    Returns: RHS of unparametrized Model-A dynamics
    '''
    phi_in, chi, mu_res = args

    if mobility==True:
        return  - phi_p * (fh_quantities.mu_oh(phi_p, phi_in, chi) - mu_res)
    else:
        return - (fh_quantities.mu_oh(phi_p, phi_in, chi) - mu_res)

def x_ODE(t, xp, args):
    '''
    Description: obtain the RHS of x-parametrized Model-A dynamics (specifically formatted for Diffrax)

    Parameters:
        xp (array): x-parametrized vector of output and hidden concentrations
        phi_in (array): vector of input concentrations (unparametrized)
        chi (array): interaction matrix
        mu_res (array): chemical potential of reservoir
    
    Returns: RHS of x-parametrized Model-A dynamics
    '''
    phi_in, chi, mu_res = args

    exp_xp = jnp.exp(xp)
    terms = (1-jnp.sum(phi_in))/(1+jnp.sum(exp_xp))*exp_xp

    mu_x = xp + jnp.matmul(chi, jnp.concatenate([phi_in, terms]))
    mu_diff = mu_x - mu_res
    if mobility==True:
        return -(jnp.dot(exp_xp, mu_diff) + mu_diff)
    else:
        prefactor = (1+jnp.sum(exp_xp))/(1-jnp.sum(phi_in))
        return -prefactor * (jnp.sum(mu_diff) + jnp.exp(-xp) * mu_diff)

#--------------------DIFFRAX APPROACH--------------------#

def get_integrator(int_param='x', illustrate=False):

    #---------- Set parameters ----------#

    if int_param=='x':
        term = diffrax.ODETerm(x_ODE)
    elif int_param=='phi':
        term = diffrax.ODETerm(phi_ODE)
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-7, atol=1e-9)
    max_steps = None
    t0 = 0

    if illustrate == False:
        cond_fn = diffrax.steady_state_event(rtol=1e-3, atol=1e-3)
        event = diffrax.Event(cond_fn)
        t1 = jnp.inf
        dt0 = None
        saveat = diffrax.SaveAt(t1=True)
        adjoint = diffrax.ImplicitAdjoint()

    else:
        event = None
        t1 = 50
        dt0 = 0.01
        adjoint = diffrax.RecursiveCheckpointAdjoint()
        saveat = diffrax.SaveAt(ts=jnp.arange(t0, t1, dt0))

    #---------- Define integrator ----------#

    @equinox.filter_jit
    def diffrax_integrate(phi_p, phi_in, chi_p, mu_res_p):
        '''
        Description: perform x-parametrized Model-A dynamics using Diffrax library

        Parameters:
            phi_p (array): vector of output and hidden concentrations
            phi_in (array): vector of input concentrations
            chi_p (array): interaction matrix (excluding input rows)
            mu_res_p (array): chemical potential vector of reservoir (excluding input components)

        Returns: final vector of output and hidden concentrations, and auxiliary concentration vector data
        '''
        # Parametrization
        phi_in_tot = jnp.sum(phi_in)
        if int_param=='x':
            x0 = get_x_from_phi(phi_p, 1-phi_in_tot)
        else:
            x0 = phi_p
        
        solution = diffrax.diffeqsolve(term, 
                            solver,
                            t0=t0,
                            t1=t1,
                            dt0=dt0,
                            y0=x0,
                            args=(phi_in, chi_p, mu_res_p),
                            stepsize_controller=stepsize_controller,
                            max_steps=max_steps,
                            event=event,
                            saveat=saveat,
                            adjoint=adjoint
                            )

        # Invert Parametrization
        if int_param == 'x':
            phi_final = get_phi_from_x(solution.ys[-1], 1-phi_in_tot)
            phi_aux = equinox.filter_vmap(get_phi_from_x, in_axes=(0,None))(solution.ys, 1-phi_in_tot)
        else:
            phi_final = solution.ys[-1]
            phi_aux = solution.ys
        t_vals = solution.ts

        return phi_final, (solution.stats, t_vals, solution.ys, phi_aux)

    return diffrax_integrate

if __name__ == "__main__":

    import os
    import time

    #----------Set problem type----------#

    boundary_type = 'and'
    n_inputs, n_outputs, n_hidden = [2, 2, 4]
    batch_length=128

    phi_in = (0.5/n_inputs)*jnp.ones(n_inputs)
    phi_out_and_hidden = 0.001*jnp.ones(n_outputs+n_hidden)

    #----------Set parameters compatible with data logging----------#

    diffrax_integrate = get_integrator(illustrate=True)

    #----------Import Parameters and Run Simulation----------#
    
    path = os.getcwd() + '/results/classifier_optimization_results/'+boundary_type+'/'+str(n_hidden)+'_hidden'+'/batch_length_'+str(batch_length)+'/data_files'
    # Interaction Matrix (must be of shape `n_components x n_components`)
    chi0 = jnp.load(path+'/chi_matrix_'+boundary_type+'_'+str(n_hidden)+'hidden.npy')
    # Reservoir Chemical Potential (excludes input species, which are assumed to be absent from the reservoir)
    mu_res_0 = jnp.load(path+'/mu_reservoir_'+boundary_type+'_'+str(n_hidden)+'hidden.npy')

    start=time.time()
    _, (stats, t_vals, x_aux, phi_aux_from_x_diffrax) = diffrax_integrate(phi_out_and_hidden, phi_in, chi0[n_inputs:], mu_res_0)
    end=time.time()
    print('time elapsed: '+str(end-start))

    get_mu_diffs = lambda phi, phi_in, chi: fh_quantities.mu_analytic(jnp.concatenate([phi_in, phi]), chi)[n_inputs:] - mu_res_0
    mu_diffs = jax.vmap(get_mu_diffs, in_axes=(0,None,None))(phi_aux_from_x_diffrax, phi_in, chi0)
    rhs_vals = jax.vmap(x_ODE, in_axes=(0,0,None))(t_vals, x_aux, (phi_in, chi0[n_inputs:], mu_res_0))
    
    #----------Plotting Run Data----------#

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,4,figsize=(15,5))

    ax[0].plot(t_vals, x_aux)
    ax[0].set_xlabel(r'$t$', fontsize=14)
    ax[0].set_ylabel(r'$x_i(t)$', fontsize=14)
    ax[0].set_title(r'$x$ parametrization')

    ax[1].plot(t_vals, phi_aux_from_x_diffrax)
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$t$', fontsize=14)
    ax[1].set_ylabel(r'$\phi_i(t)$', fontsize=14)
    ax[1].set_title(r'$\phi$ parametrization')

    ax[2].plot(t_vals, mu_diffs)
    ax[2].set_xlabel(r'$t$', fontsize=14)
    ax[2].set_ylabel(r'$\mu_i-\mu_\text{ref}^{(i)}$', fontsize=14)
    ax[2].set_title('Chem. Potential Differences')

    ax[3].plot(t_vals, jnp.abs(rhs_vals))
    ax[3].set_yscale('log')
    ax[3].set_xlabel(r'$t$', fontsize=14)
    ax[3].set_ylabel('RHS', fontsize=14)
    ax[3].set_title('ODE RHS Values', fontsize=14)

    plt.tight_layout()
    plt.show()

    #----------Export Run Data----------#

    phi_string = ("-".join([str(i) for i in phi_in])).replace(".", "p")
    path_save = path+'/point_runs/'+phi_string+'/'

    if not os.path.exists(path_save):
        os.makedirs(path_save)

    jnp.save(path_save+boundary_type+'_tvals_'+phi_string+'.npy', t_vals)
    jnp.save(path_save+boundary_type+'_xvals_'+phi_string+'.npy', x_aux)
    jnp.save(path_save+boundary_type+'_phiVals_'+phi_string+'.npy', phi_aux_from_x_diffrax)
    jnp.save(path_save+boundary_type+'_muDiffs_'+phi_string+'.npy', mu_diffs)
    jnp.save(path_save+boundary_type+'_rhs_'+phi_string+'.npy', rhs_vals)
    fig.savefig(path_save+boundary_type+'_plot_'+phi_string)