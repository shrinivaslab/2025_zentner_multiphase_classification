from fh_utils import model_A_dynamics, explicit_solvent_dynamics
from classifier_utils import classifier_setup
from analysis import cluster_utils
from gen_utils import plot_gen

import os

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import equinox

# Function to run classifier on test data

def get_classes(phi_out, threshold, lower_thresh, n_components):

    len_phi = len(phi_out[0])

    classes = len_phi*jnp.ones(len(phi_out), dtype=jnp.int32)

    for i in range(len(phi_out)):
        for j in range(len_phi):
            if phi_out[i,j]>(1+threshold)/n_components and jnp.all(phi_out[i,jnp.delete(jnp.arange(len_phi),j)]<lower_thresh/n_components):
                classes=classes.at[i].set(j)
    return classes

def run(chimat_opt, mu_res_opt, test_data, test_indices, sizes, threshold=0.1, lower_thresh=0.25, fast_solvent=True):
    '''
    Description: function for testing the classifier using optimized parameters

    Parameters:
        chimat_opt (array): the optimized interaction matrix
        mu_res_opt (array): the optimized chemical potential of the reservoir
        test_data (array): input concentration vectors used for testing
        test_indices (array): target indices (corresponding to the output species that should be enriched) of the test data.

    Returns: a length-4 vector that includes
                - phi_test_initial (array): the initial full concentration vector (including randomly-generated output and hidden concentrations)
                - phi_test_final (array): the finall full concentration vectors
                - cluster_mean_densities (array): the dominant phases in the test data
                - cluster_indices (array): the index of dominant phase to which each test final concentration vector corresponds
    '''
    if fast_solvent == True:
        diffrax_integrate = model_A_dynamics.get_integrator()
    else:
        diffrax_integrate = explicit_solvent_dynamics.get_integrator()

    n_inputs, n_outputs, n_hidden = sizes

    #----------Run Test Data----------#

    test_setup = classifier_setup.DataSetup_Custom(test_data, test_indices, n_outputs=n_outputs, n_hidden=n_hidden, seed=101)
    phi_test_initial = jnp.concatenate([test_setup.data_in.T, test_setup.data_out_and_hidden.T]).T
    phi_final, _ = equinox.filter_vmap(diffrax_integrate, in_axes=(0,0,None,None))(test_setup.data_out_and_hidden, test_setup.data_in, chimat_opt[n_inputs:], mu_res_opt)
    phi_test_final = jnp.concatenate([test_setup.data_in.T, phi_final.T]).T

    indices_pred = get_classes(phi_final[:,:n_outputs], threshold, lower_thresh, sum(sizes))

    #----------Get Relevant Eigenvalues----------#

    cluster_mean_densities, cluster_indices = cluster_utils.get_cluster_mean_densities(phi_final)

    #----------Return arrays----------#

    return phi_test_initial, phi_test_final, test_indices, indices_pred, cluster_mean_densities, cluster_indices

# Function to export data files for test results

def export(test_results, boundary_type, sizes, max_input, threshold, path):

    n_inputs, n_outputs, n_hidden = sizes

    path_test = path + '/data_files/test_data'
    path_plot = path + '/output_plots'
    for p in [path_test, path_plot]:
        if not os.path.exists(p):
            os.makedirs(p)

    #-----------------------------------------------------Data Files-----------------------------------------------------#

    phi_test_initial, phi_test_final, test_indices, indices_pred, cluster_mean_densities, cluster_indices = test_results
    
    jnp.save(path_test + f'/test_phis_initial_{boundary_type}_{n_hidden}hidden', phi_test_initial) # Initial Test Data
    jnp.save(path_test + f'/test_indices_{boundary_type}_{n_hidden}hidden', test_indices)
    jnp.save(path_test + f'/test_phis_final_{boundary_type}_{n_hidden}hidden', phi_test_final) # Final Test Data
    jnp.save(path_test + f'/cluster_densities_{boundary_type}_{n_hidden}hidden', cluster_mean_densities) # Mean Cluster Densities
    jnp.save(path_test + f'/cluster_indices_{boundary_type}_{n_hidden}hidden', cluster_indices) # Mean Cluster Densities

    #---------------------------------------------------------Output Plots---------------------------------------------------------#

    # Final concentration vectors for a few points
    fig = plot_gen.plot_phi_test(phi_test_final, test_indices, sizes)
    plot_gen.fig_export(fig, path_plot + f'/phis_{boundary_type}_{n_hidden}hidden')

    # Principle Vectors
    fig = plot_gen.plot_cluster_vecs(cluster_mean_densities, sizes)
    plot_gen.fig_export(fig, path_plot + f'/cluster_vecs_{boundary_type}_{n_hidden}hidden')

    # Confusion Matrix
    fig = plot_gen.plot_confusion_matrix(test_indices, indices_pred)
    plot_gen.fig_export(fig, path_plot + f'/confusionMatrix_{boundary_type}_{n_hidden}hidden')
    
    #----------Extra Plots for 2D Cases----------#

    # Classifier Guesses
    if n_inputs==2:
        fig = plot_gen.plot_classifier_guesses(phi_test_final[:,:n_inputs], indices_pred, boundary_type, max_input)
        plot_gen.fig_export(fig, path_plot + f'/class_{boundary_type}_{n_hidden}hidden')

        fig = plot_gen.plot_classifier_guesses(phi_test_final[:,:n_inputs], cluster_indices, boundary_type, max_input, phase_plot=True)
        plot_gen.fig_export(fig, path_plot + f'/clustering_{boundary_type}_{n_hidden}hidden')