import jax.numpy as jnp
from scipy import cluster

def get_num_phases(phi_vecs, eig_thresh = None):
    '''
    Description: determine the number of significant phases in a collection of concentration vectors based on the Marchenko-Pastur distribution

    Parameters:
        phi_vecs (array): collection of concentration vectors
        eig_thresh (float): the threshold above which an eigenvalue must be for the corresponding eigenvector to be significant

    Returns: number of phases, eigenvalues, eigenvectors
    '''
    # Rescale vectors and construct covariance matrix
    phi_mat = ((phi_vecs - jnp.mean(phi_vecs, 0)) / jnp.std(phi_vecs, 0))
    m, n = phi_mat.shape
    if eig_thresh == None:
        eig_thresh = (1+jnp.sqrt(n/m))**2
        
    cov_mat = (phi_mat.T).dot(phi_mat) / (phi_mat.shape[0] - 1)

    # Eigenvalues and eigenvectors
    eig_vals, eig_vecs = jnp.linalg.eigh(cov_mat)

    # Number of phases
    n_phases = len(eig_vals[eig_vals>=eig_thresh]) + 1

    return n_phases, eig_vals, eig_vecs

def get_cluster_mean_densities(phi_vecs):
    '''
    Description: determine the dominant phases in a collection of concentration vectors

    Parameters:
        phi_vecs (array): collection of concentration vectors
        eig_thresh (float): the threshold above which an eigenvalue must be for the corresponding eigenvector to be significant

    Returns: the concentration vectors of the dominant phases and the index of the phase to which each `phi_vec` entry belongs
    '''
    # Get number of phases
    n_phases = get_num_phases(phi_vecs)[0]
    # Perform clustering on centered and reduced data
    cluster_data = cluster.hierarchy.fclusterdata(phi_vecs, n_phases, criterion='maxclust', method='ward')
    # Get average compositions
    compositions_new = jnp.array([jnp.mean(phi_vecs[cluster_data == i+1, :], axis=0) for i in range(n_phases)])

    return compositions_new, cluster_data