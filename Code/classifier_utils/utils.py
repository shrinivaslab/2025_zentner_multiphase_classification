import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

#----------------------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------Argument Passing Utilities----------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary_type", default="linear")
    parser.add_argument("--batch_length", type=int, default=64)
    parser.add_argument("--n_hidden", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--paramsType", default="chi_and_mu")
    parser.add_argument("--max_chi", type=float, default=15.0)
    parser.add_argument("--min_out_chi", type=float, default=10.0)

    # Extract dictionary of arguments
    args = parser.parse_args()
    # Extract parsed parameter
    return (args.boundary_type, args.batch_length, args.n_hidden, args.threshold,
            args.paramsType, args.max_chi, args.min_out_chi)

#----------------------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------------Parameter Formatting-------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------------------#

#--------------------chi matrix construction--------------------#

def chivec_to_chimat(values, min_out_chi, max_chi, n_inputs, n_outputs, n_hidden):
    '''
    Description: Convert chi from a vector into a symmetric matrix with 0 diagonal entries. Input interactions are set to 0.

    Parameters:
        values (array): the chi vector to be converted to a symmetric matrix
        min_out_chi (float): the minimum allowed output-output interaction
        max_chi (float): the maximum interaction allowed strength
        n_inputs (int): the number of input species
        n_outputs (int): the number of output species
        n_hidden (int): the number of hidden species

    Returns: the symmetric chi matrix, with 0 diagonals and input interactions
    '''
    # Scale interactions
    values = max_chi*jnp.tanh(values)

    # Make lower-triangular matrix
    size = n_inputs + n_outputs + n_hidden
    mat = jnp.zeros((size, size))
    indices = tuple(jnp.array(jnp.tril_indices(size, -1))[:,int(n_inputs*(n_inputs-1)/2):])
    mat = mat.at[indices].set(values)

    # Impose repulsive constraint on output interactions
    out_out_indices = tuple(n_inputs + jnp.array(jnp.tril_indices(n_outputs, -1)))
    mat = mat.at[out_out_indices].set(min_out_chi)# + relu(mat[out_out_indices]-min_out_chi))

    # Make symmetric matrix
    return mat + mat.T

def chimat_to_chivec(mat, max_chi, n_inputs, n_outputs, n_hidden):
    '''
    Description: Convert chi from a symmetric matrix with 0 diagonal entries into a vector. Input interactions are set to 0.

    Parameters:
        mat (array): the chi matrix to be converted to a vector
        n_inputs (int): the number of input species
        n_outputs (int): the number of output species
        n_hidden (int): the number of hidden species

    Returns: the symmetric chi matrix, with 0 diagonals and input interactions
    '''

    # Make lower-triangular matrix
    size = n_inputs + n_outputs + n_hidden
    indices = tuple(jnp.array(jnp.tril_indices(size, -1))[:,int(n_inputs*(n_inputs-1)/2):])
    
    # Scale interactions
    return jnp.atanh(mat[indices]/max_chi)