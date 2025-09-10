import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

#----------Florry-Huggins Chemical Potential----------#

def mu_analytic(phi, chimat):
    '''
    Description: calculate the Flory-Huggins chemical potential of a phase vector given an interaction matrix.

    Parameters:
        phi (array): concentration vector
        chimat (array): interaction matrix

    Returns: chemical potential vector
    '''
    return jnp.log(phi) - jnp.log(1.-jnp.sum(phi)) + jnp.matmul(chimat,phi) # Assuming no solvent interaction

def mu_oh(phi_p, phi_in, chi):
    '''
    Description: calculate the Flory-Huggins chemical potential of output and hidden components only.

    Parameters:
        phi_p (array): vector of output and hidden concentrations
        phi_in (array): vector of input concentrations
        chi (array): interaction matrix

    Returns: chemical potential vector (excluding input components)
    '''
    phi = jnp.concatenate([phi_in, phi_p])
    return jnp.log(phi_p) - jnp.log(1.-jnp.sum(phi)) + jnp.matmul(chi, phi)