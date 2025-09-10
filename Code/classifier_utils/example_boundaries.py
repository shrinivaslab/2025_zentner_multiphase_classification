import jax
import jax.numpy as jnp

#--------------------Functions for custom decision boundaries in 2D input space--------------------#

def get_dataset_2D(boundary_type, max_input, dataset_length=1000, seed=0):
    '''
    Description: generate input concentration vectors and the corresponding output indices.

    Parameters:
        boundary_type (string): the boundary with which the input concentration vector should be compared
        max_input (float): the maximum concentration of any given input species
        dataset_length (int): the number of input concentration vectors to generate
        seed (int): the random seed to use in generating the input concentration vectors

    Returns: the input concentration vector and the index of the target output species to which they correspond
    '''
    n_phases = 2 if boundary_type!='triangle' else 3

    key = jax.random.PRNGKey(seed)
    key, split = jax.random.split(key)
    data_in = jax.random.uniform(split, (10*n_phases*dataset_length, 2), minval=0., maxval=max_input)
    dataset_indices = jax.vmap(get_labels_2d, in_axes=(0,None,None))(data_in, boundary_type, max_input)

    data_idxs_trimmed = jnp.zeros(n_phases*int(dataset_length/n_phases), dtype=jnp.int32)
    idx_positions = jnp.arange(len(dataset_indices), dtype=jnp.int32)
    for i in range(n_phases):
        idxs = idx_positions[dataset_indices==i][:int(dataset_length/n_phases)]
        data_idxs_trimmed = data_idxs_trimmed.at[i*int(dataset_length/n_phases):(i+1)*int(dataset_length/n_phases)].set(idxs)

    return data_in[data_idxs_trimmed], dataset_indices[data_idxs_trimmed]

def get_labels_2d(phi_in, boundary_type, max_input):
    '''
    Description: obtain the output index for a given input concentration vector.

    Parameters:
        phi_in (array): the vector of input concentrations
        boundary_type (string): the boundary with which the input concentration vector should be compared
        max_input (float): the maximum concentration of any given input species

    Returns: the index of the target output species to which the input concentration vector corresponds
    '''
    if boundary_type=='linear':
        return jnp.int32(jnp.heaviside(jnp.diff(phi_in)[0],1))
    if boundary_type=='arbitrary_linear':
        return jnp.int32(jnp.heaviside(phi_in[1] - (0.3-2*phi_in[0]), 1))
    elif boundary_type=='and':
        return jnp.int32(jnp.heaviside(phi_in[0]-max_input/2,1)*jnp.heaviside(phi_in[1]-max_input/2,1))
    elif boundary_type=='corner':
        return jnp.int32(jnp.heaviside(phi_in[0]-max_input/2,1)*jnp.heaviside(max_input/2-phi_in[1],1))
    elif boundary_type=='or':
        return jnp.int32(jnp.heaviside(max_input/2-phi_in[0],1)*jnp.heaviside(max_input/2-phi_in[1],1))
    elif boundary_type=='xor':
        and_term = jnp.int32(jnp.heaviside(phi_in[0]-max_input/2,1)*jnp.heaviside(phi_in[1]-max_input/2,1))
        or_term = jnp.int32(jnp.heaviside(max_input/2-phi_in[0],1)*jnp.heaviside(max_input/2-phi_in[1],1))
        return and_term + or_term
    elif boundary_type=='circle':
        r=max_input/3
        return jnp.int32(jnp.heaviside(r - jnp.sqrt((phi_in[0]-max_input/2)**2 + (phi_in[1]-max_input/2)**2), 1))
    elif boundary_type=='parabola':
        #return jnp.int32(jnp.heaviside((max_input/2)-(1/max_input)*(phi_in[0]-max_input/2)**2 - phi_in[1], 1))
        return jnp.int32(jnp.heaviside((3*(phi_in[0]-0.05)**2 + max_input/2) - phi_in[1], 1))
    elif boundary_type=='sine':
        return jnp.int32(jnp.heaviside((max_input/2)*(1+jnp.sin((3/2)*2*jnp.pi*(phi_in[0]/max_input-0.2))) - phi_in[1], 1))
    elif boundary_type=='tanh':
        return jnp.int32(jnp.heaviside((max_input/4)*(1+jnp.tanh(60*(phi_in[0]-max_input/2))) - phi_in[1], 1))
    elif boundary_type=='checkerboard':
        condition_1 = jnp.int32(jnp.heaviside(phi_in[0]-max_input/3,1)*jnp.heaviside(2*max_input/3-phi_in[0],1))
        condition_2 = jnp.int32(jnp.heaviside(phi_in[1]-max_input/3,1)*jnp.heaviside(2*max_input/3-phi_in[1],1))
        return condition_1^condition_2
    elif boundary_type=='triangle':
        vecs = jnp.array([[1,1],[-2,1],[-1,3]])
        r0 = jnp.array([max_input/2,max_input/2])
        cond1 = jnp.int32(jnp.heaviside(jnp.dot(phi_in-r0,vecs[1]),1)*jnp.heaviside(jnp.dot(phi_in-r0,vecs[2]),1))
        cond2 = jnp.int32(jnp.heaviside(-jnp.dot(phi_in-r0,vecs[0]),1)*jnp.heaviside(-jnp.dot(phi_in-r0,vecs[2]),1))
        cond3 = jnp.int32(jnp.heaviside(jnp.dot(phi_in-r0,vecs[0]),1)*jnp.heaviside(-jnp.dot(phi_in-r0,vecs[1]),1))
        return jnp.nonzero(jnp.array([cond1,cond2,cond3]), size=1)[0][0]

#--------------------Functions for decision boundaries formed by hyperplanes in n-dimensional input space--------------------#

def get_dataset_nd(in_out_dimensions, max_input, vecs, r0, dataset_length=1000, seed=0):
    '''
    Description: generate input concentration vectors and the corresponding output indices.

    Parameters:
        boundary_type (string): the boundary with which the input concentration vector should be compared
        max_input (float): the maximum concentration of any given input species
        dataset_length (int): the number of input concentration vectors to generate
        seed (int): the random seed to use in generating the input concentration vectors

    Returns: the input concentration vector and the index of the target output species to which they correspond
    '''
    possible_labels = get_regions(max_input, vecs, r0)
    n_phases = len(possible_labels)

    key = jax.random.PRNGKey(seed)
    key, split = jax.random.split(key)
    data_in = jax.random.uniform(split, (10*n_phases*dataset_length, in_out_dimensions[0]), minval=0., maxval=max_input)
    '''Added for testing'''
    #center = jnp.matmul(jnp.linalg.inv(vecs), jnp.array([jnp.dot(vecs[0], r0[0]), jnp.dot(vecs[1], r0[1])]))
    #data_in = data_in[jnp.linalg.norm(jnp.array([d-center for d in data_in]), axis=1)>0.05]
    '''End of added'''
    dataset_indices = get_labels_nd(data_in, possible_labels, vecs, r0)

    data_idxs_trimmed = jnp.zeros(n_phases*int(dataset_length/n_phases), dtype=jnp.int32)
    idx_positions = jnp.arange(len(dataset_indices), dtype=jnp.int32)
    for i in range(n_phases):
        idxs = idx_positions[dataset_indices==i][:int(dataset_length/n_phases)]
        data_idxs_trimmed = data_idxs_trimmed.at[i*int(dataset_length/n_phases):(i+1)*int(dataset_length/n_phases)].set(idxs)

    return data_in[data_idxs_trimmed], dataset_indices[data_idxs_trimmed], 

def get_regions(max_input, vecs, r0): # Only set up for 2D input space

    # Generate grid of sampling points
    lattice_pts = jnp.meshgrid(*(len(vecs[0])*[jnp.linspace(0, max_input, 50)]))
    phi_in = jnp.array([x.reshape(-1) for x in lattice_pts]).T

    # Get condition for each sampling point and determine number of regions in input space
    conditions = jax.vmap(lambda phi_in: jnp.int32(jnp.heaviside(jnp.diag((phi_in-r0) @ vecs.T), 1)))(phi_in)
    possible_labels = jnp.unique(conditions, axis=0)

    return possible_labels

def get_labels_nd(phi_in, possible_labels, vecs, r0):
    '''
    Description: obtain the output index for a given input concentration vector.

    Parameters:
        phi_in (array): the vector of input concentrations
        boundary_type (string): the boundary with which the input concentration vector should be compared
        max_input (float): the maximum concentration of any given input species

    Returns: the index of the target output species to which the input concentration vector corresponds
    '''
    conditions = jax.vmap(lambda phi_in: jnp.int32(jnp.heaviside(jnp.diag((phi_in-r0) @ vecs.T), 1)))(phi_in)

    labels = jnp.zeros(len(conditions), dtype=jnp.int32)
    for i in range(len(possible_labels)):
        labels = labels.at[jnp.where(jnp.all(conditions==possible_labels[i], axis=1))].set(i)

    return labels