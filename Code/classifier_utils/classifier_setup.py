import jax.numpy as jnp
import jax

import pdb

jax.config.update("jax_enable_x64", True)

class DataSetup_Custom:
    
    def __init__(self, data_in, dataset_indices, n_outputs, n_hidden, seed=0):
        '''
        Description: Given input concentration data and corresponding target output indices, construct full concentration vectors.
                     An object of this class contains all data needed for training and testing.
        
        Parameters:
            data_in (array): vectors of input concentrations
            dataset_indices (array): target output indicess
            n_outputs (int): the number of output species
            n_hidden (int): the number of hidden species
            seed (int): the random seed to use in generating output and hidden concentrations
        '''
        self.n_inputs = (data_in.shape)[-1]
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_components = self.n_inputs + self.n_outputs + self.n_hidden
        self.data_in, self.data_out_and_hidden, self.dataset_indices = self.get_dataset(seed, data_in, dataset_indices)
        if dataset_indices!=None:
            self.class_index_range = jnp.array([(jnp.min(jnp.where(self.dataset_indices == i)[0]), jnp.max(jnp.where(self.dataset_indices == i)[0])) for i in jnp.unique(dataset_indices)]).T
            self.min_range = jnp.min(self.class_index_range[1]-self.class_index_range[0])

    def get_dataset(self, seed, data_in, target_indices=None):
        '''
        Description: function for generating output and hidden concentration vectors.
        
        Parameters:
            seed (int): the random seed to use in generating output and hidden concentrations
            data_in (array): vectors of input concentrations
            target_indices (array): target output indices

        Returns: input concentration vectors, output and hidden concentration vectors, target output indices 
        '''
        data_length = len(data_in)
        
        key = jax.random.PRNGKey(seed)
        key, split = jax.random.split(key)
        phi_tot_vals = jax.random.uniform(split, (1,data_length), minval=0.75, maxval=0.95)[0]

        key, split = jax.random.split(key)
        data_out_and_hidden = jax.random.uniform(split, (data_length, self.n_outputs + self.n_hidden), minval=0, maxval=1)
        scalings = (phi_tot_vals - jnp.sum(data_in,1))/jnp.sum(data_out_and_hidden,1)
        data_out_and_hidden = data_out_and_hidden * scalings[:,None]

        return data_in, data_out_and_hidden, target_indices

    def get_batch(self, seed, batch_length):
        '''
        Description: select a finite batch out of the full training set for stochastic gradient descent

        Parameters:
            seed (int): the random seed to use in selecting the training batch
            batch_length (int): number of training points to use in each optimization epoch
        
        Returns: the input concentration vectors, output+hidden concentration vectors, and target indices for the selected batch
        '''
        length_per_class = int(batch_length/self.n_outputs)

        key = jax.random.PRNGKey(seed)
        split = jax.random.split(key, num = self.n_outputs)
        random_choice_vmapped = jax.vmap(lambda min_idx, split_val: min_idx + jax.random.choice(split_val, jnp.arange(0, self.min_range), (1,length_per_class), replace=False)[0])
        batch_indices = random_choice_vmapped(self.class_index_range[0], split).reshape(-1)
        
        # Set fixed input concentrations for each example in the training set:
        phi_inputs = self.data_in[batch_indices]
        phi_out_and_hidden = self.data_out_and_hidden[batch_indices]
        target_output_labels = self.dataset_indices[batch_indices]

        return phi_inputs, phi_out_and_hidden, target_output_labels