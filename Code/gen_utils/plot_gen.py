import jax.numpy as jnp
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

def plot_loss_and_outputs(loss_vals, outputs, sizes, max_chi=15, resParamType='mu'):

    χmat_opt, res_opt = outputs

    n_inputs, n_outputs, n_hidden = sizes
    n_components = n_inputs + n_outputs + n_hidden

    fig, ax = plt.subplots(1,3, figsize=(14,4), gridspec_kw={'width_ratios': [1, 0.9, 0.2]})

    ax[0].plot(loss_vals, color='black')
    #ax[0].set_yscale('log')
    ax[0].set_xlabel('Optimization Epoch')
    ax[0].set_ylabel('Loss')

    im = ax[1].imshow(χmat_opt, vmin=-max_chi, vmax=max_chi, cmap='coolwarm')
    plt.colorbar(im, ax=ax[1], shrink=1, label=r'$\chi$')

    x = [[n_inputs-0.5, n_inputs-0.5], [n_inputs+n_outputs-0.5, n_inputs+n_outputs-0.5], [-0.5,n_components-0.5], [-0.5,n_components-0.5]]
    y = [[-0.5,n_components-0.5], [-0.5,n_components-0.5], [n_inputs-0.5, n_inputs-0.5], [n_inputs+n_outputs-0.5, n_inputs+n_outputs-0.5]]
    for i in range(4):
        ax[1].plot(x[i], y[i], color="black", linewidth=1)

    ax[1].set_xlabel('Species')
    ax[1].set_ylabel('Species')

    im = ax[2].imshow(jnp.array([res_opt]).T, vmin = -5, vmax = 0, cmap='gray')
    plt.colorbar(im, ax=ax[2], shrink=1, label=r'$\vec{\mu}_\text{ref}$')
    ax[2].get_xaxis().set_visible(False)

    fig.tight_layout()

    return fig

def get_boundary_lines(boundary_type, max_input):
    #-----Boundary Types-----#
    if boundary_type=='linear':
        line1 = jnp.array([jnp.linspace(0.,max_input,10),jnp.linspace(0.,max_input,10)]).T
        return [line1]
    elif boundary_type=='and':
        line1 = jnp.array([jnp.linspace(max_input/2,max_input,10),(max_input/2)*jnp.ones(10)]).T
        line2 = jnp.array([(max_input/2)*jnp.ones(10), jnp.linspace(max_input/2,max_input,10)]).T
        return [line1, line2]
    elif boundary_type=='or':
        line1 = jnp.array([jnp.linspace(0,max_input/2,10), (max_input/2)*jnp.ones(10)]).T
        line2 = jnp.array([(max_input/2)*jnp.ones(10), jnp.linspace(0,max_input/2,10)]).T
        return [line1, line2]
    elif boundary_type=='xor':
        line1 = jnp.array([jnp.linspace(0,max_input,10), (max_input/2)*jnp.ones(10)]).T
        line2 = jnp.array([(max_input/2)*jnp.ones(10), jnp.linspace(0,max_input,10)]).T
        return [line1, line2]
    elif boundary_type=='circle':
        r=max_input/3
        p0 = (max_input/2)*jnp.ones(2)
        xvals = p0[0] + jnp.linspace(-0.999*r,0.999*r,100)
        line1 = jnp.array([xvals, p0[1] + jnp.sqrt(r**2 - (xvals-p0[0])**2)]).T
        line2 = jnp.array([xvals, p0[1] - jnp.sqrt(r**2 - (xvals-p0[0])**2)]).T
        return [line1, line2]
    elif boundary_type=='sine':
        xvals = jnp.linspace(0,max_input,100)
        line1 = jnp.array([xvals,(max_input/2)*(1+jnp.sin((3/2)*2*jnp.pi*(xvals/max_input-0.2)))]).T
        return [line1]
    elif boundary_type=='checkerboard':
        line1 = jnp.array([jnp.linspace(0,max_input,10), (max_input/3)*jnp.ones(10)]).T
        line2 = jnp.array([jnp.linspace(0,max_input,10), (2*max_input/3)*jnp.ones(10)]).T
        line3 = jnp.array([(max_input/3)*jnp.ones(10), jnp.linspace(0,max_input,10)]).T
        line4 = jnp.array([(2*max_input/3)*jnp.ones(10), jnp.linspace(0,max_input,10)]).T
        return [line1, line2, line3, line4]
    elif boundary_type=='triangle':
        xvals = jnp.linspace(0,max_input/2,100)
        p0 = jnp.array([max_input/2,max_input/2])
        line1 = jnp.array([jnp.linspace(0,max_input/2,100), p0[1] + (1/3)*(xvals-p0[0])]).T
        xvals = jnp.linspace(max_input/2,max_input,100)
        line2 = jnp.array([jnp.linspace(max_input/2,max_input,100), p0[1] + 2*(xvals-p0[0])]).T
        line3 = jnp.array([jnp.linspace(max_input/2,max_input,100), p0[1] - 1*(xvals-p0[0])]).T
        return [line1, line2, line3]
    else:
        return None

def plot_classifier_guesses(phi_inputs, indices_pred, boundary_type, max_input, phase_plot=False):

    node_pink = '#ef559f'
    node_green = '#3ab261'

    fig, ax = plt.subplots(1, 1, figsize=(4,4))

    classes = jnp.unique(indices_pred)
    num_classes = len(classes)

    ax.plot(*phi_inputs[indices_pred==classes[0]].T,'o',color=node_green)
    ax.plot(*phi_inputs[indices_pred==classes[1]].T,'o',color=node_pink)
    for i in range(2,num_classes-1):
        ax.plot(*phi_inputs[indices_pred==classes[i]].T,'o')
    if phase_plot==False:
        ax.plot(*phi_inputs[indices_pred==classes[-1]].T,'o',color='gray')

    lines = get_boundary_lines(boundary_type, max_input)
    if lines!=None:
        for l in lines:
            ax.plot(l[:,0], l[:,1],'--',color='black')

    ax.set_xlim(0,max_input)
    ax.set_ylim(0,max_input)
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    ax.set_xticks([0,max_input/2, max_input])
    ax.set_yticks([0,max_input/2, max_input])
    if phase_plot==False:
        ax.set_title('Predicted Classes')
    else:
        ax.set_title('Clustered by Phase')

    return fig

def plot_clustering(phi_inputs, cluster_indices, boundary_type, max_input):

    fig, ax = plt.subplots(1,1, figsize=(8,8))

    num_phases = len(jnp.unique(cluster_indices))

    for i in range(num_phases):
        ax.plot(*phi_inputs[cluster_indices==i+1].T,'o')

    lines = get_boundary_lines(boundary_type, max_input)
    if lines!=None:
        for l in lines:
            ax.plot(l[:,0], l[:,1],'--',color='gray')
    
    ax.set_xlim(0,max_input)
    ax.set_ylim(0,max_input)
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    ax.set_title('Model A Outputs (Grouped by Phase)')

    return fig

def plot_success_tracker(mu_diff_vals, phi_tot_vals):

    fig, ax = plt.subplots(1, 2, figsize=(14,6))

    #phi_tot_vals = jnp.sum(phi_vals,2)
    max_args = jnp.argmax(mu_diff_vals,1)

    ax[0].plot(mu_diff_vals[jnp.arange(len(max_args)),max_args], color='blue', label='max', linewidth=0.1)
    ax[0].plot(jnp.sqrt(jnp.mean(mu_diff_vals**2,1)), color='red', label='mean', linewidth=0.1)
    ax[0].set_yscale('log')
    ax[0].set_xlabel('Optimization Epoch')
    ax[0].set_ylabel('Chemical Potential Difference')
    ax[0].legend(frameon=False)

    ax[1].plot(jnp.min(phi_tot_vals,1), color='gray', label='min total', linewidth=0.1)
    ax[1].plot(jnp.max(phi_tot_vals,1), color='black', label='max total', linewidth=0.1)
    ax[1].plot(phi_tot_vals[jnp.arange(len(max_args)),max_args], color='blue', label='max mu diff.', linewidth=0.2)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Optimization Epoch')
    ax[1].set_ylabel('Corresponding Volume Fraction')
    ax[1].legend(frameon=False)

    fig.suptitle('Model A Dynamics: Success Tracker')
    fig.tight_layout()

    return fig