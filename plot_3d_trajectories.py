import pandas as pd

import torch
import numpy as np
from network import FixTypes
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_single_3d_trajectories_by_layer(model_path, all_weights_path, status_type: FixTypes):
    """
    This plots one PCA for every net (over its n epochs) as one trajectory
    and then combines all of them in one plot
    """
    model = torch.load(model_path, map_location=torch.device('cpu')).eval()
    all_weights = pd.read_csv(all_weights_path, index_col=False)
    save_path = model_path.parent / 'trajec_plots'

    all_epochs = all_weights.Epoch.unique()
    pca = PCA(n_components=2, whiten=True)
    save_path.mkdir(exist_ok=True, parents=True)

    for layer_idx, model_layer in enumerate(model.all_layers):
        
        fixpoint_statuses = [net.is_fixpoint for net in model_layer.particles]
        num_status_of_layer = sum([net.is_fixpoint == status_type for net in model_layer.particles])
        if num_status_of_layer != 0:
            layer = all_weights[all_weights.Weight.str.startswith(f"L{layer_idx}")]
            weight_batches = [np.array(layer[layer.Weight == name].values.tolist())[:, 2:]
                              for name in layer.Weight.unique()]
            plt.clf()
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            fig.set_figheight(10)
            fig.set_figwidth(12)
            plt.tight_layout()

            for weights_of_net, status in zip(weight_batches, fixpoint_statuses):
                if status == status_type:
                    pca.fit(weights_of_net)
                    transformed_trajectory = pca.transform(weights_of_net)
                    xdata = transformed_trajectory[:, 0]
                    ydata = transformed_trajectory[:, 1]
                    zdata = all_epochs
                    ax.plot3D(xdata, ydata, zdata)
                    ax.scatter(xdata, ydata, zdata, s=7)

            ax.set_title(f"Layer {layer_idx}: {num_status_of_layer}-{status_type}", fontsize=20)
            ax.set_xlabel('PCA Transformed x-axis', fontsize=20)
            ax.set_ylabel('PCA Transformed y-axis', fontsize=20)
            ax.set_zlabel('Epochs', fontsize=30, rotation=0)
            file_path = save_path / f"layer_{layer_idx}_{num_status_of_layer}_{status_type}.png"
            plt.savefig(file_path, bbox_inches="tight", dpi=300, format="png")
            plt.clf()
            plt.close(fig)


def plot_grouped_3d_trajectories_by_layer(model_path, all_weights_path, status_type: FixTypes):
    """ This computes the PCA over all the net-weights at once and then plots that."""

    model = torch.load(model_path, map_location=torch.device('cpu')).eval()
    save_path = model_path.parent / 'trajec_plots'
    all_weights = pd.read_csv(all_weights_path, index_col=False)
    all_epochs = all_weights.Epoch.unique()
    pca = PCA(n_components=2, whiten=True)
    save_path.mkdir(exist_ok=True, parents=True)
    
    for layer_idx, model_layer in enumerate(model.all_layers):
        
        fixpoint_statuses = [net.is_fixpoint for net in model_layer.particles]
        num_status_of_layer = sum([net.is_fixpoint == status_type for net in model_layer.particles])
        if num_status_of_layer != 0:
            layer = all_weights[all_weights.Weight.str.startswith(f"L{layer_idx}")]
            weight_batches = np.vstack([np.array(layer[layer.Weight == name].values.tolist())[:, 2:]
                                        for name in layer.Weight.unique()])
            plt.clf()
            fig = plt.figure()
            fig.set_figheight(10)
            fig.set_figwidth(12)
            ax = plt.axes(projection='3d')
            plt.tight_layout()

            pca.fit(weight_batches)
            w_transformed = pca.transform(weight_batches)
            for transformed_trajectory, status in zip(
                    np.split(w_transformed, len(layer.Weight.unique())), fixpoint_statuses):
                if status == status_type:
                    xdata = transformed_trajectory[:, 0]
                    ydata = transformed_trajectory[:, 1]
                    zdata = all_epochs
                    ax.plot3D(xdata, ydata, zdata)
                    ax.scatter(xdata, ydata, zdata, s=7)

            ax.set_title(f"Layer {layer_idx}: {num_status_of_layer}-{status_type}", fontsize=20)
            ax.set_xlabel('PCA Transformed x-axis', fontsize=20)
            ax.set_ylabel('PCA Transformed y-axis', fontsize=20)
            ax.set_zlabel('Epochs', fontsize=30, rotation=0)
            file_path = save_path / f"layer_{layer_idx}_{num_status_of_layer}_{status_type}_grouped.png"
            plt.savefig(file_path, bbox_inches="tight", dpi=300, format="png")
            plt.clf()
            plt.close(fig)


if __name__ == '__main__':
    raise (NotImplementedError('Get out of here'))
    """
    weight_path = Path("weight_store.csv")
    model_path = Path("trained_model_ckpt_e100.tp")
    save_path = Path("figures/3d_trajectories/")

    weight_df = pd.read_csv(weight_path)
    weight_df = weight_df.drop_duplicates(subset=['Weight','Epoch'])
    model = torch.load(model_path, map_location=torch.device('cpu'))

    plot_single_3d_trajectories_by_layer(model, weight_df, save_path, status_type=FixTypes.identity_func)
    plot_single_3d_trajectories_by_layer(model, weight_df, save_path, status_type=FixTypes.other_func)
    plot_grouped_3d_trajectories_by_layer(model, weight_df, save_path, FixTypes.identity_func)
    #plot_grouped_3d_trajectories_by_layer(model, weight_df, save_path, FixTypes.other_func)
    """