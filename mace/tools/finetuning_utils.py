# Based on MACE (https://github.com/ACEsuit/mace, MIT License; see LICENSE.md)
# Modified for FieldMACE by Rhyan Barrett

import logging
import torch

from mace.tools.utils import AtomicNumberTable


def load_foundations(
    model: torch.nn.Module,
    model_foundations: torch.nn.Module,
    table: AtomicNumberTable,
    max_L=2,
):
    """
    Load the foundations of a model into a model for fine-tuning.
    """
    logging.debug(model)
    assert model_foundations.r_max == model.r_max
    z_table = AtomicNumberTable([int(z) for z in model_foundations.atomic_numbers])
    new_z_table = table
    num_species_foundations = len(z_table.zs)
    num_channels_foundation = (
        model_foundations.node_embedding.linear.weight.shape[0]
        // num_species_foundations
    )
    indices_weights = [z_table.z_to_index(z) for z in new_z_table.zs]
    num_radial = model.radial_embedding.out_dim
    num_species = len(indices_weights)
    max_ell = model.spherical_harmonics._lmax  # pylint: disable=protected-access
    model.node_embedding.linear.weight = torch.nn.Parameter(
        model_foundations.node_embedding.linear.weight.view(
            num_species_foundations, -1
        )[indices_weights, :]
        .flatten()
        .clone()
        / (num_species_foundations / num_species) ** 0.5
    )
    if model.radial_embedding.bessel_fn.__class__.__name__ == "BesselBasis":
        model.radial_embedding.bessel_fn.bessel_weights = torch.nn.Parameter(
            model_foundations.radial_embedding.bessel_fn.bessel_weights.clone()
        )

    for i in range(int(model.num_interactions)):
        model.interactions[i].linear_up.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear_up.weight.clone()
        )
        model.interactions[i].avg_num_neighbors = model_foundations.interactions[
            i
        ].avg_num_neighbors
        n_layers = sum(1 for name, _ in model.interactions[i].conv_tp_weights.named_children()
                       if name.startswith("layer"))
        for j in range(n_layers):
            layer_name = f"layer{j}"
            if j == 0:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        )
                        .weight[:num_radial, :]
                        .clone()
                    )
                )
            else:
                getattr(model.interactions[i].conv_tp_weights, layer_name).weight = (
                    torch.nn.Parameter(
                        getattr(
                            model_foundations.interactions[i].conv_tp_weights,
                            layer_name,
                        ).weight.clone()
                    )
                )

        model.interactions[i].linear.weight = torch.nn.Parameter(
            model_foundations.interactions[i].linear.weight.clone()
        )
        if (
            model.interactions[i].__class__.__name__
            == "RealAgnosticResidualInteractionBlock"
        ):
            model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                model_foundations.interactions[i]
                .skip_tp.weight.reshape(
                    num_channels_foundation,
                    num_species_foundations,
                    num_channels_foundation,
                )[:, indices_weights, :]
                .flatten()
                .clone()
                / (num_species_foundations / num_species) ** 0.5
            )
        else:
            try:
                model.interactions[i].skip_tp.weight = torch.nn.Parameter(
                    model_foundations.interactions[i]
                    .skip_tp.weight.reshape(
                        num_channels_foundation,
                        (max_ell + 1),
                        num_species_foundations,
                        num_channels_foundation,
                    )[:, :, indices_weights, :]
                    .flatten()
                    .clone()
                    / (num_species_foundations / num_species) ** 0.5
                )
            except:
                pass

    for i in range(min(len(model.products), len(model_foundations.products))):
        n_contr = min(len(model.products[i].symmetric_contractions.contractions),
                      len(model_foundations.products[i].symmetric_contractions.contractions))
        for j in range(n_contr):
            model.products[i].symmetric_contractions.contractions[j].weights_max = (
                torch.nn.Parameter(
                    model_foundations.products[i]
                    .symmetric_contractions.contractions[j]
                    .weights_max[indices_weights, :, :]
                    .clone()
                )
            )

            for k in range(len(model.products[i].symmetric_contractions.contractions[j].weights)):
                model.products[i].symmetric_contractions.contractions[j].weights[k] = (
                    torch.nn.Parameter(
                        model_foundations.products[i]
                        .symmetric_contractions.contractions[j]
                        .weights[k][indices_weights, :, :]
                        .clone()
                    )
                )

        model.products[i].linear.weight = torch.nn.Parameter(
            model_foundations.products[i].linear.weight.clone()
        )
    if hasattr(model_foundations, "multipole_max_ell"):  # a FieldMACE model: also its field and readout layers
        _transfer_field_layers(model, model_foundations)
    logging.info("Foundation model transferred")
    logging.debug(model)
    return model


# element-independent parts of a field model; copied when the shapes match
_FIELD_PARTS = ("multipole_radial_embedding.", "linear_up.", "field_interactions.", "readouts.", "property_readouts.",
                "odd_readouts.", "odd_dipole_readouts.")


def _transfer_field_layers(model: torch.nn.Module, foundation: torch.nn.Module) -> None:
    src = foundation.state_dict()
    dst = model.state_dict()
    copied, skipped = 0, []
    for name, value in dst.items():
        if not name.startswith(_FIELD_PARTS):
            continue
        if name in src and src[name].shape == value.shape:
            dst[name] = src[name].clone().to(value.dtype)
            copied += 1
        else:
            skipped.append(name)
    model.load_state_dict(dst)
    logging.info(f"Field layers of the foundation model transferred ({copied} tensors)"
                 + (f"; not transferred (shape differs): {skipped}" if skipped else ""))
