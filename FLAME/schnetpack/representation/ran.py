import torch
from math import e as exponential
from torch.nn import Linear

__all__ = [
    "RAN"
]


class RAN(torch.nn.Module):
    def __init__(self, type: str = "direct_ran"):
        super().__init__()
        self.type = type
        if (self.type == "sumbatch_ran") or (self.type == "summolecule_ran"):
            self.atom_env = Linear(1, 10)

    def ran(self, atomic_numbers, neighbors, r_ij, type: str = "direct"):
        # tune bondsteps with atomic_numbers
        atom_num_exp = atomic_numbers.view(-1, atomic_numbers.size(1), 1)
        atom_nbh = atom_num_exp.expand(atom_num_exp.size(0), atom_num_exp.size(1), atom_num_exp.size(1))
        atom_nbh = torch.gather(atom_nbh, 1, neighbors)
        dist_edit = torch.exp(
            -0.05 * exponential * torch.pow(
                (
                    atom_num_exp * atom_nbh
                ), 1.01
            ) * torch.reciprocal(
                torch.pow(
                    (
                        atom_num_exp + atom_nbh
                    ), 1.1
                )
            )
        )
        if type == 'sumbatch':
            dist_edit_sum = torch.sum(dist_edit).view(-1, 1)
            atom_mod = self.atom_env(dist_edit_sum)
            atom_mod = torch.exp(-0.005 * torch.sum(atom_mod)).view(-1, 1, 1)
            dist_edit = dist_edit + atom_mod
        elif type == 'summolecule':
            dist_edit_sum = torch.sum(dist_edit, dim=(1, 2)).view(dist_edit.size(0), 1)
            atom_mod = self.atom_env(dist_edit_sum)
            atom_mod = torch.exp(-0.005 * torch.sum(atom_mod, dim=1)).view(-1, 1, 1)
            dist_edit = dist_edit + atom_mod

        r_ij = r_ij + dist_edit
        return r_ij

    def rane(self, atomic_numbers, neighbors, r_ij):
        atom_num_exp = atomic_numbers.view(-1, atomic_numbers.size(1), 1)
        atom_nbh = atom_num_exp.expand(atom_num_exp.size(0), atom_num_exp.size(1), atom_num_exp.size(1))
        atom_nbh = torch.gather(atom_nbh, 1, neighbors)

        # atomic_numbers is modified by bondsteps, then tunes bondsteps by atomic_numbers
        atom_mod = atom_num_exp * (atom_nbh + r_ij * 0.01)
        atom_mod_prod = atom_mod.clone()
        atom_mod_prod[atom_mod_prod == float(0)] = float(1)
        dist_edit = torch.exp(
            -0.01 * exponential * torch.log10(
                torch.pow(
                    torch.prod(
                        atom_mod_prod, dim=2
                    ), 1.01
                ) * torch.reciprocal(
                    torch.pow(
                        torch.sum(
                            atom_mod, dim=2
                        ), 1.1
                    )
                )
            )
        )
        dist_edit = dist_edit.unsqueeze(-1)
        atom_num_exp = atom_num_exp + dist_edit
        atom_nbh = atom_num_exp.expand(atom_num_exp.size(0), atom_num_exp.size(1), atom_num_exp.size(1))
        atom_nbh = torch.gather(atom_nbh, 1, neighbors)
        dist_edit = torch.exp(
            -0.4 * torch.exp(
                torch.pow(
                    atom_num_exp * atom_nbh, 1.01
                ) * torch.reciprocal(
                    torch.pow(
                        atom_num_exp + atom_nbh, 1.1
                    )
                )
            )
        )

        r_ij = r_ij + dist_edit
        return r_ij

    def forward(self, atomic_numbers, neighbors, r_ij):
        if self.type == 'direct_ran':
            outputs = self.ran(atomic_numbers, neighbors, r_ij)
        elif self.type == 'sumbatch_ran':
            outputs = self.ran(atomic_numbers, neighbors, r_ij, 'sumbatch')
        elif self.type == 'summolecule_ran':
            outputs = self.ran(atomic_numbers, neighbors, r_ij, 'summolecule')
        elif self.type == 'rane':
            outputs = self.rane(atomic_numbers, neighbors, r_ij)

        return outputs