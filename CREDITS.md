# Credits and provenance

- Original project: John Meyor Medina Diaz,
  [PINN-2D-Biology](https://github.com/johnmeyor/PINN-2D-Biology).
- Pinned original commit: `5cfcce750f2c5f91b8172c4301c4e517faf9f0e7`.
- Original notebook SHA256:
  `c130aade83923ac7ba3eb18fab1c393c008e160a58385fb61552b0047eb377fc`.
- Corrected implementation, analytical boundary lifting, reference checks,
  experiments and portable packaging: September 2026 revision.

The boundary lifting is a standard way to impose essential boundary conditions,
not a claimed methodological invention. The series reference follows separation
of variables for this linear constant-coefficient equation.

## Software documentation

- [DeepXDE Hessian API](https://deepxde.readthedocs.io/en/latest/modules/deepxde.gradients.html)
- [DeepXDE source and licence](https://github.com/lululxvi/deepxde)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [Jupyter NBClient execution documentation](https://nbclient.readthedocs.io/en/latest/client.html)

Unless noted otherwise, repository source and documentation are released under
the [MIT License](LICENSE). Dependencies retain their own licences.
