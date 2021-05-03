# cristian_lenta - BA code

Changes made from last meeting:
- `batch_size` renamed to `log_step_size`
- changed `second order fixpoints`: now comparing first input with second output, without weight change of the network
- removed rounding of the weights (line 99, network.py): need full float precision

**Important**:
- now that we have full float precision, the robustness test is **never** failing for the noise values of `10^-6, 10^-7, 10^-8, 10^-9`. May just be coding mistake, I am not sure yet.
