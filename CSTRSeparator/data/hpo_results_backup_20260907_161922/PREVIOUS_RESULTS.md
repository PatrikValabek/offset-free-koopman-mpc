# Previous HPO Results (backed up 2026-09-07)

Backup location: `CSTRSeparator/data/hpo_results_backup_20260907_161922/`

## Koopman C (`matrix_C=True`)
- **nz**: 24, encoder depth=2, GELU, nsteps=120, bs=40, lr≈3.7e-4
- **Dev MAE sum**: 9.83
- **Test MAE sum**: 13.03
  - T1=4.70, T2=4.05, T3=4.25, xB3=0.023

## Koopman noC (`matrix_C=False`, autoencoder)
- **nz**: 32, encoder depth=2, ELU, nsteps=120, bs=40, lr≈7.5e-4
- **Dev MAE sum**: 2.11
- **Test MAE sum**: 2.30
  - T1=0.71, T2=0.74, T3=0.85, xB3=0.0037

## SIPPY N4SID (orders 4–20)
- **Best order**: 13
- **Test MAE sum**: 10.57

## Conclusion from previous run
noC autoencoder model strongly outperformed both linear-C Koopman and SIPPY.
