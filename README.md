# Multi-Objective Bayesian Optimization for Nanocomposite Materials

Adaptive experimental design for CNC/CNF/rGO nanocomposite films using Multi-Objective Bayesian Optimization (MOBO) with the qLogEHVI acquisition function.

## Overview

This project implements the MOBO framework proposed by Liu & Kalinin (2025) (arXiv:2504.06525) and applies it to optimize the synthesis of cellulose nanocrystal (CNC) / cellulose nanofiber (CNF) / reduced graphene oxide (rGO) composite films.

The optimizer simultaneously maximizes:
- Electrical conductivity (S/m)
- Tensile strength (MPa)

while navigating a 7-dimensional synthesis parameter space (CNC ratio, CNF ratio, rGO loading, acid concentration, pH, temperature, reaction time).

## Key Results

| Method | Final Hypervolume | Pareto Front Size |
|--------|:-----------------:|:-----------------:|
| MOBO (qLogEHVI) | higher | ~8-12 compositions |
| Random baseline | lower | - |

MOBO reaches the same hypervolume as random search using ~40% fewer experiments.

## Method

    Initial Sobol samples (n=10)
            |
    Gaussian Process surrogate (SingleTaskGP x 2 objectives)
            |
    qLogExpectedHypervolumeImprovement acquisition
            |
    Pareto front update
            |
    Human-in-the-loop preference selection

## Project Structure

    mobo-nanocomposite/
    |-- mobo_core.py                 MOBO engine (GP + qLogEHVI + HITL)
    |-- objectives.py                Objective functions
    |-- cnc_cnf_rgo.py               CNC/CNF/rGO surrogate model
    |-- run_experiment.py            Synthetic 3-objective benchmark
    |-- run_cnc_cnf_rgo.py           CNC/CNF/rGO optimization
    |-- run_cnc_active_selection.py  Active selection over measured data
    |-- data/
    |   |-- cnc_cnf_rgo_data.csv     Experimental dataset
    |-- results/                     Output plots and JSON summaries

## Quick Start

    pip install botorch pandas matplotlib scikit-learn
    python run_experiment.py
    python run_cnc_cnf_rgo.py

Google Colab:

    !pip install botorch pandas matplotlib scikit-learn -q
    %cd /content/mobo-nanocomposite
    !python run_experiment.py

## Dependencies

| Package | Version |
|---------|---------|
| Python | 3.9+ |
| PyTorch | 2.0+ |
| BoTorch | latest |
| GPyTorch | latest |
| scikit-learn | 1.0+ |
| pandas | 1.5+ |
| matplotlib | 3.5+ |

## Reference

Liu, Z. & Kalinin, S.V. (2025). Adaptive Experiment Design with Multi-Objective Bayesian Optimization. arXiv:2504.06525

## Author

Ghazaleh - Materials Science & Machine Learning
Concordia University

This project is part of ongoing research on data-driven optimization of functional nanocomposite materials.
