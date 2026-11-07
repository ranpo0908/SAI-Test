# SAI experiment report

Repetitions per configuration: 50

Significance level: 0.05; solver: full

Total repetitions: 1400; configurations: 28

Raw results and both RNG seeds are retained in `detail.csv`.

## Configuration summaries

| study | case | n | B | a | alpha | solver | n_rep | reject_rate | mean_p_value | mean_statistic | mean_critical_value | mean_runtime_sec | max_violation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gaussian_near | gaussian_near | 50 | 199 | nan | 0.050000 | full | 50 | 0.000000 | 0.518000 | 0.071765 | 0.132613 | 3.568377 | 0.000000 |
| gaussian_near | gaussian_near | 150 | 199 | nan | 0.050000 | full | 50 | 0.240000 | 0.182400 | 0.070019 | 0.081086 | 17.561583 | 0.000000 |
| gaussian_near | gaussian_near | 300 | 199 | nan | 0.050000 | full | 50 | 0.560000 | 0.062900 | 0.062950 | 0.059463 | 101.780748 | 0.000000 |
| gaussian_near | gaussian_near | 500 | 199 | nan | 0.050000 | full | 50 | 0.920000 | 0.018400 | 0.059309 | 0.047737 | 487.888886 | 0.000000 |
| gaussian_table | gaussian_alt | 50 | 19 | nan | 0.050000 | full | 50 | 1.000000 | 0.050000 | 0.245905 | 0.130722 | 0.321179 | 0.000000 |
| gaussian_table | gaussian_alt | 50 | 99 | nan | 0.050000 | full | 50 | 1.000000 | 0.010200 | 0.245106 | 0.115348 | 1.722332 | 0.000000 |
| gaussian_table | gaussian_alt | 50 | 199 | nan | 0.050000 | full | 50 | 1.000000 | 0.005000 | 0.243940 | 0.115047 | 3.276836 | 0.000000 |
| gaussian_table | gaussian_alt | 50 | 499 | nan | 0.050000 | full | 50 | 1.000000 | 0.002000 | 0.245757 | 0.115343 | 8.325244 | 0.000000 |
| gaussian_table | gaussian_alt | 100 | 19 | nan | 0.050000 | full | 50 | 1.000000 | 0.050000 | 0.244174 | 0.091659 | 0.779367 | 0.000000 |
| gaussian_table | gaussian_alt | 100 | 99 | nan | 0.050000 | full | 50 | 1.000000 | 0.010000 | 0.242540 | 0.086620 | 3.946577 | 0.000000 |
| gaussian_table | gaussian_alt | 100 | 199 | nan | 0.050000 | full | 50 | 1.000000 | 0.005000 | 0.241005 | 0.084833 | 7.704660 | 0.000000 |
| gaussian_table | gaussian_alt | 100 | 499 | nan | 0.050000 | full | 50 | 1.000000 | 0.002000 | 0.237917 | 0.083645 | 19.642983 | 0.000000 |
| gaussian_table | gaussian_alt | 200 | 19 | nan | 0.050000 | full | 50 | 1.000000 | 0.050000 | 0.239289 | 0.068518 | 3.183598 | 0.000000 |
| gaussian_table | gaussian_alt | 200 | 99 | nan | 0.050000 | full | 50 | 1.000000 | 0.010000 | 0.235492 | 0.061050 | 16.231086 | 0.000000 |
| gaussian_table | gaussian_alt | 200 | 199 | nan | 0.050000 | full | 50 | 1.000000 | 0.005000 | 0.240855 | 0.061214 | 34.021259 | 0.000000 |
| gaussian_table | gaussian_alt | 200 | 499 | nan | 0.050000 | full | 50 | 1.000000 | 0.002000 | 0.245214 | 0.062375 | 82.076527 | 0.000000 |
| gaussian_table | gaussian_null | 50 | 19 | nan | 0.050000 | full | 50 | 0.000000 | 0.999000 | 0.015834 | 0.138194 | 0.326354 | 0.000000 |
| gaussian_table | gaussian_null | 50 | 99 | nan | 0.050000 | full | 50 | 0.000000 | 0.999800 | 0.014748 | 0.122149 | 2.024873 | 0.000000 |
| gaussian_table | gaussian_null | 50 | 199 | nan | 0.050000 | full | 50 | 0.000000 | 0.999500 | 0.015509 | 0.122530 | 3.773041 | 0.000000 |
| gaussian_table | gaussian_null | 50 | 499 | nan | 0.050000 | full | 50 | 0.000000 | 0.999720 | 0.014624 | 0.121871 | 9.395616 | 0.000000 |
| gaussian_table | gaussian_null | 100 | 19 | nan | 0.050000 | full | 50 | 0.000000 | 1.000000 | 0.011402 | 0.102427 | 0.981299 | 0.000000 |
| gaussian_table | gaussian_null | 100 | 99 | nan | 0.050000 | full | 50 | 0.000000 | 1.000000 | 0.011483 | 0.092787 | 4.522990 | 0.000000 |
| gaussian_table | gaussian_null | 100 | 199 | nan | 0.050000 | full | 50 | 0.000000 | 1.000000 | 0.011017 | 0.091305 | 9.325512 | 0.000000 |
| gaussian_table | gaussian_null | 100 | 499 | nan | 0.050000 | full | 50 | 0.000000 | 1.000000 | 0.011455 | 0.090592 | 23.190401 | 0.000000 |
| gaussian_table | gaussian_null | 200 | 19 | nan | 0.050000 | full | 50 | 0.000000 | 1.000000 | 0.008759 | 0.071775 | 4.028797 | 0.000000 |
| gaussian_table | gaussian_null | 200 | 99 | nan | 0.050000 | full | 50 | 0.000000 | 1.000000 | 0.008902 | 0.068068 | 20.821132 | 0.000000 |
| gaussian_table | gaussian_null | 200 | 199 | nan | 0.050000 | full | 50 | 0.000000 | 1.000000 | 0.009134 | 0.067516 | 41.203755 | 0.000000 |
| gaussian_table | gaussian_null | 200 | 499 | nan | 0.050000 | full | 50 | 0.000000 | 1.000000 | 0.008322 | 0.066916 | 101.276336 | 0.000000 |


## Tables

- [table1_gaussian.tex](tables/table1_gaussian.tex)


## Figures

- [plot_alt_var_reject_vs_n.pdf](figures/plot_alt_var_reject_vs_n.pdf)
