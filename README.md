[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/g5Rk6CRe)
[![Run Notebook](https://github.com/eisenhauerIO/projects-businss-decisions/actions/workflows/run-notebook.yml/badge.svg)](https://github.com/eisenhauerIO/projects-businss-decisions/actions/workflows/run-notebook.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

### Replication project
Bruce Zhou, ECON 481, Spring 2026

---

[replication_project.ipynb](replication_project.ipynb) is a replication of the following paper:

Angrist, J. D., & Lavy, V. (1999). [Using Maimonides' rule to estimate the effect of class size on scholastic achievement](https://academic.oup.com/qje/article-abstract/114/2/533/1844228). *The Quarterly Journal of Economics*, 114(2), 533–575.

#### Project Structure

```
project/
├── auxiliary/
│   └── dag.py                       # Create DAG graphs
│   ├── dataframe_analysis.py        # Father class
│   └── plots.py                     # Create plots
│   └── tables.py                    # Create tables
├── data/
│   ├── final4.dta/                  # Data on 4th graders
│   └── final5.dta/                  # Data on 5th graders
├── paper/
│   ├── AngristLavy1999.pdf          # The original article
│   └── read_paper.py                # AI access to the article
├── png
│   ├── angrist_lavy_dag_no_z.png    # DAG with no IV
│   └── angrist_lavy_dag.png         # DAG with IV
└── replication_project.ipynb        # Main project
```

---

#### Dependencies

```bash
pip install pandas numpy matplotlib statsmodels great_tables importlib
```