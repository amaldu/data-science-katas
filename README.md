# Data Science Katas

A hands-on collection of **250+ exercises, cheatsheets, and interview prep materials** covering the core skills tested in Data Science and ML Engineering interviews.

Every section follows a **question / solution** format so you can practice actively, not just read passively.

If you find this useful, a **star** helps others discover it and keeps me motivated.

---

## What's Inside

| Section | Exercises | Format | Topics |
|---|---|---|---|
| [Python OOP](#python-oop) | 40+ | Notebooks + `.py` | Classes, inheritance, encapsulation, polymorphism, dunder methods, abstract classes |
| [SQL](#sql) | 30 sets | Notebooks + PostgreSQL | Joins, CTEs, window functions, subqueries + 22 real-world applied scenarios |
| [Machine Learning](#machine-learning) | 10+ notebooks, 28 cheatsheets | Notebooks + Markdown | EDA, regression, SVMs, trees, ensembles, PCA, metrics, feature engineering |
| [PyTorch](#pytorch) | 2 tutorials + cheatsheet | Notebooks | Tensors, model building, training loops, neural net fundamentals |
| [LangGraph](#langgraph) | 10 exercises | Notebooks | Graph construction, conditional/looping flows, ReAct agents, RAG agents |
| [A/B Testing](#ab-testing) | 5 notebooks + 3 cheatsheets | Notebooks + Markdown | Frequentist tests, Bayesian testing, multivariant tests, interview Q&A |
| [Algorithms & Data Structures](#algorithms--data-structures) | 10 implementations + theory | `.py` + Markdown | Search, sorting, arrays, linked lists, hash tables, heaps, Big-O |

---

## Python OOP

Practice object-oriented Python from first principles through advanced patterns.

```
python/
├── classes 101/         # 8 notebook pairs: instances, class vars, methods,
│                        #   inheritance, dunder methods, property decorators,
│                        #   abstract classes
├── encapsulation/       # 6 exercise pairs
├── polymorphism/        # 2 notebook pairs + 10 extra .py exercises
└── extra_inheritance/   # 8 .py exercise pairs
```

## SQL

8 foundational topics plus 22 real-world applied scenarios, all runnable against a local PostgreSQL container.

```
sql/
├── basic_exercises/     # CREATE/INSERT, SELECT/GROUP BY, UNION, JOINs,
│                        #   CTEs, subqueries, window functions
├── applied_exercises/   # 22 scenarios: events, trades, viewership,
│                        #   transactions, sessions, reviews, signups...
├── migrations/          # DDL/DML scripts for each applied exercise
├── theory_questions.ipynb
└── migrate.py           # One command to set up all tables
```

**Setup:** requires Docker. See [SQL setup instructions](#instructions-on-how-to-run-the-sql-exercises) below.

## Machine Learning

Cheatsheets and hands-on notebooks spanning the full ML interview syllabus.

```
machine_learning/
├── 00_eda/              # Univariate, bivariate, multivariate EDA guides
├── 01_regression/       # Linear, logistic, polynomial (notebooks + cheatsheets + QA)
├── 02_SVMs/             # Exercise, cheatsheet, QA
├── 03_decision_trees/   # Cheatsheet + QA
├── 04_ensemble_learning_random_forests/
├── 05_dimensionality_reduction/
├── 06_metrics/          # Regression + classification metrics
└── 07_feature_engineering/  # 9 cheatsheets: missing data, encoding, scaling,
                             #   outliers, feature selection, class imbalance,
                             #   splitting, pipelines, regularization
```

## PyTorch

Neural network fundamentals and hands-on PyTorch workflows.

```
pytorch/
├── intro_nn_cheatsheet.md         # Neurons, perceptrons, activations, backprop
└── tutorials/
    ├── 00_pytorch_fundamentals.ipynb  # Tensors and operations
    └── 01_pytorch_workflow.ipynb      # Model building and training loops
```

## LangGraph

Build agentic LLM applications from simple graphs to full RAG agents.

```
langgraph/
├── 01-05 graph exercises   # Sequential, conditional, looping graphs
└── agents/
    ├── 01_simple_bot        # Basic chatbot
    ├── 02_agent_bot         # Tool-using agent
    ├── 03_react_agent       # ReAct pattern
    ├── 04_agent_drafter     # Drafting agent
    └── 05_rag_agent         # Retrieval-augmented generation
```

## A/B Testing

End-to-end experimentation: theory, statistical tests, and hands-on notebooks.

```
ab_testing/
├── AB_TESTING_CHEATSHEET.md          # Full workflow: hypotheses to decisions
├── AB_TESTING_GUIDELINE.md           # Step-by-step with PICOT criteria
├── ab_testing_statistical_tests.md   # Z-test, t-test, chi-square, Mann-Whitney,
│                                     #   bootstrap, Bayesian, ANOVA...
├── 01_ecommerce_conversion_rate      # Frequentist conversion test
├── 02_saas_session_duration           # Session duration test
├── 03_bayesian_ab_testing            # Bayesian approach
├── 04_multivariant_landing_page      # Multivariant test
└── 05_ab_testing_interview_qa        # Interview Q&A
```

## Algorithms & Data Structures

Core implementations and theory notes for coding interviews.

```
algorithms_data_structures/
├── algorithms/          # linear search, binary search (iterative + recursive),
│                        #   quicksort, merge sort (array + linked list)
├── data_structures/     # arrays, linked lists, hash tables
└── NOTES.md             # Theory: stacks, queues, heaps, union find, Big-O
```

---

## Getting Started

### Prerequisites

- Python >= 3.10
- [Poetry](https://python-poetry.org/)
- [Docker](https://www.docker.com/) (for SQL exercises only)

### Installation

```sh
git clone https://github.com/amaldu/data-science-katas.git
cd data-science-katas
poetry install
poetry shell
```

### Jupyter Kernel Setup

```sh
poetry run python -m ipykernel install --user --name=ds-katas --display-name "DS Katas"
```

Then select the **DS Katas** kernel when opening notebooks in VS Code or Jupyter.

### Instructions on How to Run the SQL Exercises

1. Make sure Docker is running
2. Start the PostgreSQL container:

```sh
docker compose up -d
```

3. Connect using the PostgreSQL extension in VS Code (credentials are in `docker-compose.yaml`)
4. Run the migration script to create all tables:

```sh
poetry run python sql/migrate.py
```

5. Open any notebook in `sql/` and add `%%sql` at the top of SQL cells

---

## Contributing

Found an error or want to add exercises? Open an issue or submit a PR.

## License

MIT

