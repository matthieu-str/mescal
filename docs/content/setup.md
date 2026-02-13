# Setup

This guide walks you through setting up your Brightway project using Activity Browser, preparing it for use with MESCAL. We provide Jupyter notebooks for each step, available on the [MESCAL GitHub repository](https://github.com/matthieu-str/mescal/tree/master/dev).

> **Questions or issues?** Contact the MESCAL maintainers at matthieu.souttre@polymtl.ca

---
<!-- 

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **A valid ecoinvent license**  — [Get a license](https://ecoinvent.org/licenses/)
- **Jupyter Notebook or JupyterLab** for running the setup notebooks

---

## Setup Steps

| Step | Description | Required? |
|------|-------------|-----------|
| 1 | Install Activity Browser | ✅ Yes |
| 2 | Create a new project | ✅ Yes |
| 3 | Import ecoinvent database | ✅ Yes |
| 4 | Transform database with Premise | ⬜ Optional |
| 5 | Install ImpactWorld+ LCIA method | ⬜ Optional |
| 6 | Add supplementary databases | ⬜ Optional | 
-->

---

### Step 1: Install Activity Browser

Activity Browser is a graphical user interface for Brightway that makes project management much easier.

**Installation via conda:**
```bash
conda create -n ab -c conda-forge activity-browser
conda activate ab
activity-browser
```

📖 [Activity Browser Documentation](https://github.com/LCA-ActivityBrowser/activity-browser#installation)

---

### Step 2: Create Your Project

1. Open Activity Browser
2. Go to **Project → New project**
3. Give your project a meaningful name (e.g., `mescal_project`)

Your project will store all databases, LCIA methods, and calculation setups in an isolated environment.

---

### Step 3: Import ecoinvent Database

This is the foundation for all LCA calculations in MESCAL.

**In Activity Browser:**
1. Navigate to **Database → Import database**
2. Select **ecoinvent** as the source
3. Choose your ecoinvent version (3.9.1 or later recommended)
4. Select a system model:
   - **Cutoff** — recommended for most attributional LCA studies
   - **Consequential** — for consequential LCA studies
   - **APOS** — allocation at point of substitution
5. Enter your ecoinvent credentials when prompted

📖 [Brightway ecoinvent import guide](https://docs.brightway.dev/en/latest/content/examples/brightway-examples/import_data/import_ecoinvent/import_ecoinvent.html)

---

### Step 4: Transform Database with Premise (Optional)

[Premise](https://premise.readthedocs.io/en/latest/) allows you to create prospective LCA databases by projecting ecoinvent into future scenarios using Integrated Assessment Models (IAMs).

**Why use Premise?**
- Model future energy transitions
- Include emerging technologies
- Align inventories with climate scenarios (e.g., SSP, RCP pathways)

**📓 Notebook:** [import_premise_db.ipynb](https://github.com/matthieu-str/mescal/blob/master/dev/import_premise_db.ipynb)

> ⚠️ **Note:** You may need an encryption key from the Premise developers to access standard IAM scenarios. Contact romain.sacchi@psi.ch

---

### Step 5: Install ImpactWorld+ LCIA Method (Optional)

[ImpactWorld+](http://www.impactworldplus.org/) is a globally regionalized life cycle impact assessment method developed by CIRAIG, providing both midpoint and endpoint indicators.

**📓 Notebook:** [download_impact_world_plus.ipynb](https://github.com/matthieu-str/mescal/blob/master/dev/download_impact_world_plus.ipynb)

---

### Step 6: Add Supplementary Databases (Optional)

Depending on your energy system model, you may need additional life cycle inventories not covered in ecoinvent.

#### Carculator — Vehicle LCA

[Carculator](https://carculator.psi.ch/) provides detailed LCI for passenger vehicles, trucks, and buses with prospective scenarios.

**📓 Notebook:** [carculator.ipynb](https://github.com/matthieu-str/mescal/blob/master/dev/carculator.ipynb)

#### Other Databases

| Database | Description | Notebook |
|----------|-------------|----------|
| Premise additional inventories | Emerging technologies (hydrogen, DAC, etc.) | Included with Premise |
| *[Add your database]* | *[Description]* | *[Link to notebook]* |

---

## Verification

Once setup is complete, verify your installation by running:
```python
import bw2data as bd

bd.projects.set_current('your_project_name')
print("Available databases:", list(bd.databases))
print("Available methods:", len(bd.methods), "LCIA methods")
```

You should see your imported ecoinvent database and any additional databases you've added.