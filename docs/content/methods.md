# Methods

## Mapping

Modellers should provide a mapping between the technologies and resources in the ESM and the datasets in the LCI database. Typically, a technology is mapped with an infrastructure LCI dataset (construction and end-of-life) and an operation LCI dataset, while resources are mapped with a single (operation) LCI dataset. However, this structure is not mandatory and can be adapted to the specific needs of the modeller.

![mapping between esm technologies and lci datasets](../pics/mapping.png "mapping")

## Regionalization

LCI datasets can be regionalized to match the location of the ESM. For instance, if a LCI dataset does not exist for the region of interest, a new LCI dataset can be copied and adapted from an existing one (by changing the list of providers) as a proxy. 

## Harmonization

Harmonization is the process of ensuring that the LCI datasets are consistent with the ESM. It mainly consists of unit conversion and assumptions alignement. 

## Double-counting removal

Double-counting occurs regarding the flows that are explicitly modelled in the ESM, e.g., energy flows (electricity, heat, fuels, etc.). Such flows should be set to zero in the LCI datasets in order to avoid an overestimation of the impacts. The double-counting removal algorithm is as follows:

![flowchart of the double-counting removal algorithm](../pics/double_counting_flowchart.png "double_counting_removal")

## Life-cycle impact assessment

The set of LCA indicators to be included in the ESM is computed using an LCIA method, e.g., [IMPACT World+](https://www.impactworldplus.org/), [ReCiPe](https://www.rivm.nl/en/life-cycle-assessment-lca/recipe), etc. 