# User inputs

You can download the different files by clicking the download button below, or suggest edits by filling in a form.

For any question or remark about this data, please contact the maintainers of _mescal_ (for example, matthieu.souttre@polymtl.ca). 

## Mapping between ESM technologies and LCI datasets

The mapping file links the ESM technologies, resources and vectors with LCI datasets. Here, we present a generic mapping 
between typical ESM technologies, resources and vectors, with LCI datasets from [ecoinvent 3.10.1](https://support.ecoinvent.org/ecoinvent-version-3.10.1) (cut-off), 
complemented with additional datasets from [premise](https://github.com/polca/premise) (see [this notebook](https://github.com/matthieu-str/mescal/blob/master/dev/import_premise_db.ipynb)) and [carculator_truck](https://github.com/Laboratory-for-Energy-Systems-Analysis/carculator_truck) 
(see [this notebook](https://github.com/matthieu-str/mescal/blob/master/dev/carculator.ipynb)).

The table below has the following columns:
- **Name**: Name of the ESM technology, resource or vector.
- **Type**: Type of the LCI dataset (operation, construction, resource or flow).
- **Product**: Name of the corresponding LCI dataset product.
- **Activity**: Name of the corresponding LCI dataset activity.

<div id="table-container"></div>
<button id="download-btn">⬇️ Download CSV</button>
<button id="form-btn">📝 Suggest an edit</button>

<script>
  // Disable AMD temporarily
  var define_backup = window.define;
  window.define = undefined;
</script>

<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js"></script>

<script>
  window.define = define_backup;

  document.addEventListener('DOMContentLoaded', function() {
    const csvPath = '../_static/mapping_generic.csv';
    const csvPathES = '../_static/mapping_energyscope.csv';
    const formLink = 'https://forms.gle/dkqB3qa92oETEow97';
    
    fetch(csvPath)
      .then(response => response.text())
      .then(data => {
        const parsed = Papa.parse(data.trim(), { skipEmptyLines: true });
        const rows = parsed.data;
        const headers = rows[0];
        const body = rows.slice(1).filter(r => r.length === headers.length);

        let html = '<table id="data-table-1" class="display"><thead><tr>';
        headers.forEach(h => html += `<th>${h}</th>`);
        html += '</tr></thead><tbody>';
        body.forEach(r => html += '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>');
        html += '</tbody></table>';

        document.getElementById('table-container').innerHTML = html;
        $('#data-table-1').DataTable();
        document.getElementById('download-btn').onclick = () => window.open(csvPathES);
        document.getElementById('form-btn').onclick = () => window.open(formLink, '_blank');
      });
  });
</script>

## Unit conversion factors

The previous mapping file goes with a set of unit conversion factors to convert ESM technology, resource and vector 
physical units to LCI dataset units.

The table below has the following columns:
- **Name**: Name of the ESM technology, resource or vector.
- **Type**: Type of the LCI dataset (operation, construction, resource or flow).
- **Value**: Conversion factor value.
- **LCA unit**: Unit used in LCI datasets.
- **ESM unit**: Unit used in ESM.
- **Assumptions & Sources**: Any assumptions made or sources used to determine the conversion factor.

<div id="conv-table-container"></div>
<button id="conv-download-btn">⬇️ Download Excel</button>
<button id="conv-form-btn">📝 Suggest an edit</button>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const csvPathConv = '../_static/unit_conversion_generic.csv';
  const excelPathConv = '../_static/unit_conversion_energyscope.xlsx';
  const formLinkConv = 'https://forms.gle/G6Uo2YJVZiRrnK4f9';

  fetch(csvPathConv)
    .then(response => response.text())
    .then(data => {
      const parsed = Papa.parse(data.trim(), { skipEmptyLines: true });
      const rows = parsed.data;
      const headers = rows[0];
      const body = rows.slice(1).filter(r => r.length === headers.length);

      let html = '<table id="data-table-2" class="display"><thead><tr>';
      headers.forEach(h => html += `<th>${h}</th>`);
      html += '</tr></thead><tbody>';
      body.forEach(r => html += '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>');
      html += '</tbody></table>';

      document.getElementById('conv-table-container').innerHTML = html;
      $('#data-table-2').DataTable();
      document.getElementById('conv-download-btn').onclick = () => window.open(excelPathConv);
      document.getElementById('conv-form-btn').onclick = () => window.open(formLinkConv, '_blank');
    });
});
</script>

## Energy System Model

The technologies and resources presented in the mapping files above correspond to those used in the Energy System 
Model (ESM).

The table below has the following columns:
- **Name**: Name of the ESM technology.
- **Flow**: Name of the ESM resource or vector.
- **Amount**: Amount of resource or vector used/produced by the technology (positive for production, negative for consumption).

<div id="model-table-container"></div>
<button id="model-download-btn">⬇️ Download CSV</button>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const csvPathModel = '../_static/model_generic.csv';
  const csvPathModelES = '../_static/model_energyscope.csv';

  fetch(csvPathModel)
    .then(response => response.text())
    .then(data => {
      const parsed = Papa.parse(data.trim(), { skipEmptyLines: true });
      const rows = parsed.data;
      const headers = rows[0];
      const body = rows.slice(1).filter(r => r.length === headers.length);

      let html = '<table id="data-table-4" class="display"><thead><tr>';
      headers.forEach(h => html += `<th>${h}</th>`);
      html += '</tr></thead><tbody>';
      body.forEach(r => html += '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>');
      html += '</tbody></table>';

      document.getElementById('model-table-container').innerHTML = html;
      $('#data-table-4').DataTable();
      document.getElementById('model-download-btn').onclick = () => window.open(csvPathModelES);
    });
});
</script>

## Mapping between ESM vectors and CPC categories

When removing double-counted flows in LCI datasets, we identify flows based on their correspondence to CPC categories.
Therefore, each ESM vector is mapped to one or more CPC categories.

The table below has the following columns:
- **Vector**: Description of the ESM vector.
- **CPC**: List of corresponding CPC categories.

<div id="cpc-table-container"></div>
<button id="cpc-download-btn">⬇️ Download CSV</button>
<button id="cpc-form-btn">📝 Suggest an edit</button>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const csvPathCPC = '../_static/mapping_vectors_to_cpc.csv';
  const formLinkCPC = 'https://forms.gle/JoStgTnf41VUccTr8';

  fetch(csvPathCPC)
    .then(response => response.text())
    .then(data => {
      const parsed = Papa.parse(data.trim(), { skipEmptyLines: true });
      const rows = parsed.data;
      const headers = rows[0];
      const body = rows.slice(1).filter(r => r.length === headers.length);

      let html = '<table id="data-table-3" class="display"><thead><tr>';
      headers.forEach(h => html += `<th>${h}</th>`);
      html += '</tr></thead><tbody>';
      body.forEach(r => html += '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>');
      html += '</tbody></table>';

      document.getElementById('cpc-table-container').innerHTML = html;
      $('#data-table-3').DataTable();
      document.getElementById('cpc-download-btn').onclick = () => window.open(csvPathCPC);
      document.getElementById('cpc-form-btn').onclick = () => window.open(formLinkCPC, '_blank');
    });
});
</script>

## References

Althaus, Hans-Joerg, Mike Chudacoff, Roland Hischier, Niels Jungbluth, Maggie Osses, and Alex Primas. 2007. Life Cycle Inventories of Chemicals. Ecoinvent Report No. 8, v2.0. Dübendorf, CH: Swiss Centre for Life Cycle Inventories.

Biedermann, Ferenc. 2023. Comportement de La Population En Matière de Mobilité. Résultats Du Microrecensement Mobilité et Transports 2021. Office fédérale de la statistique.

Boehm, Randall C., Zhibin Yang, David C. Bell, John Feldhausen, and Joshua S. Heyne. 2022. “Lower Heating Value of Jet Fuel from Hydrocarbon Class Concentration Data and Thermo-Chemical Reference Data: An Uncertainty Quantification.” Fuel 311:122542. doi:10.1016/j.fuel.2021.122542.

Chaire Mobilité. 2018. Rapport d’activités 2014-2015, Version Préliminaire. Polytechnique Montréal.

CIRAIG. 2022. Analyse Du Cycle De Vie De Filières Énergétiques Et De Leur Utilisation Pour Le Transport Routier Au Québec – Partie 2: Utilisation.

Engineering ToolBox. 2003. “Fuels - Higher and Lower Calorific Values.” https://www.engineeringtoolbox.com/fuels-higher-calorific-values-d_169.html.

Engineering ToolBox. 2017. “Combustion Heat.” https://www.engineeringtoolbox.com/standard-heat-of-combustion-energy-content-d_1987.html.

Engineering ToolBox. 2018. “Carbon Dioxide - Density and Specific Weight vs. Temperature and Pressure.” https://www.engineeringtoolbox.com/carbon-dioxide-density-specific-weight-temperature-pressure-d_2018.html.

Fischer, Loïc. 2023. “Own Computation from Various Sources” [Excel File]. (Available upon request).

Gerloff, Niklas. 2021. “Comparative Life-Cycle-Assessment Analysis of Three Major Water Electrolysis Technologies While Applying Various Energy Scenarios for a Greener Hydrogen Production.” Journal of Energy Storage 43:102759. doi:10.1016/j.est.2021.102759.

Grasby, S. E., D. M. Allen, S. Bell, Z. Chen, G. Ferguson, A. Jessop, M. Kelman, M. Ko, J. Majorowicz, M. Moore, J. Raymond, and R. Therrien. 2012. Geothermal Energy Resource Potential of Canada. rev. 6914. doi:10.4095/291488.

IEA-SHC. 2004. “Solar Thermal Statistics Conversion Method - M2 to GWth.”

Kamidelivand, Mitra, Peter Deeney, Fiona Devoy McAuliffe, Kevin Leyne, Michael Togneri, and Jimmy Murphy. 2023. “Scenario Analysis of Cost-Effectiveness of Maintenance Strategies for Fixed Tidal Stream Turbines in the Atlantic Ocean.” Journal of Marine Science and Engineering 11(5):1046. doi:10.3390/jmse11051046.

Limpens, Gauthier. 2021. “Generating Energy Transition Pathways : Application to Belgium.” PhD thesis, Université catholique de Louvain.

Marin-Batista, J. D., J. A. Villamil, S. V. Qaramaleki, C. J. Coronella, A. F. Mohedano, and M. A. De La Rubia. 2020. “Energy Valorization of Cow Manure by Hydrothermal Carbonization and Anaerobic Digestion.” Renewable Energy 160:623–32. doi:10.1016/j.renene.2020.07.003.

Moret, Stefano. 2017. “Strategic Energy Planning under Uncertainty.” PhD thesis, EPFL.

Notter, Dominic A., Marcel Gauch, Rolf Widmer, Patrick Wäger, Anna Stamp, Rainer Zah, and Hans-Jörg Althaus. 2010. “Contribution of Li-Ion Batteries to the Environmental Impact of Electric Vehicles.” Environmental Science & Technology 44(17):6550–56. doi:10.1021/es903729a.

Pöschl, Martina, Shane Ward, and Philip Owende. 2010. “Evaluation of Energy Efficiency of Various Biogas Production and Utilization Pathways.” Applied Energy 87(11):3305–21. doi:10/fwz87p.

Prina, Matteo Giacomo, Giampaolo Manzolini, David Moser, Benedetto Nastasi, and Wolfram Sparber. 2020. “Classification and Challenges of Bottom-up Energy System Models - A Review.” Renewable and Sustainable Energy Reviews 129:109917. doi:10.1016/j.rser.2020.109917.

Sacchi, R., T. Terlouw, K. Siala, A. Dirnaichner, C. Bauer, B. Cox, C. Mutel, V. Daioglou, and G. Luderer. 2022. “PRospective EnvironMental Impact asSEment (Premise): A Streamlined Approach to Producing Databases for Prospective Life Cycle Assessment Using Integrated Assessment Models.” Renewable and Sustainable Energy Reviews 160:112311. doi:10.1016/j.rser.2022.112311.

Sacchi, Romain, Christian Bauer, and Brian L. Cox. 2021. “Does Size Matter? The Influence of Size, Load Factor, Range Autonomy, and Application Type on the Life Cycle Assessment of Current and Future Medium- and Heavy-Duty Vehicles.” Environmental Science & Technology 55(8):5224–35. doi:10.1021/acs.est.0c07773.

Schnidrig, Jonas, Rachid Cherkaoui, Yasmine Calisesi, Manuele Margni, and François Maréchal. 2023. “On the Role of Energy Infrastructure in the Energy Transition. Case Study of an Energy Independent and CO2 Neutral Energy System for Switzerland.” Frontiers in Energy Research 11:1164813. doi:10.3389/fenrg.2023.1164813.

Schnidrig, Jonas, Tuong-Van Nguyen, Paul Stadler, and François Maréchal. 2020. “Assessment of Green Mobility Scenarios on European Energy Systems.” Master thesis, EPFL.

Spielmann, M., R. Dones, and C. Bauer. 2007. Life Cycle Inventories of Transport Services. Final Report Ecoinvent v2.0 No. 14. Dübendorf, CH: Swiss Centre for Life Cycle Inventories.

The World Bank. 2009. Air Freight: A Market Study with Implications for Landlocked Countries. Washington, D.C.: The International Bank for Reconstruction and Development / The World Bank. https://documents1.worldbank.org/curated/en/265051468324548129/pdf/517470NWP0tp1210Box342045B01PUBLIC1.pdf.

Thomson, R. Camilla, John P. Chick, and Gareth P. Harrison. 2019. “An LCA of the Pelamis Wave Energy Converter.” The International Journal of Life Cycle Assessment 24(1):51–63. doi:10.1007/s11367-018-1504-2.

Van Der Giesen, Coen, René Kleijn, and Gert Jan Kramer. 2014. “Energy and Climate Impacts of Producing Synthetic Hydrocarbon Fuels from CO 2.” Environmental Science & Technology 48(12):7111–21. doi:10.1021/es500191g.

Wernet, Gregor, Christian Bauer, Bernhard Steubing, Jürgen Reinhard, Emilia Moreno-Ruiz, and Bo Weidema. 2016. “The Ecoinvent Database Version 3 (Part I): Overview and Methodology.” The International Journal of Life Cycle Assessment 21(9):1218–30. doi:10.1007/s11367-016-1087-8.

Wikipedia. 2023. “Heat of Combustion.” https://en.wikipedia.org/wiki/Heat_of_combustion.

Willauer, Heather, Dennis Hardy, Kenneth Schultz, and Frederick Williams. 2012. “The Feasibility and Current Estimated Capital Costs of Producing Jet Fuel at Sea Using Carbon Dioxide and Hydrogen.” Journal of Renewable Sustainable Energy. doi:10.1063/1.4719723.

World Nuclear Association. n.d. “Heat Values of Various Fuels.” Retrieved April 16, 2024. https://world-nuclear.org/information-library/facts-and-figures/heat-values-of-various-fuels.aspx.
