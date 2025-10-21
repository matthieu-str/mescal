# User inputs

## Mapping between ESM technologies and LCI datasets

The mapping file links the ESM technologies, resources and vectors with LCI datasets. Here, we present a generic mapping 
between typical ESM technologies, resources and vectors, with LCI datasets from 
[ecoinvent 3.10.1](https://support.ecoinvent.org/ecoinvent-version-3.10.1) (cut-off), complemented with additional 
datasets from [premise](https://github.com/polca/premise) 
(see [this notebook](https://github.com/matthieu-str/mescal/blob/master/dev/import_premise_db.ipynb)) and 
[carculator_truck](https://github.com/Laboratory-for-Energy-Systems-Analysis/carculator_truck) 
(see [this notebook](https://github.com/matthieu-str/mescal/blob/master/dev/carculator.ipynb)).

You can download the full mapping file as a CSV by clicking the button below, or suggest edits to the mapping
by filling in a form.

The table below has the following columns:
- **Name**: Name of the ESM technology, resource or vector.
- **Type**: Type of the ESM item (technology, resource or vector).
- **Product**: Name of the corresponding LCI dataset product.
- **Activity**: Name of the corresponding LCI dataset activity.

<div id="table-container"></div>

<button id="download-btn">⬇️ Download CSV</button>
<button id="form-btn">📝 Suggest an edit</button>

<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">

<script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js"></script>
<script>
const csvPath = '/_static/mapping_generic.csv';
const formLink = 'https://forms.gle/3Yu1qjrpMp7gMfgH8';

fetch(csvPath)
  .then(response => response.text())
  .then(data => {
    const parsed = Papa.parse(data.trim(), { skipEmptyLines: true });
    const rows = parsed.data;
    const headers = rows[0];
    const body = rows.slice(1).filter(r => r.length === headers.length);

    let html = '<table id="data-table" class="display"><thead><tr>';
    headers.forEach(h => html += `<th>${h}</th>`);
    html += '</tr></thead><tbody>';
    body.forEach(r => {
      html += '<tr>' + r.map(c => `<td>${c}</td>`).join('') + '</tr>';
    });
    html += '</tbody></table>';
    document.getElementById('table-container').innerHTML = html;

    $('#data-table').DataTable();
    document.getElementById('download-btn').onclick = () => window.open(csvPath);
    document.getElementById('form-btn').onclick = () => window.open(formLink, '_blank');
  });
</script>