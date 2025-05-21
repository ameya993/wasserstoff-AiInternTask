// CHANGE THIS TO YOUR BACKEND URL
const API_URL = "http://127.0.0.1:8002"

function populateThemeCheckboxes(files) {
  const themeDiv = document.getElementById('themeDocCheckboxes');
  if (!themeDiv) return;
  themeDiv.innerHTML = files.map(name =>
    `<label class="me-2">
      <input type="checkbox" name="themeDocs" value="${name}"> ${name}
    </label>`
  ).join('');
}


// Drag & drop upload
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
let fileToUpload = null;

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  fileToUpload = e.dataTransfer.files[0];
  fileInput.files = e.dataTransfer;
});
fileInput.addEventListener('change', (e) => {
  fileToUpload = e.target.files[0];
});
uploadBtn.addEventListener('click', async () => {
  if (!fileToUpload) {
    Swal.fire("No file selected", "Please select or drag a file to upload.", "warning");
    return;
  }
  const formData = new FormData();
  formData.append('file', fileToUpload);
  document.getElementById('uploadMsg').innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Uploading...';
  try {
    const res = await fetch(`${API_URL}/upload`, { method: "POST", body: formData });
    if (res.ok) {
      Swal.fire("Success", "File uploaded successfully!", "success");
      document.getElementById('uploadMsg').textContent = "";
      fileInput.value = "";
      fileToUpload = null;
      loadFiles();
      loadDocCheckboxes();
    } else {
      const data = await res.json();
      Swal.fire("Upload failed", data.detail || res.statusText, "error");
      document.getElementById('uploadMsg').textContent = "";
    }
  } catch (err) {
    Swal.fire("Upload failed", err.message, "error");
    document.getElementById('uploadMsg').textContent = "";
  }
});

// List Files
// List Files
// List Files
async function loadFiles() {
  const fileList = document.getElementById('fileList');
  fileList.innerHTML = '<li class="list-group-item"><i class="fa-solid fa-spinner fa-spin"></i> Loading...</li>';

  try {
    const res = await fetch(`${API_URL}/files`);
    const data = await res.json();

    // Debug the response structure
    console.log("Files API response:", data);

    fileList.innerHTML = '';

    // Support both array of strings or array of objects
    let files = [];
    if (Array.isArray(data.files)) {
      files = data.files.map(fileObj => {
        if (typeof fileObj === 'string') {
          return fileObj;
        } else if (typeof fileObj === 'object') {
          // Try common filename properties
          return fileObj.filename || fileObj.name || fileObj.file_name || fileObj.path || fileObj.id || "Unknown file";
        } else {
          return "Unknown file";
        }
      });
    }

    // Populate checkboxes for themes and other forms
    populateThemeCheckboxes(files);
    if (typeof populateDocCheckboxes === 'function') populateDocCheckboxes(files);
    if (typeof populateCompareCheckboxes === 'function') populateCompareCheckboxes(files);

    if (files.length === 0) {
      fileList.innerHTML = '<li class="list-group-item text-muted"><i class="fa-solid fa-circle-info me-2"></i>No files uploaded.</li>';
      return;
    }

    // Process each file for display and deletion
    files.forEach(filename => {
      console.log("Extracted filename:", filename);

      const li = document.createElement('li');
      li.className = "list-group-item d-flex justify-content-between align-items-center";
      li.innerHTML = `<span><i class="fa-solid fa-file-lines me-2 text-primary"></i>${filename}</span>`;

      // Delete button
      const del = document.createElement('button');
      del.className = "btn btn-sm btn-danger file-actions";
      del.innerHTML = '<i class="fa-solid fa-trash"></i> Delete';
      del.onclick = async () => {
        Swal.fire({
          title: `Delete "${filename}"?`,
          text: "This action cannot be undone!",
          icon: "warning",
          showCancelButton: true,
          confirmButtonText: "Yes, delete it!",
        }).then(async (result) => {
          if (result.isConfirmed) {
            try {
              const deleteUrl = `${API_URL}/delete_file?filename=${encodeURIComponent(filename)}`;
              console.log("Deleting file using URL:", deleteUrl);

              const deleteRes = await fetch(deleteUrl, { method: "DELETE" });

              if (deleteRes.ok) {
                await loadFiles();
                if (typeof loadDocCheckboxes === 'function') loadDocCheckboxes();
                Swal.fire("Deleted!", `${filename} has been deleted.`, "success");
              } else {
                console.error("Delete response:", await deleteRes.text());
                Swal.fire("Error!", `Failed to delete ${filename}.`, "error");
              }
            } catch (err) {
              console.error("Delete error:", err);
              Swal.fire("Error!", `Failed to delete ${filename}: ${err.message}`, "error");
            }
          }
        });
      };

      li.appendChild(del);
      fileList.appendChild(li);
    });
  } catch (err) {
    console.error("Error loading files:", err);
    fileList.innerHTML = `<li class="list-group-item text-danger">Failed to load files: ${err.message}</li>`;
  }
}

// Apply the same fix to the document checkboxes
async function loadDocCheckboxes() {
  const container = document.getElementById('docCheckboxes');
  container.innerHTML = '<div><i class="fa-solid fa-spinner fa-spin"></i> Loading...</div>';
  
  try {
    const res = await fetch(`${API_URL}/files`);
    const data = await res.json();
    container.innerHTML = '';
    
    // Handle the files array
    const files = data.files || [];
    populateThemeCheckboxes(files);

    if (files.length === 0) {
      container.innerHTML = '<div class="text-muted">No files available</div>';
      return;
    }
    
    // Process each file for checkboxes
    files.forEach(fileObj => {
      // Extract filename using the same logic as above
      let filename;
      
      if (typeof fileObj === 'string') {
        filename = fileObj;
      } else if (typeof fileObj === 'object') {
        filename = fileObj.filename || fileObj.name || fileObj.file_name || 
                  fileObj.path || fileObj.id || "Unknown file";
        
        if (filename === "Unknown file" && Object.keys(fileObj).length > 0) {
          const firstKey = Object.keys(fileObj)[0];
          if (firstKey) {
            filename = fileObj[firstKey];
          }
        }
      } else {
        filename = "Unknown file";
      }
      
      container.innerHTML += `
        <label class="me-2">
          <input type="checkbox" name="selectedDocs" value="${filename}" checked> ${filename}
        </label>
      `;
    });
  } catch (err) {
    console.error("Error loading document checkboxes:", err);
    container.innerHTML = `<div class="text-danger">Failed to load documents: ${err.message}</div>`;
  }
}

loadFiles();

// Single-document Query
document.getElementById('queryForm').onsubmit = async (e) => {
  e.preventDefault();
  const question = document.getElementById('queryInput').value;
  const resultBox = document.getElementById('queryResult');
  resultBox.style.display = "block";
  resultBox.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Querying...';
  try {
    const res = await fetch(`${API_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: question })
    });
    const data = await res.json();
    resultBox.className = "alert alert-success";
    resultBox.innerHTML = `<b>Answer:</b> <br>${data.synthesized_answer || data.answer || JSON.stringify(data)}`;
  } catch (err) {
    resultBox.className = "alert alert-danger";
    resultBox.textContent = "Query failed: " + err;
  }
};

// Per-Document Query UI
async function loadDocCheckboxes() {
  const res = await fetch(`${API_URL}/files`);
  const data = await res.json();
  const container = document.getElementById('docCheckboxes');
  container.innerHTML = '';
  
  if (Array.isArray(data.files)) {
    data.files.forEach(file => {
      // Extract filename from file object if needed
      const filename = typeof file === 'object' ? file.filename || file.name : file;
      
      const id = "doc_" + btoa(filename);
      container.innerHTML += `
        <label class="me-2">
          <input type="checkbox" name="selectedDocs" value="${filename}" checked> ${filename}
        </label>
      `;
    });
  }
}

loadDocCheckboxes();

document.getElementById('docQueryForm').onsubmit = async (e) => {
  e.preventDefault();
  const query = document.getElementById('docQueryInput').value;
  const selected = Array.from(document.querySelectorAll('input[name="selectedDocs"]:checked')).map(cb => cb.value);
  const res = await fetch(`${API_URL}/per_document_query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, selected_files: selected })
  });
  const data = await res.json();
  const table = document.getElementById('docResultsTable');
  const tbody = table.querySelector('tbody');
  tbody.innerHTML = '';
  (data || []).forEach(row => {
    tbody.innerHTML += `
      <tr>
        <td>${row.document}</td>
        <td>${row.answer}</td>
        <td>${row.citation}</td>
      </tr>
    `;
  });
  table.style.display = (data.length > 0) ? "" : "none";
};

function renderComparisonOutput(comp) {
  let html = '';
  if (comp.similarities && comp.similarities.length) {
    html += '<h5>Similarities</h5><ul>';
    comp.similarities.forEach(item => html += `<li>${item}</li>`);
    html += '</ul>';
  }
  if (comp.differences && Object.keys(comp.differences).length) {
    html += '<h5>Differences</h5>';
    for (const [doc, diffList] of Object.entries(comp.differences)) {
      html += `<b>${doc}</b>:<ul>`;
      diffList.forEach(diff => html += `<li>${diff}</li>`);
      html += '</ul>';
    }
  }
  if (comp.summary) {
    html += `<h5>Summary</h5><p>${comp.summary}</p>`;
  }
  return html;
}

document.getElementById('compareDocsForm').onsubmit = async (e) => {
  e.preventDefault();
  const query = document.getElementById('compareDocsQueryInput').value;
  const selected = Array.from(document.querySelectorAll('input[name="selectedDocs"]:checked')).map(cb => cb.value);

  const resultDiv = document.getElementById('compareDocsResult');
  resultDiv.style.display = "block";
  resultDiv.className = "alert alert-info";
  resultDiv.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Comparing documents...';

  if (selected.length < 2) {
    resultDiv.className = "alert alert-warning";
    resultDiv.textContent = "Please select at least two documents to compare.";
    return;
  }

  try {
    const res = await fetch(`${API_URL}/compare_documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, selected_files: selected })
    });
    const data = await res.json();

    // >>>>>>>>>>>> PUT THE CODE HERE <<<<<<<<<<<<
    if (res.ok) {
      resultDiv.className = "alert alert-success";
      resultDiv.innerHTML = `<b>Compared Documents:</b> ${data.compared_documents.join(", ")}<br>${renderComparisonOutput(data.comparison)}`;
    } else {
      resultDiv.className = "alert alert-danger";
      resultDiv.textContent = data.detail || "Comparison failed.";
    }
    // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

  } catch (err) {
    resultDiv.className = "alert alert-danger";
    resultDiv.textContent = "Comparison failed: " + err;
  }
};


document.getElementById('themeForm').onsubmit = async (e) => {
  e.preventDefault();
  const selected = Array.from(document.querySelectorAll('input[name="themeDocs"]:checked')).map(cb => cb.value);

  if (selected.length === 0) {
    Swal.fire('Please select at least one document.');
    return;
  }

  const themeList = document.getElementById('themeList');
  themeList.innerHTML = '<li class="list-group-item">Extracting themes...</li>';

  try {
    const res = await fetch(`${API_URL}/themes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: "Extract main themes",
        selected_files: selected
      })
    });
    const data = await res.json();
    if (res.ok) {
      if (data.themes && data.themes.length) {
        themeList.innerHTML = data.themes.map(theme => `<li class="list-group-item">${theme}</li>`).join('');
      } else {
        themeList.innerHTML = `<li class="list-group-item text-warning">No themes found for the selected document(s).</li>`;
      }
    } else {
      themeList.innerHTML = `<li class="list-group-item text-danger">${data.detail || "Theme extraction failed."}</li>`;
    }
  } catch (err) {
    themeList.innerHTML = `<li class="list-group-item text-danger">Theme extraction failed: ${err}</li>`;
  }
};

// Themes (POST)
async function loadThemes() {
  const themeList = document.getElementById('themeList');
  themeList.innerHTML = '<li class="list-group-item"><i class="fa-solid fa-spinner fa-spin"></i> Loading...</li>';
  try {
    const res = await fetch(`${API_URL}/themes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: "themes" }) // or any string
    });
    const data = await res.json();
    themeList.innerHTML = '';
    (data.themes || []).forEach(theme => {
      const li = document.createElement('li');
      li.className = "list-group-item";
      li.innerHTML = `<i class="fa-solid fa-circle-dot me-2 text-info"></i>${theme}`;
      themeList.appendChild(li);
    });
    if ((data.themes || []).length === 0) {
      themeList.innerHTML = '<li class="list-group-item text-muted">No themes found.</li>';
    }
  } catch (err) {
    themeList.innerHTML = `<li class="list-group-item text-danger">Failed to load themes: ${err}</li>`;
  }
}
document.getElementById('loadThemesBtn').onclick = loadThemes;
loadThemes();