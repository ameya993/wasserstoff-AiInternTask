// CHANGE THIS TO YOUR BACKEND URL
const API_URL = "http://127.0.0.1:8002";

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
    
    // Handle the files array
    const files = data.files || [];
    
    if (files.length === 0) {
      fileList.innerHTML = '<li class="list-group-item text-muted"><i class="fa-solid fa-circle-info me-2"></i>No files uploaded.</li>';
      return;
    }
    
    // Process each file
    files.forEach(fileObj => {
      // Extract filename from object - try multiple possible properties
      let filename;
      
      if (typeof fileObj === 'string') {
        filename = fileObj;
      } else if (typeof fileObj === 'object') {
        // Try common filename properties
        filename = fileObj.filename || fileObj.name || fileObj.file_name || 
                  fileObj.path || fileObj.id || "Unknown file";
        
        // If we still don't have a filename, show the object keys
        if (filename === "Unknown file") {
          console.log("File object keys:", Object.keys(fileObj));
          // Just use the first property as a fallback
          const firstKey = Object.keys(fileObj)[0];
          if (firstKey) {
            filename = fileObj[firstKey];
          }
        }
      } else {
        filename = "Unknown file";
      }
      
      console.log("Extracted filename:", filename);
      
      const li = document.createElement('li');
      li.className = "list-group-item d-flex justify-content-between align-items-center";
      li.innerHTML = `<span><i class="fa-solid fa-file-lines me-2 text-primary"></i>${filename}</span>`;
      
      // Delete button
      const del = document.createElement('button');
      del.className = "btn btn-sm btn-danger file-actions";
      del.innerHTML = '<i class="fa-solid fa-trash"></i> Delete';
      del.onclick = async () => {
        // Store the original fileObj for deletion
        const fileToDelete = fileObj;
        
        Swal.fire({
          title: `Delete "${filename}"?`,
          text: "This action cannot be undone!",
          icon: "warning",
          showCancelButton: true,
          confirmButtonText: "Yes, delete it!",
        }).then(async (result) => {
          if (result.isConfirmed) {
            try {
              // If fileObj is an object, we might need to send it differently
              let deleteUrl;
              if (typeof fileToDelete === 'string') {
                deleteUrl = `${API_URL}/delete_file?filename=${encodeURIComponent(fileToDelete)}`;
              } else {
                // Try to use the same property we extracted the filename from
                deleteUrl = `${API_URL}/delete_file?filename=${encodeURIComponent(filename)}`;
              }
              
              console.log("Deleting file using URL:", deleteUrl);
              
              const deleteRes = await fetch(deleteUrl, { method: "DELETE" });
              
              if (deleteRes.ok) {
                loadFiles();
                loadDocCheckboxes();
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
