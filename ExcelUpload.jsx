import React, { useState, useEffect } from 'react';
import socketIOClient from 'socket.io-client';

const App = () => {
  const [file, setFile] = useState(null);
  const [fileUploaded, setFileUploaded] = useState(false);
  const [fileContent, setFileContent] = useState('');
  const [flashMessage, setFlashMessage] = useState('');

  useEffect(() => {
    const socket = socketIOClient('/file_updates');

    socket.on('update_table', data => {
      console.log('Table updated:', data);
      setFileContent(data.file_content);
    });

    return () => socket.disconnect();
  }, []);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async (event) => {
    event.preventDefault();

    if (!file) {
      setFlashMessage('Please select a file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setFileUploaded(true);
        setFileContent(data.file_content);
        setFlashMessage('File uploaded successfully.');
      } else {
        setFlashMessage('Failed to upload file.');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      setFlashMessage('Error uploading file.');
    }
  };

  const handleSaveChanges = async (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append('file_content', fileContent);

    try {
      const response = await fetch('/save_changes', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        setFlashMessage('Changes saved successfully.');
      } else {
        setFlashMessage('Failed to save changes.');
      }
    } catch (error) {
      console.error('Error saving changes:', error);
      setFlashMessage('Error saving changes.');
    }
  };

  return (
    <div>
      {flashMessage && (
        <div className="alert alert-info mb-4" role="alert">
          {flashMessage}
        </div>
      )}

      <h2 className="text-2xl font-bold mb-4">Upload an Excel File</h2>
      <form id="upload-form" onSubmit={handleUpload} encType="multipart/form-data" className="mb-4">
        <input type="file" name="file" id="file" onChange={handleFileChange} className="mb-2 border border-gray-300 rounded p-2" />
        <button type="submit" className="bg-blue-500 text-white rounded p-2">Upload File</button>
      </form>

      {fileUploaded && (
        <div>
          <div className="alert alert-info mb-4">
            File uploaded successfully
          </div>

          <h2 className="text-2xl font-bold mb-4">Processed File Content</h2>
          <div id="file-content" className="overflow-auto mb-4">
            <table>
              <tbody dangerouslySetInnerHTML={{ __html: fileContent }} />
            </table>
          </div>

          <form id="file-content-form" onSubmit={handleSaveChanges}>
            <input type="hidden" name="file_content" value={fileContent} />
            <button type="submit" className="bg-green-500 text-white rounded p-2">Save Changes</button>
          </form>
        </div>
      )}
    </div>
  );
};

export default App;