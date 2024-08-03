import React, { useState, useEffect } from 'react';
import socketIOClient from 'socket.io-client';

const ENDPOINT = 'http://localhost:5000/file_updates'; 

const App = () => {
  const [fileContent, setFileContent] = useState('');

  useEffect(() => {
    const socket = socketIOClient(ENDPOINT);

    socket.on('update_table', data => {
      setFileContent(data.file_content);
    });

    return () => socket.disconnect();
  }, []);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(event) {
      const fileContent = event.target.result;
      updateFileContent(fileContent);
    };

    reader.readAsText(file);
  };

  const updateFileContent = (fileContent) => {
    const socket = socketIOClient(ENDPOINT);
    socket.emit('update_file_content', { file_content: fileContent });
  };

  return (
    <div>
      <h2>Real-Time Excel File Updates</h2>
      <input type="file" onChange={handleFileChange} />
      <hr />
      <div>
        <h3>File Content</h3>
        <pre>{fileContent}</pre>
      </div>
    </div>
  );
};

export default App;