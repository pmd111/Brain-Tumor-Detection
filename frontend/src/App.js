
import React, { useState } from "react";
import "./styles.css";

export default function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = (event) => {
    setImage(event.target.files[0]);
  };

  const handlePredict = async () => {
    const formData = new FormData();
    formData.append("file", image);

    const response = await fetch("http://localhost:5000/predict", {
      method: "POST",
      body: formData,
    });

    const blob = await response.blob();
    setResult(URL.createObjectURL(blob));
  };

  return (
    <div className="container">
      <h1>Brain Tumor Detection</h1>
      <input type="file" onChange={handleUpload} />
      <button onClick={handlePredict}>Predict</button>
      {result && <img src={result} alt="Segmentation Result" className="result-img" />}
    </div>
  );
}
