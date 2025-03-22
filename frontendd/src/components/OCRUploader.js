import React, { useState } from "react";
// import { useDropzone } from "react-dropzone";
import "../styles.css";

const OCRUploader = () => {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prompt, setPrompt] = useState("<OCR>");
  const [ocrResult, setOcrResult] = useState("");
  const [annotatedImage, setAnnotatedImage] = useState("")
  const [labels, setLabels] = useState("")
  const [loading, setLoading] = useState(false);
  const [customPrompt, setCustomPrompt] = useState("");
  const [model, setModel] = useState("florence");

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);

      const previewURL = URL.createObjectURL(file);
      setImagePreview(previewURL);
    }
  };
  
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!image) {
      alert("Please upload an image.");
      return;
    }
    setLoading(true);
  
    const formData = new FormData();
    formData.append("image", image);
    formData.append("prompt", prompt);
    formData.append("model", model);
  
    console.log("Sending request to backend...");
  
    try {
      const response = await fetch("http://localhost:8000/ocr", {
        method: "POST",
        body: formData,
      });
  
      console.log("Response received:", response);
  
      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }
  
      const data = await response.json();
      console.log("OCR Output:", data);
  
      // Extract OCR text
      if (typeof data.ocrText === "object") {
        setOcrResult(JSON.stringify(data.ocrText, null, 2)); // Convert object to readable string
      } else {
        setOcrResult(data.ocrText);
      }

      // Store annotated image URL
      if (data.annotatedImage) {
        setAnnotatedImage(`http://localhost:8000/${data.annotatedImage}`);
      } else {
        setAnnotatedImage("");
      }

      // Store extracted labels
      if (data.labels) {
        setLabels(data.labels)
      } else {
        setLabels("");
      }

    } catch (error) {
      console.error("Error fetching OCR results:", error);
      setOcrResult("Error processing the image.");
    }
    setLoading(false);
  };

  return (
    <div className="ocr-container">
        <h2 style={{marginTop: "20px", marginBottom: "40px"}}>Image Processing Component Testing</h2>
        <form onSubmit={handleSubmit} className="ocr-form">

            {/* Model Selection */}
            <label htmlFor="prompt" className="ocr-label">Select Model:</label>
            <select value={model} onChange={(e) => setModel(e.target.value)} className="ocr-select">
              <option value="florence">Florence-2</option>
              <option value="qwen">Qwen-2.5</option>
            </select>

            {/* Image Upload */}
            <label htmlFor="prompt" className="ocr-label">Select Image:</label>
            <input type="file" accept="image/*" onChange={handleImageChange} required className="ocr-input" />

            {/* OCR Mode Selection */}
            <div className="ocr-options">
                <label htmlFor="prompt" className="ocr-label">Select Prompt:</label>
                {model === "florence" && (
                  <select
                    id="prompt"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    className="ocr-select"
                  >
                    <option value="<OCR>">Basic OCR</option>
                    <option value="<OCR_WITH_REGION>">OCR with Bounding Boxes</option>
                    <option value="<CAPTION>">Basic Image Caption</option>
                    <option value="<DETAILED_CAPTION>">Detailed Image Caption</option>
                    <option value="<MORE_DETAILED_CAPTION>">Image Description</option>
                  </select>
                )}

                {/* If model is Qwen, only show custom prompt input */}
                {model === "qwen" && (
                  <>
                    <input
                      type="text"
                      value={customPrompt}
                      onChange={(e) => {
                        setCustomPrompt(e.target.value);
                        setPrompt(e.target.value);
                      }}
                      placeholder="Enter your custom prompt"
                      className="custom-prompt"
                    />
                  </>
                )}
                <button type="submit" disabled={loading} className="ocr-submit" onClick={handleSubmit}>
                    {loading ? "Processing..." : "Submit"}
                </button>
            </div>
        </form>

        {/* Image Preview */}
        {imagePreview && (
            <div className="ocr-result-container">
                <h3 className="ocr-result-title">Image Preview:</h3>
                <img src={imagePreview} alt="Uploaded Preview" className="ocr-preview"/>
            </div>
        )}
        
        {/* Annotated Image (Bounding Boxes) */}
        {annotatedImage && (
            <div className="ocr-annotated-image">
                <h3>Annotated Image:</h3>
                <img src={annotatedImage} alt="Annotated OCR Result" />
            </div>
        )}

        {/* Display either Labels or OCR Result */}
        {labels.length > 0 ? (
            <div className="ocr-labels-container">
                <h3 className="ocr-result-title">Extracted Labels:</h3>
                <ul className="ocr-labels-list">
                    {labels.map((label, index) => (
                        <li key={index}>{label}</li>
                    ))}
                </ul>
            </div>
        ) : (
            ocrResult && (
                <div className="ocr-result-container">
                    <h3 className="ocr-result-title">OCR Result:</h3>
                    <div className="ocr-result">{ocrResult}</div>
                </div>
            )
        )}
    </div>
);
}

export default OCRUploader;
