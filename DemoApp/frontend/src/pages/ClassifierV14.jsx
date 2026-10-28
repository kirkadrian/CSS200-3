import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import '../App.css'

const SEEDS = [1, 2, 3];
const MODEL_TYPES = [
  { key_suffix: "Baseline DenseNet", title: "Baseline DenseNet" },
  { key_suffix: "Frozen Control", title: "Frozen Control" },
  { key_suffix: "Our Model", title: "Unfrozen BN-Frozen (Our Model)" },
  { key_suffix: "Baseline MobileNet", title: "Baseline MobileNet" }
];

// Flat array used for the batch processing table
const DISPLAY_MODELS = SEEDS.flatMap(s => 
  MODEL_TYPES.map(mt => ({
    key: `${mt.key_suffix} S${s}`,
    title: `${mt.title} (Seed ${s})`,
    isMobileNet: mt.key_suffix === "Baseline MobileNet"
  }))
);

const TARGET_CLASSES = ["Motherboard", "Battery", "RAM", "WiFi Card", "CPU Fan"];

function ModelCard({ modelKey, modelTitle, data, isFullWidth, heatmap, loadingHeatmap, onRequestHeatmap }) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div style={{ 
      border: '1px solid #444', padding: '20px', borderRadius: '12px',
      backgroundColor: '#1a1a1a', boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
      opacity: data ? 1 : 0.6, gridColumn: isFullWidth ? '1 / -1' : 'auto',
      display: 'flex', flexDirection: 'column'
    }}>
      <h4 style={{ margin: '0 0 15px 0', color: '#fff', fontSize: '16px' }}>{modelTitle}</h4>
      <div style={{ fontSize: '20px', marginBottom: '10px' }}>
        <span style={{ color: '#888', fontSize: '14px', display: 'block' }}>Prediction</span>
        <strong style={{ color: data ? '#4caf50' : '#555' }}>{data ? data.prediction : '--'}</strong>
      </div>
      <div style={{ marginBottom: '15px' }}>
        <span style={{ color: '#888', fontSize: '14px', display: 'block' }}>Confidence</span>
        <strong style={{ color: data ? '#ddd' : '#555' }}>{data ? `${(data.confidence * 100).toFixed(1)}%` : '--%'}</strong>
      </div>
      
      {data && (
        <div style={{ marginBottom: '15px' }}>
          {!heatmap ? (
            <button 
              onClick={() => onRequestHeatmap(modelKey)}
              disabled={loadingHeatmap}
              style={{
                padding: '8px 12px', backgroundColor: '#333', color: 'white', border: '1px solid #555',
                borderRadius: '6px', cursor: loadingHeatmap ? 'not-allowed' : 'pointer', fontSize: '13px', width: '100%'
              }}
            >
              {loadingHeatmap ? 'Generating Heatmap...' : 'Show Heatmap'}
            </button>
          ) : (
            <div style={{ marginTop: '10px' }}>
              <img 
                src={`data:image/jpeg;base64,${heatmap}`} 
                alt={`${modelTitle} Heatmap`}
                style={{ width: '100%', borderRadius: '6px', border: '1px solid #555', objectFit: 'cover' }}
              />
            </div>
          )}
        </div>
      )}

      <div style={{ backgroundColor: '#333', borderRadius: '6px', pointerEvents: data ? 'auto' : 'none', opacity: data ? 1 : 0.5, marginTop: 'auto' }}>
        <div onClick={() => data && setIsOpen(!isOpen)} style={{ cursor: 'pointer', color: '#aaa', fontWeight: 'bold', padding: '10px 12px', fontSize: '13px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', userSelect: 'none' }}>
          <span>View All Classes</span>
          <span style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s ease', fontSize: '10px' }}>▼</span>
        </div>
        <div style={{ maxHeight: isOpen ? '200px' : '0px', overflow: 'hidden', transition: 'max-height 0.3s ease-in-out, opacity 0.3s ease-in-out', opacity: isOpen ? 1 : 0 }}>
          {data && (
            <ul style={{ padding: '0 12px 12px 30px', margin: '0', listStyleType: 'circle', color: '#ccc', fontSize: '13px' }}>
              {Object.entries(data.all_scores).map(([className, score]) => (
                <li key={className} style={{ marginBottom: '6px' }}>{className}: {(score * 100).toFixed(2)}%</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div style={{ backgroundColor: '#222', padding: '10px', border: '1px solid #555', borderRadius: '5px', color: '#fff' }}>
        <p style={{ margin: '0 0 5px 0', fontWeight: 'bold' }}>{label}</p>
        {payload.map((entry, index) => (
          <p key={index} style={{ color: entry.color, margin: '2px 0', fontSize: '13px' }}>
            {entry.name}: {entry.payload[`${entry.dataKey}Pred`]} ({entry.value.toFixed(1)}%)
          </p>
        ))}
      </div>
    );
  }
  return null;
};

function ClassifierV14() {
  const navigate = useNavigate();
  const [appMode, setAppMode] = useState('single') 
  
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  
  const [batchFiles, setBatchFiles] = useState([])
  const [batchFileUrls, setBatchFileUrls] = useState({})
  const [batchResults, setBatchResults] = useState({})
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchProgress, setBatchProgress] = useState(0)
  
  const [heatmaps, setHeatmaps] = useState({})
  const [loadingHeatmaps, setLoadingHeatmaps] = useState({})

  const singleFileInputRef = useRef(null)
  const batchFileInputRef = useRef(null)

  const handleSingleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setResults(null)
      setError(null)
      setHeatmaps({})
    }
  }

  const triggerSingleFileInput = () => singleFileInputRef.current.click()

  const handleSingleUpload = async () => {
    if (!selectedFile) return
    setLoading(true); setError(null); setHeatmaps({})
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post('http://localhost:8000/predict/v14', formData, { headers: { 'Content-Type': 'multipart/form-data' } })
      if (response.data.error) throw new Error(response.data.error)
      setResults(response.data)
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'An error occurred connecting to the backend.')
    } finally {
      setLoading(false)
    }
  }

  const handleRequestHeatmap = async (modelKey) => {
    if (!selectedFile) return;
    
    setLoadingHeatmaps(prev => ({ ...prev, [modelKey]: true }));
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model_key', modelKey);

    try {
      const response = await axios.post('http://localhost:8000/heatmap/v14', formData, { 
        headers: { 'Content-Type': 'multipart/form-data' } 
      });
      if (response.data.error) throw new Error(response.data.error);
      
      setHeatmaps(prev => ({ ...prev, [modelKey]: response.data.heatmap_base64 }));
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'An error occurred fetching the heatmap.');
    } finally {
      setLoadingHeatmaps(prev => ({ ...prev, [modelKey]: false }));
    }
  };

  const handleBatchFileChange = (e) => {
    const files = Array.from(e.target.files)
    if (files.length > 0) {
      setBatchFiles(files)
      
      const urls = {}
      files.forEach(f => urls[f.name] = URL.createObjectURL(f))
      setBatchFileUrls(urls)
      
      setBatchResults({})
      setError(null)
      setBatchProgress(0)
    }
  }

  const triggerBatchFileInput = () => batchFileInputRef.current.click()

  const handleBatchUpload = async () => {
    if (batchFiles.length === 0) return
    setBatchLoading(true); setError(null); setBatchProgress(0);
    const newResults = {}

    for (let i = 0; i < batchFiles.length; i++) {
      const file = batchFiles[i]
      const formData = new FormData()
      formData.append('file', file)

      try {
        const response = await axios.post('http://localhost:8000/predict/v14', formData, { headers: { 'Content-Type': 'multipart/form-data' } })
        if (response.data.error) throw new Error(response.data.error)
        newResults[file.name] = response.data
      } catch (err) {
        newResults[file.name] = { error: err.message }
      }
      setBatchResults({...newResults})
      setBatchProgress(i + 1)
    }
    setBatchLoading(false)
  }

  const chartData = results ? MODEL_TYPES.map(mt => {
    const s1 = results[`${mt.key_suffix} S1`];
    const s2 = results[`${mt.key_suffix} S2`];
    const s3 = results[`${mt.key_suffix} S3`];
    
    return {
      name: mt.key_suffix,
      Seed1: s1 ? s1.confidence * 100 : 0,
      Seed2: s2 ? s2.confidence * 100 : 0,
      Seed3: s3 ? s3.confidence * 100 : 0,
      Seed1Pred: s1 ? s1.prediction : "",
      Seed2Pred: s2 ? s2.prediction : "",
      Seed3Pred: s3 ? s3.prediction : "",
    }
  }) : [];

  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif', width: '100%', maxWidth: '1200px', margin: '0 auto', boxSizing: 'border-box' }}>
      
      {/* Header and Mode Toggle */}
      <div style={{ borderBottom: '2px solid #333', paddingBottom: '15px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px' }}>
        <h1 style={{ margin: '0', fontSize: '28px' }}>E-Waste Laptop Components (v14)</h1>
        
        <div style={{ display: 'flex', gap: '10px' }}>
          <button 
            onClick={() => navigate('/')}
            style={{
              padding: '10px 16px', backgroundColor: 'transparent', color: '#ccc', border: '1px solid #444',
              borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold'
            }}
          >
            &larr; Back to Menu
          </button>
          
          <button 
            onClick={() => setAppMode(m => m === 'single' ? 'batch' : 'single')}
            style={{
              padding: '10px 16px', backgroundColor: '#333', color: 'white', border: '1px solid #555',
              borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold'
            }}
          >
            Switch to {appMode === 'single' ? 'Batch Processing Mode' : 'Single Image Mode'}
          </button>
        </div>
      </div>
      
      {error && <div style={{ color: 'white', backgroundColor: '#ff4444', padding: '10px', borderRadius: '4px', margin: '20px 0' }}>Error: {error}</div>}

      {/* SINGLE MODE */}
      {appMode === 'single' && (
        <div style={{ display: 'flex', gap: '40px', flexWrap: 'wrap', marginTop: '30px', alignItems: 'flex-start' }}>
          
          {/* Left Column - Fixed Width to prevent shifting */}
          <div style={{ flex: '0 0 350px', backgroundColor: '#242424', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)', boxSizing: 'border-box' }}>
            <h3 style={{ marginTop: 0, color: '#fff' }}>Input Image</h3>
            
            <div 
              onClick={!previewUrl ? triggerSingleFileInput : undefined}
              style={{ 
                width: '100%', aspectRatio: '1 / 1', backgroundColor: '#1a1a1a', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: !previewUrl ? 'pointer' : 'default', overflow: 'hidden', marginBottom: '20px', border: !previewUrl ? '2px dashed #555' : 'none', transition: 'border-color 0.2s'
              }}
              onMouseOver={(e) => { if(!previewUrl) e.currentTarget.style.borderColor = '#888' }}
              onMouseOut={(e) => { if(!previewUrl) e.currentTarget.style.borderColor = '#555' }}
            >
              {previewUrl ? (
                <img src={previewUrl} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              ) : (
                <div style={{ textAlign: 'center', color: '#888' }}>
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle>
                  </svg>
                  <p style={{ margin: '10px 0 0 0', fontWeight: '500' }}>Click to select an image</p>
                </div>
              )}
            </div>

            <input type="file" accept="image/*" onChange={handleSingleFileChange} ref={singleFileInputRef} style={{ display: 'none' }} />
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <button onClick={triggerSingleFileInput} style={{ padding: '12px', backgroundColor: '#333', color: 'white', border: '1px solid #555', borderRadius: '6px', cursor: 'pointer', fontWeight: '500' }}>
                Browse File...
              </button>
              <button onClick={handleSingleUpload} disabled={!selectedFile || loading} style={{ padding: '12px', cursor: selectedFile ? 'pointer' : 'not-allowed', backgroundColor: loading ? '#555' : '#646cff', color: 'white', border: 'none', borderRadius: '6px', fontWeight: 'bold', fontSize: '15px' }}>
                {loading ? 'Analyzing...' : 'Predict'}
              </button>
            </div>

            {/* Version Description Box */}
            <div style={{ marginTop: '25px', padding: '15px', backgroundColor: '#1a1a1a', borderRadius: '8px', border: '1px solid #4caf50', fontSize: '14px', color: '#ccc', lineHeight: '1.5' }}>
              <h4 style={{ margin: '0 0 10px 0', color: '#4caf50', fontSize: '15px' }}>About Version 14 (Final Rerun)</h4>
              <p style={{ margin: '0 0 8px 0' }}>This is the finalized pipeline. Like v13, it uses per-class crop variance and 6 transforms + color. However, it fixes training stability by ensuring <strong>Batch Normalization (BN) layers stay frozen</strong> during the 100-epoch staged unfreezing.</p>
              <p style={{ margin: 0, color: '#aaa' }}><strong>How it differs:</strong> The BN fix prevents catastrophic forgetting, and everything here is verified via a statistically robust <strong>multi-seed (n=3) evaluation</strong>, making it the superior, highly reliable model.</p>
            </div>

          </div>

          {/* Right Column - Takes up remaining space securely */}
          <div style={{ flex: '1', minWidth: '400px', boxSizing: 'border-box' }}>
            {results && (
              <div style={{ marginBottom: '40px', backgroundColor: '#242424', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)' }}>
                <h3 style={{ marginTop: 0, color: '#fff', marginBottom: '20px' }}>Seed Predictions Overview</h3>
                <div style={{ width: '100%', height: '300px' }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                      <XAxis dataKey="name" stroke="#ccc" tick={{fontSize: 12}} />
                      <YAxis stroke="#ccc" tick={{fontSize: 12}} domain={[0, 100]} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar dataKey="Seed1" fill="#42a5f5" name="Seed 1" />
                      <Bar dataKey="Seed2" fill="#66bb6a" name="Seed 2" />
                      <Bar dataKey="Seed3" fill="#ffa726" name="Seed 3" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
            
            <h3 style={{ marginTop: '20px' }}>Live Predictions (v14 Checkpoints)</h3>
            
            {SEEDS.map(seed => (
              <div key={`seed-${seed}`} style={{ marginBottom: '40px' }}>
                <h4 style={{ color: '#fff', borderBottom: '1px solid #444', paddingBottom: '10px', marginBottom: '20px', fontSize: '20px' }}>
                  Seed {seed} Checkpoints
                </h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
                  {MODEL_TYPES.map(mt => {
                    const fullKey = `${mt.key_suffix} S${seed}`;
                    return (
                      <ModelCard 
                        key={fullKey}
                        modelKey={fullKey}
                        modelTitle={mt.title} 
                        data={results ? results[fullKey] : null} 
                        isFullWidth={false} 
                        heatmap={heatmaps[fullKey]}
                        loadingHeatmap={loadingHeatmaps[fullKey]}
                        onRequestHeatmap={handleRequestHeatmap}
                      />
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}


      {/* BATCH MODE */}
      {appMode === 'batch' && (
        <div style={{ marginTop: '30px', width: '100%' }}>
          
          <div style={{ backgroundColor: '#242424', padding: '20px', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.3)', marginBottom: '30px', width: '100%', boxSizing: 'border-box' }}>
            <h3 style={{ marginTop: 0, color: '#fff' }}>Input Multiple Images</h3>
            <input type="file" multiple accept="image/*" onChange={handleBatchFileChange} ref={batchFileInputRef} style={{ display: 'none' }} />
            
            <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
              <button onClick={triggerBatchFileInput} style={{ padding: '12px 20px', backgroundColor: '#333', color: 'white', border: '1px solid #555', borderRadius: '6px', cursor: 'pointer', fontWeight: 'bold' }}>
                Select Folder / Files
              </button>
              
              <button onClick={handleBatchUpload} disabled={batchFiles.length === 0 || batchLoading} style={{ padding: '12px 20px', cursor: batchFiles.length > 0 ? 'pointer' : 'not-allowed', backgroundColor: batchLoading ? '#555' : '#646cff', color: 'white', border: 'none', borderRadius: '6px', fontWeight: 'bold' }}>
                {batchLoading ? 'Processing...' : 'Run Batch Prediction'}
              </button>
              
              {batchFiles.length > 0 && <span style={{ color: '#aaa' }}>{batchFiles.length} files selected</span>}
            </div>

            {batchLoading && (
              <div style={{ marginTop: '15px', color: '#4caf50', fontWeight: 'bold' }}>
                Processed {batchProgress} of {batchFiles.length} images...
              </div>
            )}
          </div>

          <div style={{ backgroundColor: '#1a1a1a', borderRadius: '12px', padding: '20px', overflowX: 'auto', width: '100%', boxSizing: 'border-box', minHeight: '300px' }}>
            <h3 style={{ marginTop: 0, color: '#fff' }}>Batch Results</h3>
            
            {Object.keys(batchResults).length === 0 && !batchLoading && batchFiles.length === 0 ? (
               <div style={{ color: '#555', textAlign: 'center', padding: '40px' }}>Upload images to see batch results.</div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', minWidth: '800px' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid #444' }}>
                    <th style={{ padding: '12px', color: '#888' }}>Filename</th>
                    <th style={{ padding: '12px', color: '#888' }}>Model</th>
                    <th style={{ padding: '12px', color: '#888' }}>Prediction (Confidence %)</th>
                    {TARGET_CLASSES.map(cls => <th key={cls} style={{ padding: '12px', color: '#888', fontSize: '13px' }}>{cls}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(batchResults).map(([filename, dataObj]) => {
                    if (dataObj.error) {
                      return (
                        <tr key={filename} style={{ borderBottom: '1px solid #333' }}>
                          <td style={{ padding: '12px', fontWeight: 'bold' }}>{filename}</td>
                          <td colSpan={2 + TARGET_CLASSES.length} style={{ color: '#ff4444' }}>Error: {dataObj.error}</td>
                        </tr>
                      )
                    }

                    return DISPLAY_MODELS.map((model, index) => {
                      const modelData = dataObj[model.key]
                      if (!modelData) return null
                      
                      const isEndOfSeed = index % 4 === 3;
                      const isEndOfImage = index === DISPLAY_MODELS.length - 1;
                      
                      return (
                        <tr key={`${filename}-${model.key}`} style={{ borderBottom: isEndOfImage ? '2px solid #555' : (isEndOfSeed ? '1px solid #444' : '1px solid #2a2a2a'), backgroundColor: index % 2 === 0 ? '#1a1a1a' : '#1e1e1e' }}>
                          {index === 0 ? (
                            <td rowSpan={DISPLAY_MODELS.length} style={{ padding: '12px', fontWeight: 'bold', verticalAlign: 'top', borderRight: '1px solid #333', minWidth: '150px' }}>
                              <div style={{ marginBottom: '8px', wordBreak: 'break-all' }}>{filename}</div>
                              {batchFileUrls[filename] && (
                                <img 
                                  src={batchFileUrls[filename]} 
                                  alt={filename} 
                                  style={{ width: '100px', height: '100px', objectFit: 'cover', borderRadius: '6px', border: '1px solid #555' }} 
                                />
                              )}
                            </td>
                          ) : null}
                          <td style={{ padding: '12px', color: model.title.includes('Our Model') ? '#4caf50' : '#aaa' }}>{model.title}</td>
                          <td style={{ padding: '12px' }}>
                            <strong style={{ color: '#fff' }}>{modelData.prediction}</strong> 
                            <span style={{ color: '#888', marginLeft: '8px' }}>({(modelData.confidence * 100).toFixed(1)}%)</span>
                          </td>
                          {TARGET_CLASSES.map(cls => (
                            <td key={cls} style={{ padding: '12px', fontSize: '13px', color: '#ccc' }}>
                              {(modelData.all_scores[cls] * 100).toFixed(1)}%
                            </td>
                          ))}
                        </tr>
                      )
                    })
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default ClassifierV14;
