import React from 'react';
import { useNavigate } from 'react-router-dom';
import { CircuitBoard, Cpu } from 'lucide-react'; // Some icons for visual flair

function MainMenu() {
  const navigate = useNavigate();

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '80vh',
      fontFamily: 'sans-serif',
      padding: '20px'
    }}>
      <h1 style={{ fontSize: '36px', marginBottom: '10px' }}>E-Waste Classification System</h1>
      <p style={{ color: '#aaa', marginBottom: '40px', fontSize: '18px' }}>Select a model version to proceed</p>

      <div style={{
        display: 'flex',
        gap: '20px',
        flexWrap: 'wrap',
        justifyContent: 'center',
        maxWidth: '900px'
      }}>
        {/* Version 13 Card */}
        <div
          onClick={() => navigate('/v13')}
          style={{
            backgroundColor: '#242424',
            border: '1px solid #444',
            borderRadius: '12px',
            padding: '30px',
            width: '300px',
            cursor: 'pointer',
            transition: 'transform 0.2s, borderColor 0.2s',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            textAlign: 'center'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = 'translateY(-5px)';
            e.currentTarget.style.borderColor = '#646cff';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.borderColor = '#444';
          }}
        >
          <CircuitBoard size={48} color="#646cff" style={{ marginBottom: '20px' }} />
          <h2 style={{ margin: '0 0 10px 0', fontSize: '24px' }}>Version 13</h2>
          <p style={{ color: '#aaa', margin: '0', fontSize: '14px', lineHeight: '1.5' }}>
            Current System using the Baseline & Optimized DenseNet-121 Step B models.
          </p>
        </div>

        {/* Version 14 Card */}
        <div
          onClick={() => navigate('/v14')}
          style={{
            backgroundColor: '#242424',
            border: '1px solid #444',
            borderRadius: '12px',
            padding: '30px',
            width: '300px',
            cursor: 'pointer',
            transition: 'transform 0.2s, borderColor 0.2s',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            textAlign: 'center'
          }}
          onMouseOver={(e) => {
            e.currentTarget.style.transform = 'translateY(-5px)';
            e.currentTarget.style.borderColor = '#4caf50';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.transform = 'translateY(0)';
            e.currentTarget.style.borderColor = '#444';
          }}
        >
          <Cpu size={48} color="#4caf50" style={{ marginBottom: '20px' }} />
          <h2 style={{ margin: '0 0 10px 0', fontSize: '24px' }}>Version 14</h2>
          <p style={{ color: '#aaa', margin: '0', fontSize: '14px', lineHeight: '1.5' }}>
            New System with updated checkpoints (Coming Soon / Setup Pending).
          </p>
        </div>

        {/* Future Versions Placeholder */}
        <div
          style={{
            backgroundColor: '#1a1a1a',
            border: '1px dashed #444',
            borderRadius: '12px',
            padding: '30px',
            width: '300px',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            opacity: 0.6
          }}
        >
          <div style={{ fontSize: '32px', color: '#666', marginBottom: '10px' }}>+</div>
          <h2 style={{ margin: '0 0 10px 0', fontSize: '20px', color: '#888' }}>Add New Version</h2>
          <p style={{ color: '#666', margin: '0', fontSize: '14px' }}>
            Space for future model versions or architecture changes.
          </p>
        </div>
      </div>
    </div>
  );
}

export default MainMenu;
